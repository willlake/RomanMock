"""Slice-wise construction of analysis samples (stage B).

Everything here works on one slice of the parent catalog at a time, so the memory of stage B
does not grow with the size of the catalog (about 1.3 GB per worker) and it runs on a login
node. Quantities that need the whole catalog, the redshift histograms behind ``p_keep``, are
accumulated in passes over the slices:

1. histograms of the emitter populations (only with ``population_thinning``),
2. histograms of each sample after population thinning (only for samples with ``downsample``),
3. selection and writing, one file per slice and sample.

A pass that needs a redshift range which the configuration does not fix (a target without
bins, a downsampled sample without ``z_min``/``z_max``) is preceded by a pass collecting the
minimum and maximum. Random numbers are drawn per slice from ``(seed, stream, step)``, one
uniform deviate per galaxy, so every pass and any number of workers see the same selection.

The state handed to the workers (``plan``) is a plain dictionary of numbers and arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import opencosmo as oc

from .io import step_from_filename, write_dataset
from .lines import get_line_set
from .selection import SampleDefinition, probability_per_galaxy, sample_mask

THINNING_STREAM = 20_000       # rng stream of the population thinning
SAMPLE_STREAM = 10_000         # rng stream of sample k is SAMPLE_STREAM + k


def sample_slice_name(parent_file, sample: str) -> str:
    """``lc_cores-219.<tag>.hdf5`` -> ``lc_cores-219.<sample>.hdf5``."""
    return f"lc_cores-{step_from_filename(parent_file)}.{sample}.hdf5"


def thinning_keep(data: Mapping[str, np.ndarray], plan: dict, step: int) -> np.ndarray:
    """Population-thinning mask of one slice (all True without thinning)."""
    n = len(next(iter(data.values())))
    th = plan.get("thinning")
    keep = np.ones(n, dtype=bool)
    if th is None:
        return keep
    member = np.asarray(data[th["by"]]).astype(np.int64)
    z = np.asarray(data[th["z_column"]], dtype=np.float64)
    u = np.random.default_rng([plan["seed"], THINNING_STREAM, int(step)]).random(n)
    for pop in th["populations"].values():
        if pop.get("p_keep") is None:       # population without any galaxy
            continue
        edges = np.asarray(pop["edges"], dtype=np.float64)
        sel = member == pop["bit"]
        in_range = (z >= edges[0]) & (z <= edges[-1])
        p = probability_per_galaxy(z, edges, np.asarray(pop["p_keep"]))
        if th["outside_range"] == "keep":
            p = np.where(in_range, p, 1.0)
        keep[sel] = (u < p)[sel]
    if th["outside_range"] == "drop":
        keep[~np.isin(member, [pop["bit"] for pop in th["populations"].values()])] = False
    return keep


def _definition(sample: dict) -> SampleDefinition:
    return SampleDefinition(**sample["definition"])


def sample_keep(data: Mapping[str, np.ndarray], plan: dict, k: int, step: int, thin: np.ndarray,
                downsample: bool = True) -> np.ndarray:
    """Mask of sample ``k`` in one slice: cuts, population thinning and (optionally) the
    sample's own downsampling to its target dN/dz."""
    sample = plan["samples"][k]
    keep = sample_mask(data, get_line_set(plan["line_set"]), _definition(sample)) & thin
    if downsample and sample.get("p_keep") is not None:
        u = np.random.default_rng([plan["seed"], SAMPLE_STREAM + k, int(step)]).random(keep.size)
        p = probability_per_galaxy(data["redshift_assigned"], np.asarray(sample["edges"]), np.asarray(sample["p_keep"]))
        keep &= u < p
    return keep


def _minmax(z: np.ndarray) -> tuple[float, float]:
    z = z[np.isfinite(z)]
    return (float(z.min()), float(z.max())) if z.size else (np.inf, -np.inf)


def process_slice(plan: dict, path: str, tasks: dict) -> dict:
    """One pass over one slice of the parent catalog.

    ``tasks`` may contain ``pop_range`` / ``pop_hist`` (lists of population names),
    ``sample_range`` / ``sample_hist`` (lists of sample indices) and ``write`` (True).
    """
    step = step_from_filename(path)
    catalog = oc.open(path)
    data = catalog.select(*plan["columns"]).get_data("numpy", wrap_single=True)
    n = len(data["redshift_assigned"])
    out: dict = {"step": step, "n": n}
    th = plan.get("thinning")

    if tasks.get("pop_range") or tasks.get("pop_hist"):
        member = np.asarray(data[th["by"]]).astype(np.int64)
        z = np.asarray(data[th["z_column"]], dtype=np.float64)
        out["pop_range"], out["pop_hist"], out["pop_n"] = {}, {}, {}
        for name in tasks.get("pop_range", []):
            out["pop_range"][name] = _minmax(z[member == th["populations"][name]["bit"]])
        for name in tasks.get("pop_hist", []):
            pop = th["populations"][name]
            zs = z[member == pop["bit"]]
            out["pop_n"][name] = int(zs.size)
            out["pop_hist"][name] = np.histogram(zs[np.isfinite(zs)], bins=np.asarray(pop["edges"]))[0]

    if not (tasks.get("sample_range") or tasks.get("sample_hist") or tasks.get("write")):
        return out
    thin = thinning_keep(data, plan, step)
    z_assigned = np.asarray(data["redshift_assigned"], dtype=np.float64)
    out["sample_range"] = {k: _minmax(z_assigned[sample_keep(data, plan, k, step, thin, downsample=False)])
                           for k in tasks.get("sample_range", [])}
    out["sample_hist"] = {}
    for k in tasks.get("sample_hist", []):
        zs = z_assigned[sample_keep(data, plan, k, step, thin, downsample=False)]
        out["sample_hist"][k] = np.histogram(zs[np.isfinite(zs)], bins=np.asarray(plan["samples"][k]["edges"]))[0]

    if tasks.get("write"):
        if th is not None:
            member = np.asarray(data[th["by"]]).astype(np.int64)
            out["pop_after"] = {name: int(np.sum(thin & (member == pop["bit"]))) for name, pop in th["populations"].items()}
        out["written"] = {}
        for k, sample in enumerate(plan["samples"]):
            idx = np.where(sample_keep(data, plan, k, step, thin))[0].astype(np.int64)
            out["written"][k] = int(idx.size)
            if idx.size == 0:
                continue
            rows = catalog.take_rows(idx)
            if sample["columns"]:
                rows = rows.select(*sample["columns"])
            write_dataset(Path(sample["dir"]) / sample_slice_name(path, sample["name"]), rows, overwrite=True,
                          compress_index=plan["compress_index"])
    return out


def run_pass(plan: dict, files, tasks: dict, n_workers: int = 1) -> list[dict]:
    """Apply :func:`process_slice` to every parent file, serially or in a process pool."""
    files = [str(f) for f in files]
    if n_workers <= 1 or len(files) <= 1:
        return [process_slice(plan, f, tasks) for f in files]
    import concurrent.futures as cf
    import multiprocessing as mp

    with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(process_slice, plan, f, tasks) for f in files]
        return [f.result() for f in futures]

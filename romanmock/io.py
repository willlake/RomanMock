"""Catalog access through OpenCosmo and the diffsky loaders.

* :func:`list_catalog_files` / :func:`file_info` discover the ``lc_cores-*.diffsky_gals.hdf5``
  slices of a diffsky mock directory and their redshift ranges (header only, cheap).
* :func:`open_catalog` opens one or more slices as an ``opencosmo.Lightcone`` using the
  same open flags as ``diffsky.data_loaders.opencosmo_utils.load_diffsky_mock``.
* :func:`load_aux_data` returns the diffsky auxiliary data (SSP line wavelengths, ...)
  when diffsky is installed.
* :func:`write_dataset` writes with ``opencosmo.write`` so outputs remain OpenCosmo files
  (header, cosmology and spatial index preserved) and can be re-opened with ``oc.open``.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import hdf5plugin  # noqa: F401  (registers the compression filters used by the mock)
import numpy as np
import opencosmo as oc

DATA_FILE_GLOB = "lc_cores-*.diffsky_gals.hdf5"
DEFAULT_OPEN_KWARGS = {"synth_cores": False, "keep_top_host": True}
_STEP_RE = re.compile(r"lc_cores-(\d+)")


@dataclass(frozen=True)
class FileInfo:
    path: Path
    step: int
    z_min: float
    z_max: float
    n_gal: int

    def overlaps(self, z_min: float | None, z_max: float | None) -> bool:
        lo = -np.inf if z_min is None else z_min
        hi = np.inf if z_max is None else z_max
        return (self.z_max >= lo) and (self.z_min <= hi)


def step_from_filename(path) -> int:
    m = _STEP_RE.search(Path(path).name)
    if not m:
        raise ValueError(f"cannot infer step number from {path}")
    return int(m.group(1))


def list_catalog_files(catalog_dir) -> list[Path]:
    catalog_dir = Path(catalog_dir)
    files = sorted(catalog_dir.glob(DATA_FILE_GLOB), key=step_from_filename)
    if not files:
        raise FileNotFoundError(f"no {DATA_FILE_GLOB} files in {catalog_dir}")
    return files


def file_info(path) -> FileInfo:
    path = Path(path)
    with h5py.File(path, "r") as f:
        zr = f["header/lightcone"].attrs["z_range"]
        n = f["data/gal_id"].shape[0]
    return FileInfo(path, step_from_filename(path), float(zr[0]), float(zr[1]), int(n))


def select_files(
    catalog_dir,
    files: Sequence[str | Path] | None = None,
    steps: Iterable[int] | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
) -> list[Path]:
    """Choose lightcone slices by explicit list, step numbers and/or redshift overlap."""
    if files:
        paths = [Path(f) if Path(f).is_absolute() else Path(catalog_dir) / f for f in files]
    else:
        paths = list_catalog_files(catalog_dir)
    if steps is not None:
        steps = set(int(s) for s in steps)
        paths = [p for p in paths if step_from_filename(p) in steps]
    if z_min is not None or z_max is not None:
        paths = [p for p in paths if file_info(p).overlaps(z_min, z_max)]
    if not paths:
        raise FileNotFoundError("no lightcone files match the requested selection")
    return paths


def mock_name(path) -> str:
    with h5py.File(Path(path), "r") as f:
        return str(f["header/catalog_info"].attrs["mock_version_name"])


def open_catalog(files: Sequence[str | Path], **open_kwargs):
    """Open lightcone slices with OpenCosmo (same flags as diffsky's loader)."""
    kwargs = {**DEFAULT_OPEN_KWARGS, **open_kwargs}
    files = [Path(f) for f in files]
    return oc.open(*files, **kwargs)


def load_aux_data(catalog_dir, name: str | None = None):
    """diffsky auxiliary data (``ssp_data`` with ``ssp_emline_wave``, ``tcurves``,
    ``t_table``, ``param_collection``); ``None`` if diffsky is unavailable."""
    try:
        from diffsky.data_loaders.hacc_utils.lc_mock import (
            load_diffsky_param_collection_merging,
            load_diffsky_ssp_data,
            load_diffsky_t_table,
            load_diffsky_tcurves,
        )
    except ImportError:
        warnings.warn("diffsky is not installed; auxiliary SSP data not loaded", stacklevel=2)
        return None
    catalog_dir = Path(catalog_dir)
    name = name or mock_name(list_catalog_files(catalog_dir)[0])
    try:
        return dict(
            ssp_data=load_diffsky_ssp_data(catalog_dir, name),
            tcurves=load_diffsky_tcurves(catalog_dir, name),
            t_table=load_diffsky_t_table(catalog_dir, name),
            param_collection=load_diffsky_param_collection_merging(catalog_dir, name),
        )
    except (FileNotFoundError, OSError, KeyError) as err:
        warnings.warn(f"could not load diffsky auxiliary data: {err}", stacklevel=2)
        return None


def load_diffsky_mock(catalog_dir, **kwargs):
    """Thin wrapper over ``diffsky.data_loaders.opencosmo_utils.load_diffsky_mock``
    (opens every slice). Returns ``(lightcone, aux_data)``."""
    from diffsky.data_loaders.opencosmo_utils import load_diffsky_mock as _load

    return _load(catalog_dir, **kwargs)


def iter_slices(catalog):
    """Yield ``(step, dataset)`` for each redshift slice of a Lightcone (or the dataset itself)."""
    if isinstance(catalog, oc.Lightcone):
        for step, ds in catalog.items():
            yield step, ds
    else:
        yield getattr(catalog.header.file, "step", 0), catalog


def write_dataset(path, dataset, overwrite: bool = False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    oc.write(path, dataset, overwrite=overwrite)
    return path


def output_filename(input_path, output_dir, tag: str = "romanmock") -> Path:
    """``lc_cores-100.diffsky_gals.hdf5`` -> ``<output_dir>/lc_cores-100.<tag>.hdf5``."""
    name = Path(input_path).name.replace("diffsky_gals", tag)
    return Path(output_dir) / name


def write_run_metadata(path, metadata: dict):
    """Store run metadata (config, bit ordering, outcome codes) as a YAML sidecar."""
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    return path


# ---------------------------------------------------------------------------------
# Provenance: manifest of completed slices and in-file configuration stamps
# ---------------------------------------------------------------------------------
def manifest_path(output_dir, tag: str) -> Path:
    return Path(output_dir) / f"{tag}_manifest.yaml"


def read_manifest(output_dir, tag: str) -> dict | None:
    import yaml

    path = manifest_path(output_dir, tag)
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f) or {}


def write_manifest(output_dir, tag: str, manifest: dict) -> Path:
    """Write the manifest atomically (temp file + rename)."""
    import os
    import yaml

    path = manifest_path(output_dir, tag)
    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    os.replace(tmp, path)
    return path


def stamp_output(path, attrs: dict):
    """Store provenance attributes on the root of a written HDF5 file."""
    with h5py.File(Path(path), "r+") as f:
        for k, v in attrs.items():
            f.attrs[k] = v


def read_stamp(path) -> dict:
    with h5py.File(Path(path), "r") as f:
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items() if k.startswith("romanmock_")}


# ---------------------------------------------------------------------------------
# Resource helpers
# ---------------------------------------------------------------------------------
PER_WORKER_GB = 5.0   # measured peak of one stage-A worker on the largest diffsky slice: 4.4 GB


def memory_budget_gb() -> float:
    """Memory (GB) this process tree may still use: the minimum of the node's available
    memory and the headroom of every enclosing cgroup that sets a limit (NERSC login
    nodes cap each user at 30 GB through such a cgroup). Falls back to 8 GB if unknown."""
    budget = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    budget = int(line.split()[1]) / 1e6
    except OSError:
        pass
    try:
        with open("/proc/self/cgroup") as f:
            path = f.read().strip().split(":")[-1]
        while path and path != "/":
            base = Path("/sys/fs/cgroup") / path.lstrip("/")
            try:
                limit = (base / "memory.max").read_text().strip()
                if limit != "max":
                    headroom = (int(limit) - int((base / "memory.current").read_text())) / 1e9
                    budget = headroom if budget is None else min(budget, headroom)
            except OSError:
                pass
            path = str(Path(path).parent)
    except OSError:
        pass
    return 8.0 if budget is None else max(budget, 0.0)


def suggest_n_workers(max_workers: int = 32, per_worker_gb: float = PER_WORKER_GB, reserve_gb: float = 2.0) -> int:
    """Number of stage-A worker processes that fit in :func:`memory_budget_gb`, capped by
    ``max_workers`` and the CPUs this process may use."""
    import os

    try:
        n_cpu = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover
        n_cpu = os.cpu_count() or 1
    by_memory = int((memory_budget_gb() - reserve_gb) / per_worker_gb)
    return max(1, min(max_workers, n_cpu, by_memory))

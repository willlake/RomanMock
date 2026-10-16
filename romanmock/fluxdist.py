"""Distribution of the line fluxes of the whole mock as a function of redshift.

For every line of the line set, the observed flux (dust attenuation included, ``D_L`` at the
cosmological redshift) of every galaxy of the mock is histogrammed in ``log10 F`` per redshift
bin, with no flux cut at all. The histograms give the quantiles of the flux at each redshift
and the fraction of galaxies above any threshold: the quantities that explain the dN/dz of
the emitters above a flux limit (:mod:`romanmock.census`) and the single-line fractions.

One slice per worker process, like the census; ``romanmock flux-distribution`` writes the
result to a file that :meth:`FluxDistribution.load` reads back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .fluxes import feature_fluxes
from .io import DEFAULT_OPEN_KWARGS, file_info, open_catalog, select_files
from .lines import LineSet, get_line_set

DEFAULT_Z_EDGES = np.round(np.arange(0.0, 4.0 + 1e-9, 0.1), 6)
DEFAULT_LOGF_EDGES = np.round(np.arange(-21.0, -11.0 + 1e-9, 0.02), 6)


@dataclass
class FluxDistribution:
    """Histograms of ``log10`` line flux per redshift bin.

    ``counts[i, j, k]``: galaxies with line ``i`` in redshift bin ``j`` and flux bin ``k``;
    ``below[i, j]`` / ``above[i, j]``: galaxies with a flux under the first edge (including
    zero flux) or over the last; ``n_gal[j]``: galaxies in the redshift bin.
    """

    line_names: list[str]
    z_edges: np.ndarray
    logf_edges: np.ndarray
    z_column: str = "redshift_true"
    counts: np.ndarray = None
    below: np.ndarray = None
    above: np.ndarray = None
    n_gal: np.ndarray = None
    steps: list[int] = field(default_factory=list)

    @classmethod
    def empty(cls, z_edges, logf_edges, line_set, z_column: str = "redshift_true") -> "FluxDistribution":
        line_set = get_line_set(line_set)
        z_edges, logf_edges = np.asarray(z_edges, dtype=np.float64), np.asarray(logf_edges, dtype=np.float64)
        shape = (len(line_set), z_edges.size - 1, logf_edges.size - 1)
        return cls(list(line_set.names), z_edges, logf_edges, z_column, np.zeros(shape, dtype=np.int64),
                   np.zeros(shape[:2], dtype=np.int64), np.zeros(shape[:2], dtype=np.int64),
                   np.zeros(shape[1], dtype=np.int64))

    def add(self, counts, below, above, n_gal, step: int) -> "FluxDistribution":
        self.counts += counts
        self.below += below
        self.above += above
        self.n_gal += n_gal
        self.steps.append(int(step))
        return self

    def rebin(self, step: int) -> "FluxDistribution":
        """The same histograms with ``step`` consecutive redshift bins merged (trailing bins dropped)."""
        step = int(step)
        nz = (self.z_edges.size - 1) // step * step
        merge = lambda a: a[..., :nz, :].reshape(*a.shape[:-2], nz // step, step, a.shape[-1]).sum(axis=-2) if a.ndim == 3 \
            else a[..., :nz].reshape(*a.shape[:-1], nz // step, step).sum(axis=-1)
        return FluxDistribution(list(self.line_names), self.z_edges[: nz + 1 : step], self.logf_edges.copy(), self.z_column,
                                merge(self.counts), merge(self.below), merge(self.above), merge(self.n_gal), list(self.steps))

    # ---- access -------------------------------------------------------------------------
    @property
    def z(self) -> np.ndarray:
        """Centres of the redshift bins."""
        return 0.5 * (self.z_edges[1:] + self.z_edges[:-1])

    def index(self, line: str) -> int:
        if line not in self.line_names:
            raise KeyError(f"unknown line {line!r}; the lines are {self.line_names}")
        return self.line_names.index(line)

    def quantile(self, line: str, q: float) -> np.ndarray:
        """Flux [erg/s/cm^2] below which a fraction ``q`` of the galaxies of each redshift bin lies.

        Galaxies under the first flux edge count as fainter than every bin, so a quantile that falls
        among them is reported as the first edge, and one that falls beyond the last edge as that edge.
        The flux is interpolated linearly in ``log10 F`` inside the bin.
        """
        i = self.index(line)
        n = self.below[i] + self.counts[i].sum(axis=1) + self.above[i]
        cum = np.concatenate([self.below[i][:, None], self.below[i][:, None] + np.cumsum(self.counts[i], axis=1)], axis=1)
        target = q * n
        out = np.full(n.size, np.nan)
        for j in range(n.size):
            if n[j] == 0:
                continue
            k = np.searchsorted(cum[j], target[j], side="left")        # first edge with cum >= target
            if k == 0:
                out[j] = self.logf_edges[0]
            elif k > self.logf_edges.size - 1:
                out[j] = self.logf_edges[-1]
            else:
                lo, hi = cum[j][k - 1], cum[j][k]
                frac = (target[j] - lo) / (hi - lo) if hi > lo else 0.0
                out[j] = self.logf_edges[k - 1] + frac * (self.logf_edges[k] - self.logf_edges[k - 1])
        return 10.0 ** out

    def fraction_above(self, line: str, flux: float) -> np.ndarray:
        """Fraction of the galaxies of each redshift bin with the line brighter than ``flux``."""
        i = self.index(line)
        logf = np.log10(flux)
        k = np.searchsorted(self.logf_edges, logf, side="right") - 1
        n = self.below[i] + self.counts[i].sum(axis=1) + self.above[i]
        if k < 0:
            n_above = self.counts[i].sum(axis=1) + self.above[i]
        elif k >= self.logf_edges.size - 1:
            n_above = self.above[i].astype(np.float64)
        else:
            frac = (self.logf_edges[k + 1] - logf) / (self.logf_edges[k + 1] - self.logf_edges[k])
            n_above = self.counts[i][:, k + 1:].sum(axis=1) + frac * self.counts[i][:, k] + self.above[i]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, n_above / n, np.nan)

    def number_above(self, line: str, flux: float) -> np.ndarray:
        """Galaxies of each redshift bin with the line brighter than ``flux``."""
        n = self.below[self.index(line)] + self.counts[self.index(line)].sum(axis=1) + self.above[self.index(line)]
        return self.fraction_above(line, flux) * n

    # ---- disk ---------------------------------------------------------------------------
    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, line_names=np.array(self.line_names), z_edges=self.z_edges, logf_edges=self.logf_edges,
                            z_column=np.array(self.z_column), counts=self.counts, below=self.below, above=self.above,
                            n_gal=self.n_gal, steps=np.array(sorted(self.steps), dtype=np.int64))
        return path

    @classmethod
    def load(cls, path) -> "FluxDistribution":
        with np.load(path) as f:
            return cls([str(x) for x in f["line_names"]], f["z_edges"], f["logf_edges"], str(f["z_column"]), f["counts"],
                       f["below"], f["above"], f["n_gal"], [int(s) for s in f["steps"]])


# ---------------------------------------------------------------------------------
# array and file level
# ---------------------------------------------------------------------------------
def flux_histograms(data: Mapping[str, np.ndarray], line_set, cosmology, z_edges, logf_edges, z_column="redshift_true",
                    backend: str = "astropy"):
    """Histograms of one dict of catalog arrays: ``(counts, below, above, n_gal)`` as in :class:`FluxDistribution`."""
    line_set = get_line_set(line_set)
    z_edges, logf_edges = np.asarray(z_edges, dtype=np.float64), np.asarray(logf_edges, dtype=np.float64)
    z = np.asarray(data[z_column], dtype=np.float64)
    nz, nf = z_edges.size - 1, logf_edges.size - 1
    jz = np.digitize(z, z_edges) - 1
    in_range = (jz >= 0) & (jz < nz) & (z < z_edges[-1])
    n_gal = np.bincount(jz[in_range], minlength=nz).astype(np.int64)
    flux = feature_fluxes(data, line_set, np.asarray(data["redshift_true"], dtype=np.float64), cosmology, backend=backend)
    counts = np.zeros((len(line_set), nz, nf), dtype=np.int64)
    below = np.zeros((len(line_set), nz), dtype=np.int64)
    above = np.zeros((len(line_set), nz), dtype=np.int64)
    with np.errstate(divide="ignore"):
        logf = np.log10(np.where(flux > 0, flux, np.nan))
    for i in range(len(line_set)):
        f = logf[in_range, i]
        j = jz[in_range]
        under = ~(f >= logf_edges[0])                              # zero flux (NaN) or under the first edge
        over = f >= logf_edges[-1]
        below[i] = np.bincount(j[under], minlength=nz)
        above[i] = np.bincount(j[over], minlength=nz)
        ok = ~under & ~over
        k = np.digitize(f[ok], logf_edges) - 1
        np.add.at(counts[i], (j[ok], k), 1)
    return counts, below, above, n_gal


def _one_file(spec: dict, path: str):
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    line_set = get_line_set(spec["line_set"])
    info = file_info(path)
    catalog = open_catalog([path], **spec["open_kwargs"])
    cols = list(dict.fromkeys([spec["z_column"], "redshift_true", *line_set.required_columns]))
    total = None
    for ds in (catalog.values() if hasattr(catalog, "values") else [catalog]):
        data = ds.select(*cols).get_data("numpy", wrap_single=True)
        parts = flux_histograms(data, line_set, ds.cosmology, spec["z_edges"], spec["logf_edges"], spec["z_column"], spec["backend"])
        total = parts if total is None else tuple(a + b for a, b in zip(total, parts))
    return info.step, info.n_gal, total


def flux_distribution(
    catalog_dir,
    z_edges=DEFAULT_Z_EDGES,
    logf_edges=DEFAULT_LOGF_EDGES,
    line_set="roman_transitions",
    z_column: str = "redshift_true",
    backend: str = "astropy",
    files: Sequence[str | Path] | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    n_workers: int = 1,
    open_kwargs: Mapping | None = None,
    verbose: bool = True,
) -> FluxDistribution:
    """Flux histograms of a diffsky lightcone directory (or a subset of its slices), one slice per worker."""
    z_edges = np.asarray(z_edges, dtype=np.float64)
    paths = select_files(catalog_dir, files=files, z_min=float(z_edges[0]) if z_min is None else z_min,
                         z_max=float(z_edges[-1]) if z_max is None else z_max)
    line_set = get_line_set(line_set)
    result = FluxDistribution.empty(z_edges, logf_edges, line_set, z_column)
    spec = {"line_set": line_set.to_config(), "z_edges": z_edges.tolist(), "logf_edges": np.asarray(logf_edges).tolist(),
            "z_column": z_column, "backend": backend, "open_kwargs": {**DEFAULT_OPEN_KWARGS, **(open_kwargs or {})}}

    def record(step, n_gal, parts):
        result.add(*parts, step)
        if verbose:
            print(f"[flux-distribution] step {step:4d}  n={n_gal:,}", flush=True)

    if n_workers <= 1:
        for p in paths:
            record(*_one_file(spec, str(p)))
    else:
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
            for fut in cf.as_completed([pool.submit(_one_file, spec, str(p)) for p in paths]):
                record(*fut.result())
    result.steps.sort()
    return result

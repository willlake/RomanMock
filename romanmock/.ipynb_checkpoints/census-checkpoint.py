"""Ideal-detection census: ``dN/dz`` of galaxies whose emission lines exceed a flux limit
inside the grism bandpass, with no line confusion, no interlopers and no downsampling.

This is the population the Wang et al. (2022) tables describe ("true emitters above the
flux limit, observational efficiency not included"), so the census is the right object to
compare with :mod:`romanmock.reference_dndz`. It reuses the pipeline's flux and detection
code (:func:`romanmock.fluxes.feature_fluxes`, :func:`romanmock.detection.detect_lines`)
and never runs the confusion step.

Galaxies are counted per line and split by the number of lines of the line set that pass
the cut: *exactly one* (a single-line galaxy, whose redshift would rely on line
identification) or *more than one* (a multi-line galaxy). Several flux limits are
evaluated in one pass over the catalog, one slice per worker process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .detection import detect_lines
from .fluxes import component_fluxes, feature_fluxes
from .io import DEFAULT_OPEN_KWARGS, file_info, open_catalog, select_files
from .lines import LineSet, get_line_set
from .roman import GRISM_WAVE_RANGE_ANGSTROM
from .sensitivity import ConstantFluxLimit, ThresholdDetection

#: index of the last axis-2 slot in :attr:`LineCensus.counts`
SINGLE, MULTI = 0, 1
CLASSES = ("single", "multi")


@dataclass
class LineCensus:
    """Histograms of galaxies with lines above ``flux_limits`` inside ``wave_range``.

    ``counts[i, j, SINGLE, k]`` is the number of galaxies in redshift bin ``k`` whose line
    ``j`` passes flux limit ``i`` and that have no other line passing it;
    ``counts[i, j, MULTI, k]`` counts those with at least one other passing line.
    ``n_all[k]`` is the number of catalog galaxies in bin ``k`` before any cut.
    """

    flux_limits: np.ndarray            # (n_lim,) erg/s/cm^2
    z_edges: np.ndarray                # (n_bins + 1,)
    line_names: list[str]
    counts: np.ndarray                 # (n_lim, n_lines, 2, n_bins) int64
    n_all: np.ndarray                  # (n_bins,) int64
    wave_range: tuple[float, float] = GRISM_WAVE_RANGE_ANGSTROM
    z_column: str = "redshift_obs"
    steps: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------ construction
    @classmethod
    def empty(cls, flux_limits, z_edges, line_set, wave_range=GRISM_WAVE_RANGE_ANGSTROM, z_column="redshift_obs"):
        line_set = get_line_set(line_set)
        flux_limits = np.atleast_1d(np.asarray(flux_limits, dtype=np.float64))
        z_edges = np.asarray(z_edges, dtype=np.float64)
        counts = np.zeros((flux_limits.size, len(line_set), 2, z_edges.size - 1), dtype=np.int64)
        return cls(flux_limits, z_edges, list(line_set.names), counts, np.zeros(z_edges.size - 1, dtype=np.int64),
                   (float(wave_range[0]), float(wave_range[1])), z_column, [])

    def add(self, counts: np.ndarray, n_all: np.ndarray, step: int | None = None) -> "LineCensus":
        self.counts += np.asarray(counts, dtype=np.int64)
        self.n_all += np.asarray(n_all, dtype=np.int64)
        if step is not None:
            self.steps.append(int(step))
        return self

    # ------------------------------------------------------------------ lookups
    @property
    def z_centers(self) -> np.ndarray:
        return 0.5 * (self.z_edges[1:] + self.z_edges[:-1])

    @property
    def n_bins(self) -> int:
        return self.z_edges.size - 1

    def limit_index(self, flux_limit: float) -> int:
        j = np.where(np.isclose(self.flux_limits, flux_limit, rtol=1e-6, atol=0.0))[0]
        if j.size == 0:
            raise ValueError(f"flux limit {flux_limit:.3g} not in census {self.flux_limits}")
        return int(j[0])

    def line_index(self, line: str) -> int:
        try:
            return self.line_names.index(line)
        except ValueError:
            raise KeyError(f"line {line!r} not in census; available: {self.line_names}") from None

    def hist(self, line: str, flux_limit: float, which: str = "all") -> np.ndarray:
        """Raw counts per redshift bin; ``which`` is ``"all"``, ``"single"`` or ``"multi"``."""
        c = self.counts[self.limit_index(flux_limit), self.line_index(line)]
        if which == "all":
            return c.sum(axis=0)
        if which in CLASSES:
            return c[CLASSES.index(which)]
        raise ValueError(f"which must be 'all', 'single' or 'multi'; got {which!r}")

    def dndz(self, line: str, flux_limit: float, area_deg2: float, which: str = "all") -> np.ndarray:
        """``dN/dz`` per deg^2 in the census bins."""
        return self.hist(line, flux_limit, which) / (np.diff(self.z_edges) * float(area_deg2))

    def integrated(self, line: str, flux_limit: float, area_deg2: float, which: str = "all",
                   z_range: tuple[float, float] | None = None) -> float:
        """Galaxies per deg^2 summed over the bins inside ``z_range`` (all bins by default)."""
        h = self.hist(line, flux_limit, which)
        if z_range is not None:
            lo, hi = z_range
            keep = (self.z_edges[:-1] >= lo - 1e-9) & (self.z_edges[1:] <= hi + 1e-9)
            h = h[keep]
        return float(h.sum() / float(area_deg2))

    def single_fraction(self, line: str, flux_limit: float) -> np.ndarray:
        """Fraction of the line's galaxies per bin that have no other line above the limit."""
        s = self.hist(line, flux_limit, "single").astype(np.float64)
        tot = s + self.hist(line, flux_limit, "multi")
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(tot > 0, s / tot, np.nan)

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> Path:
        path = Path(path)
        np.savez(path, flux_limits=self.flux_limits, z_edges=self.z_edges, line_names=np.array(self.line_names),
                 counts=self.counts, n_all=self.n_all, wave_range=np.array(self.wave_range),
                 z_column=np.array(self.z_column), steps=np.array(self.steps, dtype=np.int64))
        return path

    @classmethod
    def load(cls, path) -> "LineCensus":
        with np.load(Path(path)) as f:
            return cls(f["flux_limits"], f["z_edges"], [str(s) for s in f["line_names"]], f["counts"], f["n_all"],
                       tuple(float(w) for w in f["wave_range"]), str(f["z_column"]), [int(s) for s in f["steps"]])


# ---------------------------------------------------------------------------------
# array level
# ---------------------------------------------------------------------------------
def census_arrays(
    data: Mapping[str, np.ndarray],
    line_set,
    cosmology,
    flux_limits: Sequence[float],
    z_edges: np.ndarray,
    wave_range: tuple[float, float] = GRISM_WAVE_RANGE_ANGSTROM,
    z_column: str = "redshift_obs",
    z_dl_column: str = "redshift_true",
    backend: str = "astropy",
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram one dict of catalog arrays; returns ``(counts, n_all)`` as in :class:`LineCensus`.

    Fluxes use ``D_L(z_dl_column)`` and the bandpass test uses ``z_column`` (the redshift
    with peculiar velocity), exactly as the pipeline does; the histograms are binned in
    ``z_column`` as well.
    """
    line_set = get_line_set(line_set)
    z_edges = np.asarray(z_edges, dtype=np.float64)
    flux_limits = np.atleast_1d(np.asarray(flux_limits, dtype=np.float64))
    z_dl = np.asarray(data[z_dl_column], dtype=np.float64)
    z_rsd = np.asarray(data[z_column], dtype=np.float64)
    n_bins = z_edges.size - 1

    idx = np.digitize(z_rsd, z_edges) - 1
    in_range = (idx >= 0) & (idx < n_bins) & (z_rsd < z_edges[-1])
    n_all = np.bincount(idx[in_range], minlength=n_bins).astype(np.int64)

    flux = feature_fluxes(data, line_set, z_dl, cosmology, backend=backend)
    comp = component_fluxes(data, line_set, z_dl, cosmology, backend=backend) if line_set.has_separated_components else None
    counts = np.zeros((flux_limits.size, len(line_set), 2, n_bins), dtype=np.int64)
    for i, lim in enumerate(flux_limits):
        model = ThresholdDetection(ConstantFluxLimit(float(lim), wave_range))
        det = detect_lines(flux, z_rsd, line_set, model, component_flux=comp)
        single = in_range & (det.n_det == 1)
        multi = in_range & (det.n_det > 1)
        for j in range(len(line_set)):
            hit = det.detected[:, j]
            counts[i, j, SINGLE] = np.bincount(idx[hit & single], minlength=n_bins)
            counts[i, j, MULTI] = np.bincount(idx[hit & multi], minlength=n_bins)
    return counts, n_all


# ---------------------------------------------------------------------------------
# file level
# ---------------------------------------------------------------------------------
def _census_one_file(spec: dict, path: str) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Process-pool worker: census of one slice file from a serialisable ``spec``."""
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    line_set = get_line_set(spec["line_set"])
    info = file_info(path)
    catalog = open_catalog([path], **spec["open_kwargs"])
    cols = [spec["z_column"], spec["z_dl_column"], *line_set.required_columns]
    counts = None
    n_all = None
    for ds in (catalog.values() if hasattr(catalog, "values") else [catalog]):
        data = ds.select(*cols).get_data("numpy", wrap_single=True)
        c, n = census_arrays(data, line_set, ds.cosmology, spec["flux_limits"], np.asarray(spec["z_edges"]),
                             tuple(spec["wave_range"]), spec["z_column"], spec["z_dl_column"], spec["backend"])
        counts = c if counts is None else counts + c
        n_all = n if n_all is None else n_all + n
    return info.step, info.n_gal, counts, n_all


def line_census(
    catalog_dir,
    flux_limits: Sequence[float],
    z_edges: np.ndarray,
    line_set="roman_transitions",
    wave_range: tuple[float, float] = GRISM_WAVE_RANGE_ANGSTROM,
    z_column: str = "redshift_obs",
    z_dl_column: str = "redshift_true",
    backend: str = "astropy",
    files: Sequence[str | Path] | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    n_workers: int = 1,
    open_kwargs: Mapping | None = None,
    verbose: bool = True,
) -> LineCensus:
    """Census of a diffsky lightcone directory (or a subset of its slices).

    Slices are chosen like :meth:`RomanMockPipeline.run` (``files``, ``z_min``, ``z_max``);
    by default ``z_min``/``z_max`` are taken from ``z_edges``. With ``n_workers > 1`` one
    slice is processed per worker process (peak ~2-3 GB per worker on the largest slices).
    """
    z_edges = np.asarray(z_edges, dtype=np.float64)
    if z_min is None:
        z_min = float(z_edges[0])
    if z_max is None:
        z_max = float(z_edges[-1])
    paths = select_files(catalog_dir, files=files, z_min=z_min, z_max=z_max)
    line_set = get_line_set(line_set)
    census = LineCensus.empty(flux_limits, z_edges, line_set, wave_range, z_column)
    spec = {
        "line_set": line_set.to_config(),
        "flux_limits": [float(f) for f in census.flux_limits],
        "z_edges": z_edges.tolist(),
        "wave_range": [float(w) for w in wave_range],
        "z_column": z_column,
        "z_dl_column": z_dl_column,
        "backend": backend,
        "open_kwargs": {**DEFAULT_OPEN_KWARGS, **(open_kwargs or {})},
    }

    def record(step, n_gal, counts, n_all):
        census.add(counts, n_all, step)
        if verbose:
            n_det = counts[0].sum(axis=(1, 2)).max() if counts.size else 0
            print(f"[census] step {step:4d}  n={n_gal:,}  in z range {int(n_all.sum()):,}  "
                  f"most populous line at F>{census.flux_limits[0]:.2g}: {int(n_det):,}", flush=True)

    if n_workers <= 1:
        for p in paths:
            record(*_census_one_file(spec, str(p)))
        return census

    import concurrent.futures as cf
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = [pool.submit(_census_one_file, spec, str(p)) for p in paths]
        for fut in cf.as_completed(futures):
            record(*fut.result())
    census.steps.sort()
    return census

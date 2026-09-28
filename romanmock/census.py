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

A limit is a single threshold applied to every line (a number; the population of the
Wang et al. tables), or a pair ``(primary, secondary)`` for the pipeline's two-tier rule:
a galaxy needs one line above ``primary``, and its other lines then count above the lower
``secondary``. With a pair, the single-line galaxies of a line are those with that line
above ``primary`` and no other line above ``secondary``.
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


def normalize_limits(flux_limits) -> tuple[np.ndarray, np.ndarray]:
    """``(primary, secondary)`` arrays from numbers and/or ``(primary, secondary)`` pairs; a
    number or a pair with ``None`` is a single threshold (secondary = primary)."""
    if np.isscalar(flux_limits):
        flux_limits = [flux_limits]
    primary, secondary = [], []
    for lim in flux_limits:
        if np.isscalar(lim):
            lo = hi = float(lim)
        else:
            hi = float(lim[0])
            lo = hi if lim[1] is None or not np.isfinite(lim[1]) else float(lim[1])
        if not 0 < lo <= hi:
            raise ValueError(f"need 0 < secondary <= primary; got primary {hi:.3g}, secondary {lo:.3g}")
        primary.append(hi)
        secondary.append(lo)
    return np.array(primary), np.array(secondary)


@dataclass
class LineCensus:
    """Histograms of galaxies with lines above ``flux_limits`` inside ``wave_range``.

    ``counts[i, j, SINGLE, k]`` is the number of galaxies in redshift bin ``k`` whose line
    ``j`` passes flux limit ``i`` and that have no other line passing it;
    ``counts[i, j, MULTI, k]`` counts those with at least one other passing line.
    ``n_all[k]`` is the number of catalog galaxies in bin ``k`` before any cut, and
    ``n_any[i, SINGLE/MULTI, k]`` the number of *distinct* galaxies with exactly one /
    more than one passing line (each galaxy once, whichever lines pass).
    """

    flux_limits: np.ndarray            # (n_lim,) erg/s/cm^2, primary limits
    z_edges: np.ndarray                # (n_bins + 1,)
    line_names: list[str]
    counts: np.ndarray                 # (n_lim, n_lines, 2, n_bins) int64
    n_all: np.ndarray                  # (n_bins,) int64
    n_any: np.ndarray                  # (n_lim, 2, n_bins) int64
    wave_range: tuple[float, float] = GRISM_WAVE_RANGE_ANGSTROM
    z_column: str = "redshift_obs"
    steps: list[int] = field(default_factory=list)
    secondary_limits: np.ndarray | None = None   # (n_lim,) limit for the other lines; equal to flux_limits = single tier

    def __post_init__(self):
        self.flux_limits = np.atleast_1d(np.asarray(self.flux_limits, dtype=np.float64))
        self.secondary_limits = self.flux_limits.copy() if self.secondary_limits is None \
            else np.atleast_1d(np.asarray(self.secondary_limits, dtype=np.float64))

    @property
    def limits(self) -> list[tuple[float, float | None]]:
        """The limits as ``(primary, secondary)`` pairs, ``secondary`` None for a single threshold."""
        return [(float(p), None if s == p else float(s)) for p, s in zip(self.flux_limits, self.secondary_limits)]

    # ------------------------------------------------------------------ construction
    @classmethod
    def empty(cls, flux_limits, z_edges, line_set, wave_range=GRISM_WAVE_RANGE_ANGSTROM, z_column="redshift_obs"):
        line_set = get_line_set(line_set)
        flux_limits, secondary = normalize_limits(flux_limits)
        z_edges = np.asarray(z_edges, dtype=np.float64)
        n_bins = z_edges.size - 1
        counts = np.zeros((flux_limits.size, len(line_set), 2, n_bins), dtype=np.int64)
        return cls(flux_limits, z_edges, list(line_set.names), counts, np.zeros(n_bins, dtype=np.int64),
                   np.zeros((flux_limits.size, 2, n_bins), dtype=np.int64),
                   (float(wave_range[0]), float(wave_range[1])), z_column, [], secondary)

    def add(self, counts: np.ndarray, n_all: np.ndarray, n_any: np.ndarray, step: int | None = None) -> "LineCensus":
        self.counts += np.asarray(counts, dtype=np.int64)
        self.n_all += np.asarray(n_all, dtype=np.int64)
        self.n_any += np.asarray(n_any, dtype=np.int64)
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

    def limit_index(self, flux_limit: float, secondary: float | None = None) -> int:
        """Index of the limit ``flux_limit`` with secondary limit ``secondary`` (None: single threshold)."""
        second = flux_limit if secondary is None else secondary
        j = np.where(np.isclose(self.flux_limits, flux_limit, rtol=1e-6, atol=0.0)
                     & np.isclose(self.secondary_limits, second, rtol=1e-6, atol=0.0))[0]
        if j.size == 0:
            raise ValueError(f"limit (primary {flux_limit:.3g}, secondary {secondary}) not in census {self.limits}")
        return int(j[0])

    def line_index(self, line: str) -> int:
        try:
            return self.line_names.index(line)
        except ValueError:
            raise KeyError(f"line {line!r} not in census; available: {self.line_names}") from None

    def hist(self, line: str, flux_limit: float, which: str = "all", secondary: float | None = None) -> np.ndarray:
        """Raw counts per redshift bin; ``which`` is ``"all"``, ``"single"`` or ``"multi"``."""
        c = self.counts[self.limit_index(flux_limit, secondary), self.line_index(line)]
        if which == "all":
            return c.sum(axis=0)
        if which in CLASSES:
            return c[CLASSES.index(which)]
        raise ValueError(f"which must be 'all', 'single' or 'multi'; got {which!r}")

    def dndz(self, line: str, flux_limit: float, area_deg2: float, which: str = "all",
             secondary: float | None = None) -> np.ndarray:
        """``dN/dz`` per deg^2 in the census bins."""
        return self.hist(line, flux_limit, which, secondary) / (np.diff(self.z_edges) * float(area_deg2))

    def integrated(self, line: str, flux_limit: float, area_deg2: float, which: str = "all",
                   z_range: tuple[float, float] | None = None, secondary: float | None = None) -> float:
        """Galaxies per deg^2 summed over the bins inside ``z_range`` (all bins by default)."""
        h = self.hist(line, flux_limit, which, secondary)
        if z_range is not None:
            lo, hi = z_range
            keep = (self.z_edges[:-1] >= lo - 1e-9) & (self.z_edges[1:] <= hi + 1e-9)
            h = h[keep]
        return float(h.sum() / float(area_deg2))

    def single_fraction(self, line: str, flux_limit: float, secondary: float | None = None) -> np.ndarray:
        """Fraction of the line's galaxies per bin that have no other line above the limit."""
        s = self.hist(line, flux_limit, "single", secondary).astype(np.float64)
        tot = s + self.hist(line, flux_limit, "multi", secondary)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(tot > 0, s / tot, np.nan)

    def galaxies(self, flux_limit: float, which: str = "all", secondary: float | None = None) -> np.ndarray:
        """Distinct galaxies per bin with at least one passing line (``"all"``), exactly
        one (``"single"``) or more than one (``"multi"``); no galaxy counted twice."""
        g = self.n_any[self.limit_index(flux_limit, secondary)]
        if which == "all":
            return g.sum(axis=0)
        if which in CLASSES:
            return g[CLASSES.index(which)]
        raise ValueError(f"which must be 'all', 'single' or 'multi'; got {which!r}")

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> Path:
        path = Path(path)
        np.savez(path, flux_limits=self.flux_limits, z_edges=self.z_edges, line_names=np.array(self.line_names),
                 counts=self.counts, n_all=self.n_all, n_any=self.n_any, wave_range=np.array(self.wave_range),
                 z_column=np.array(self.z_column), steps=np.array(self.steps, dtype=np.int64),
                 secondary_limits=self.secondary_limits)
        return path

    @classmethod
    def load(cls, path) -> "LineCensus":
        with np.load(Path(path)) as f:
            n_any = f["n_any"] if "n_any" in f else np.zeros((f["flux_limits"].size, 2, f["z_edges"].size - 1), dtype=np.int64)
            return cls(f["flux_limits"], f["z_edges"], [str(s) for s in f["line_names"]], f["counts"], f["n_all"], n_any,
                       tuple(float(w) for w in f["wave_range"]), str(f["z_column"]), [int(s) for s in f["steps"]],
                       f["secondary_limits"] if "secondary_limits" in f else None)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Histogram one dict of catalog arrays; returns ``(counts, n_all, n_any)`` as in :class:`LineCensus`.

    Fluxes use ``D_L(z_dl_column)`` and the bandpass test uses ``z_column`` (the redshift
    with peculiar velocity), exactly as the pipeline does; the histograms are binned in
    ``z_column`` as well.
    """
    line_set = get_line_set(line_set)
    z_edges = np.asarray(z_edges, dtype=np.float64)
    flux_limits, secondary_limits = normalize_limits(flux_limits)
    z_dl = np.asarray(data[z_dl_column], dtype=np.float64)
    z_rsd = np.asarray(data[z_column], dtype=np.float64)
    n_bins = z_edges.size - 1

    idx = np.digitize(z_rsd, z_edges) - 1
    in_range = (idx >= 0) & (idx < n_bins) & (z_rsd < z_edges[-1])
    n_all = np.bincount(idx[in_range], minlength=n_bins).astype(np.int64)

    flux = feature_fluxes(data, line_set, z_dl, cosmology, backend=backend)
    comp = component_fluxes(data, line_set, z_dl, cosmology, backend=backend) if line_set.has_separated_components else None
    counts = np.zeros((flux_limits.size, len(line_set), 2, n_bins), dtype=np.int64)
    n_any = np.zeros((flux_limits.size, 2, n_bins), dtype=np.int64)
    for i, (lim, second) in enumerate(zip(flux_limits, secondary_limits)):
        # a plain number is a single threshold (the population Wang et al. tabulate): no second tier
        model = ThresholdDetection(ConstantFluxLimit(float(lim), wave_range,
                                                     secondary_flux_limit=None if second == lim else float(second)))
        det = detect_lines(flux, z_rsd, line_set, model, component_flux=comp)
        single = in_range & (det.n_det == 1)
        multi = in_range & (det.n_det > 1)
        n_any[i, SINGLE] = np.bincount(idx[single], minlength=n_bins)
        n_any[i, MULTI] = np.bincount(idx[multi], minlength=n_bins)
        for j in range(len(line_set)):
            hit = det.detected[:, j]
            counts[i, j, SINGLE] = np.bincount(idx[hit & single], minlength=n_bins)
            counts[i, j, MULTI] = np.bincount(idx[hit & multi], minlength=n_bins)
    return counts, n_all, n_any


# ---------------------------------------------------------------------------------
# file level
# ---------------------------------------------------------------------------------
def _census_one_file(spec: dict, path: str) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Process-pool worker: census of one slice file from a serialisable ``spec``."""
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    line_set = get_line_set(spec["line_set"])
    info = file_info(path)
    catalog = open_catalog([path], **spec["open_kwargs"])
    cols = [spec["z_column"], spec["z_dl_column"], *line_set.required_columns]
    counts = n_all = n_any = None
    for ds in (catalog.values() if hasattr(catalog, "values") else [catalog]):
        data = ds.select(*cols).get_data("numpy", wrap_single=True)
        c, n, a = census_arrays(data, line_set, ds.cosmology, spec["flux_limits"], np.asarray(spec["z_edges"]),
                                tuple(spec["wave_range"]), spec["z_column"], spec["z_dl_column"], spec["backend"])
        counts = c if counts is None else counts + c
        n_all = n if n_all is None else n_all + n
        n_any = a if n_any is None else n_any + a
    return info.step, info.n_gal, counts, n_all, n_any


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
        "flux_limits": [[p, s] for p, s in census.limits],
        "z_edges": z_edges.tolist(),
        "wave_range": [float(w) for w in wave_range],
        "z_column": z_column,
        "z_dl_column": z_dl_column,
        "backend": backend,
        "open_kwargs": {**DEFAULT_OPEN_KWARGS, **(open_kwargs or {})},
    }

    def record(step, n_gal, counts, n_all, n_any):
        census.add(counts, n_all, n_any, step)
        if verbose:
            print(f"[census] step {step:4d}  n={n_gal:,}  in z range {int(n_all.sum()):,}  "
                  f"with a line above F>{census.flux_limits[0]:.2g}: {int(n_any[0].sum()):,}", flush=True)

    if n_workers <= 1:
        for p in paths:
            record(*_census_one_file(spec, str(p)))
        census.steps.sort()
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

"""Step 4 of the plan: build analysis samples from the parent emission-line catalog.

* :func:`sample_mask` applies footprint, assigned-redshift range, tracer (assigned
  line) and redshift-quality (outcome) cuts to arrays.
* :func:`downsample_to_dndz` thins a sample to a target ``dN/dz_obs`` per deg^2 with
  ``p_keep(z) = (dN/dz)_target / (dN/dz)_mock`` (clipped at 1; plan comment 5).
* :func:`sky_area_deg2` estimates the footprint area from the OpenCosmo HEALPix index.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .lines import LineSet, get_line_set
from .redshift import IdOutcome

DEG2_PER_STERADIAN = (180.0 / np.pi) ** 2
FULL_SKY_DEG2 = 4.0 * np.pi * DEG2_PER_STERADIAN


# ---------------------------------------------------------------------------------
# Footprints
# ---------------------------------------------------------------------------------
class Footprint:
    """Callable footprint: ``mask = footprint(ra_deg, dec_deg)``."""

    def __call__(self, ra, dec) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class BoxFootprint(Footprint):
    def __init__(self, ra_min: float, ra_max: float, dec_min: float, dec_max: float):
        self.ra_min, self.ra_max, self.dec_min, self.dec_max = map(float, (ra_min, ra_max, dec_min, dec_max))

    def __call__(self, ra, dec):
        ra = np.asarray(ra) % 360.0
        dec = np.asarray(dec)
        if self.ra_min <= self.ra_max:
            in_ra = (ra >= self.ra_min) & (ra <= self.ra_max)
        else:  # wraps through 0
            in_ra = (ra >= self.ra_min) | (ra <= self.ra_max)
        return in_ra & (dec >= self.dec_min) & (dec <= self.dec_max)

    def to_config(self):
        return {"type": "box", "ra_min": self.ra_min, "ra_max": self.ra_max, "dec_min": self.dec_min, "dec_max": self.dec_max}


class HealpixFootprint(Footprint):
    """Keep objects whose HEALPix pixel (nested) is in a list, or where a boolean/float
    map is positive. Requires ``healpy``."""

    def __init__(self, nside: int, pixels=None, mask=None, nest: bool = True):
        import healpy as hp

        self.hp = hp
        self.nside = int(nside)
        self.nest = bool(nest)
        npix = hp.nside2npix(self.nside)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.size != npix:
                raise ValueError("mask must have nside2npix(nside) entries")
            self.good = mask > 0
        elif pixels is not None:
            self.good = np.zeros(npix, dtype=bool)
            self.good[np.asarray(pixels, dtype=np.int64)] = True
        else:
            raise ValueError("give either pixels or mask")

    @classmethod
    def from_file(cls, path, nside=None, nest=True):
        import healpy as hp

        p = Path(path)
        if p.suffix in (".fits", ".fit"):
            m = hp.read_map(str(p), nest=nest)
            return cls(hp.npix2nside(m.size), mask=m, nest=nest)
        pix = np.loadtxt(p, dtype=np.int64)
        if nside is None:
            raise ValueError("nside is required for a pixel-list file")
        return cls(nside, pixels=pix, nest=nest)

    def __call__(self, ra, dec):
        pix = self.hp.ang2pix(self.nside, np.asarray(ra), np.asarray(dec), nest=self.nest, lonlat=True)
        return self.good[pix]

    def area_deg2(self) -> float:
        return self.good.sum() * self.hp.nside2pixarea(self.nside, degrees=True)


def build_footprint(cfg) -> Footprint | None:
    if cfg is None or isinstance(cfg, Footprint):
        return cfg
    cfg = dict(cfg)
    kind = cfg.pop("type", "box")
    if kind == "box":
        return BoxFootprint(**cfg)
    if kind == "healpix":
        if "file" in cfg:
            return HealpixFootprint.from_file(cfg["file"], nside=cfg.get("nside"), nest=cfg.get("nest", True))
        return HealpixFootprint(**cfg)
    raise KeyError(f"unknown footprint type {kind!r}")


def sky_area_deg2(catalog, nside: int = 256) -> float:
    """Area covered by an OpenCosmo lightcone, from its occupied HEALPix pixels."""
    pixels = catalog.get_pixels(nside)
    return float(len(np.unique(pixels)) * FULL_SKY_DEG2 / (12 * nside**2))


# ---------------------------------------------------------------------------------
# Sample definition
# ---------------------------------------------------------------------------------
@dataclass
class SampleDefinition:
    """Cuts defining one analysis sample (all optional).

    ``outcomes``: allowed :class:`IdOutcome` values (names or ints). The default keeps
    every galaxy with a redshift (multi-line, single correct, single misidentified) and
    drops failures and non-detections, which is what an observer would do.
    ``assigned_lines``: tracer definition, e.g. ``["Ha"]`` for an H-alpha sample; a
    misidentified galaxy counts under its *assigned* line (that is what makes it an
    interloper).
    """

    name: str = "sample"
    z_min: float | None = None
    z_max: float | None = None
    outcomes: Sequence = (IdOutcome.MULTI_LINE, IdOutcome.SINGLE_CORRECT, IdOutcome.SINGLE_MISID)
    assigned_lines: Sequence[str] | None = None
    n_det_min: int | None = None
    footprint: dict | Footprint | None = None
    extra_cuts: dict = field(default_factory=dict)  # {column: [min, max]} applied to passthrough columns

    def outcome_codes(self) -> np.ndarray:
        codes = []
        for o in self.outcomes:
            codes.append(int(IdOutcome[o]) if isinstance(o, str) else int(o))
        return np.array(codes, dtype=np.int8)


def sample_mask(data: Mapping[str, np.ndarray], line_set, sample: SampleDefinition) -> np.ndarray:
    """Boolean mask of ``data`` (dict of arrays from the parent catalog) for a sample."""
    line_set = get_line_set(line_set)
    n = len(data["z_obs"])
    mask = np.ones(n, dtype=bool)
    mask &= np.isin(data["id_outcome"], sample.outcome_codes())
    z = np.asarray(data["z_obs"])
    if sample.z_min is not None:
        mask &= z >= sample.z_min
    if sample.z_max is not None:
        mask &= z <= sample.z_max
    mask &= np.isfinite(z) | np.isin(data["id_outcome"], [int(IdOutcome.NOT_DETECTED), int(IdOutcome.SINGLE_FAILED)])
    if sample.assigned_lines:
        mask &= np.isin(data["line_assigned"], line_set.index_of(sample.assigned_lines))
    if sample.n_det_min is not None:
        mask &= data["n_det"] >= sample.n_det_min
    fp = build_footprint(sample.footprint)
    if fp is not None:
        mask &= fp(data["ra"], data["dec"])
    for col, (lo, hi) in sample.extra_cuts.items():
        v = np.asarray(data[col])
        if lo is not None:
            mask &= v >= lo
        if hi is not None:
            mask &= v <= hi
    return mask


# ---------------------------------------------------------------------------------
# Target dN/dz and downsampling
# ---------------------------------------------------------------------------------
class TargetDNDZ:
    """Target ``dN/dz`` per square degree as a function of assigned redshift.

    Built from a table ``(z, dN/dz [deg^-2])`` with linear interpolation, zero outside.
    """

    def __init__(self, z: np.ndarray, dndz: np.ndarray):
        self.z = np.asarray(z, dtype=np.float64)
        self.dndz = np.asarray(dndz, dtype=np.float64)
        if self.z.ndim != 1 or self.z.shape != self.dndz.shape or np.any(np.diff(self.z) <= 0):
            raise ValueError("z must be increasing and match dndz in shape")

    @classmethod
    def from_file(cls, path):
        arr = np.loadtxt(Path(path))
        return cls(arr[:, 0], arr[:, 1])

    def __call__(self, z):
        return np.interp(np.asarray(z, dtype=np.float64), self.z, self.dndz, left=0.0, right=0.0)


def mock_dndz(z_obs: np.ndarray, z_edges: np.ndarray, area_deg2: float, smooth_bins: float = 0.0) -> np.ndarray:
    """Histogram estimate of the mock ``dN/dz`` per deg^2, optionally Gaussian-smoothed
    over ``smooth_bins`` bins to avoid matching one realisation's radial fluctuations."""
    counts, _ = np.histogram(z_obs[np.isfinite(z_obs)], bins=z_edges)
    dndz = counts / (np.diff(z_edges) * area_deg2)
    if smooth_bins and smooth_bins > 0:
        from scipy.ndimage import gaussian_filter1d

        dndz = gaussian_filter1d(dndz, smooth_bins, mode="nearest")
    return dndz


def downsample_to_dndz(
    z_obs: np.ndarray,
    target: Callable[[np.ndarray], np.ndarray],
    area_deg2: float,
    z_edges: np.ndarray,
    rng: np.random.Generator,
    smooth_bins: float = 0.0,
    mock_dndz_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Random downsampling to a target ``dN/dz_obs``.

    Returns ``(keep_mask, p_keep_per_bin)``. ``mock_dndz_values`` allows passing a
    dN/dz estimated on the full (multi-file) sample so that per-file calls use a
    common selection function. Bins where the target exceeds the mock are kept
    entirely and a warning is issued: downsampling cannot fill a shortage.
    """
    z_edges = np.asarray(z_edges, dtype=np.float64)
    z_obs = np.asarray(z_obs, dtype=np.float64)
    mock = mock_dndz_values if mock_dndz_values is not None else mock_dndz(z_obs, z_edges, area_deg2, smooth_bins)
    zc = 0.5 * (z_edges[1:] + z_edges[:-1])
    tgt = np.asarray(target(zc), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_keep = np.where(mock > 0, tgt / mock, 0.0)
    short = p_keep > 1.0 + 1e-9
    if np.any(short):
        warnings.warn(
            f"target dN/dz exceeds the mock in {short.sum()} of {short.size} bins "
            f"(z ~ {zc[short].min():.2f}-{zc[short].max():.2f}); those bins are kept entirely.",
            RuntimeWarning,
            stacklevel=2,
        )
    p_keep = np.clip(p_keep, 0.0, 1.0)
    b = np.searchsorted(z_edges, z_obs, side="right") - 1
    inside = (b >= 0) & (b < p_keep.size) & np.isfinite(z_obs)
    p = np.zeros(z_obs.size)
    p[inside] = p_keep[b[inside]]
    keep = rng.random(z_obs.size) < p
    return keep, p_keep

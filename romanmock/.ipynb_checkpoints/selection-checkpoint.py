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

    def area_deg2(self) -> float:
        """Exact spherical area of the box (independent of the lightcone coverage)."""
        dra = (self.ra_max - self.ra_min) % 360.0 or 360.0
        dsin = np.sin(np.radians(self.dec_max)) - np.sin(np.radians(self.dec_min))
        return float(np.radians(dra) * dsin * DEG2_PER_STERADIAN)

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


def sky_area_deg2(catalog, nside: int | None = None, footprint: Footprint | None = None,
                  boundary_correction: bool = True) -> float:
    """Area covered by an OpenCosmo lightcone, from its occupied HEALPix pixels.

    Pixels are taken at the finest resolution of the file's spatial index (``nside``
    None; 1024 for the diffsky mock) and pixels with an unoccupied neighbour (the
    footprint edge, on average half covered) count one half. For the diffsky mock this
    gives 2062.5 deg^2 against 2062.6 for the nominal 5 % of the sky, whereas plain
    counting at nside 256 gives 2114. With a ``footprint``, only pixels whose centres
    fall inside it are counted, i.e. the area of the intersection of the lightcone
    coverage and the footprint (needs ``healpy``)."""
    import healpy as hp

    if nside is None:
        for candidate in (4096, 2048, 1024, 512, 256, 128, 64):
            try:
                pixels = catalog.get_pixels(candidate)
            except ValueError:
                continue
            nside = candidate
            break
        else:
            raise ValueError("lightcone has no usable spatial index")
    else:
        pixels = catalog.get_pixels(nside)
    pixels = np.unique(pixels)
    if footprint is not None:
        ra, dec = hp.pix2ang(nside, pixels, nest=True, lonlat=True)
        pixels = pixels[np.asarray(footprint(ra, dec), dtype=bool)]
    if pixels.size == 0:
        return 0.0
    weight = float(pixels.size)
    if boundary_correction:
        occupied = np.zeros(hp.nside2npix(nside), dtype=bool)
        occupied[pixels] = True
        nb = hp.get_all_neighbours(nside, pixels, nest=True)
        interior = np.all(occupied[np.where(nb >= 0, nb, pixels)], axis=0)
        weight = interior.sum() + 0.5 * (~interior).sum()
    return float(weight * FULL_SKY_DEG2 / (12 * nside**2))


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

    ``membership`` (review M2) defines what "belongs to the tracer" means for multi-line
    galaxies: ``"assigned"`` (default) uses ``line_assigned`` (the brightest detected
    feature), which makes tracer samples disjoint; ``"detected"`` counts a correctly
    identified multi-line galaxy in every sample whose line it detects (bit set in
    ``line_bitmask``), so a galaxy with both Hα and a stronger [O III] is in the Hα sample
    too. Single-line galaxies always use their assigned identity.
    """

    name: str = "sample"
    z_min: float | None = None
    z_max: float | None = None
    outcomes: Sequence = (IdOutcome.MULTI_LINE, IdOutcome.SINGLE_CORRECT, IdOutcome.SINGLE_MISID)
    assigned_lines: Sequence[str] | None = None
    membership: str = "assigned"
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
    n = len(data["redshift_assigned"])
    mask = np.ones(n, dtype=bool)
    mask &= np.isin(data["id_outcome"], sample.outcome_codes())
    z = np.asarray(data["redshift_assigned"])
    if sample.z_min is not None:
        mask &= z >= sample.z_min
    if sample.z_max is not None:
        mask &= z <= sample.z_max
    mask &= np.isfinite(z) | np.isin(data["id_outcome"], [int(IdOutcome.NOT_DETECTED), int(IdOutcome.SINGLE_FAILED)])
    if sample.assigned_lines:
        mask &= tracer_membership(data, line_set, sample.assigned_lines, sample.membership)
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


def tracer_membership(data: Mapping[str, np.ndarray], line_set, lines: Sequence[str], membership: str = "assigned") -> np.ndarray:
    """Boolean mask of galaxies belonging to the tracer(s) ``lines`` (see SampleDefinition)."""
    line_set = get_line_set(line_set)
    idx = line_set.index_of(lines)
    assigned = np.isin(np.asarray(data["line_assigned"]), idx)
    if membership == "assigned":
        return assigned
    if membership == "detected":
        bits = np.zeros(len(assigned), dtype=np.uint64)
        bm = np.asarray(data["line_bitmask"]).astype(np.uint64)
        for i in idx:
            bits |= (bm >> np.uint64(i)) & np.uint64(1)
        multi = np.asarray(data["n_det"]) >= 2
        return np.where(multi, bits > 0, assigned)
    raise ValueError("membership must be 'assigned' or 'detected'")


# ---------------------------------------------------------------------------------
# Target dN/dz and downsampling
# ---------------------------------------------------------------------------------
class TargetDNDZ:
    """Target ``dN/dz`` per square degree as a function of assigned redshift.

    Built from a table ``(z, dN/dz [deg^-2])`` with linear interpolation, zero outside.
    """

    def __init__(self, z: np.ndarray, dndz: np.ndarray, edges: np.ndarray | None = None):
        self.z = np.asarray(z, dtype=np.float64)
        self.dndz = np.asarray(dndz, dtype=np.float64)
        if self.z.ndim != 1 or self.z.shape != self.dndz.shape or np.any(np.diff(self.z) <= 0):
            raise ValueError("z must be increasing and match dndz in shape")
        self.edges = None if edges is None else np.asarray(edges, dtype=np.float64)

    @classmethod
    def from_file(cls, path):
        arr = np.loadtxt(Path(path))
        return cls(arr[:, 0], arr[:, 1])

    @classmethod
    def from_bins(cls, edges, dndz) -> "TargetDNDZ":
        """Piecewise-constant target from bin edges (``n+1``) and per-bin ``dN/dz`` (``n``),
        as tabulated in survey forecasts. Zero outside the edges."""
        edges = np.asarray(edges, dtype=np.float64)
        dndz = np.asarray(dndz, dtype=np.float64)
        if edges.size != dndz.size + 1 or np.any(np.diff(edges) <= 0):
            raise ValueError("edges must be increasing with one more entry than dndz")
        return cls(0.5 * (edges[1:] + edges[:-1]), dndz, edges=edges)

    @property
    def z_range(self) -> tuple[float, float]:
        if self.edges is not None:
            return float(self.edges[0]), float(self.edges[-1])
        return float(self.z[0]), float(self.z[-1])

    def __call__(self, z):
        z = np.asarray(z, dtype=np.float64)
        if self.edges is not None:  # step function
            b = np.searchsorted(self.edges, z, side="right") - 1
            b = np.where(z == self.edges[-1], self.edges.size - 2, b)
            inside = (b >= 0) & (b < self.dndz.size)
            return np.where(inside, self.dndz[np.clip(b, 0, self.dndz.size - 1)], 0.0)
        return np.interp(z, self.z, self.dndz, left=0.0, right=0.0)


def mock_dndz(z_obs: np.ndarray, z_edges: np.ndarray, area_deg2: float, smooth_bins: float = 0.0,
              poly_degree: int | None = None) -> np.ndarray:
    """Histogram estimate of the mock ``dN/dz`` per deg^2, optionally Gaussian-smoothed
    over ``smooth_bins`` bins and/or replaced by a polynomial fit of degree ``poly_degree``
    to log dN/dz, so that the selection function does not divide out the realisation's own
    radial fluctuations."""
    counts, _ = np.histogram(z_obs[np.isfinite(z_obs)], bins=z_edges)
    dndz = counts / (np.diff(z_edges) * area_deg2)
    if smooth_bins and smooth_bins > 0:
        from scipy.ndimage import gaussian_filter1d

        dndz = gaussian_filter1d(dndz, smooth_bins, mode="nearest")
    if poly_degree is not None:
        # smooth selection function: low-order polynomial fit to log dN/dz (plan comment 5 /
        # review M1: do not divide out the realisation's own radial fluctuations)
        zc = 0.5 * (z_edges[1:] + z_edges[:-1])
        good = dndz > 0
        if good.sum() > poly_degree + 1:
            coef = np.polyfit(zc[good], np.log(dndz[good]), poly_degree)
            dndz = np.where(good, np.exp(np.polyval(coef, zc)), 0.0)
    return dndz


def downsample_to_dndz(
    z_obs: np.ndarray,
    target: Callable[[np.ndarray], np.ndarray],
    area_deg2: float,
    z_edges: np.ndarray,
    rng: np.random.Generator,
    smooth_bins: float = 0.0,
    mock_dndz_values: np.ndarray | None = None,
    warn_empty: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Random downsampling to a target ``dN/dz_obs``.

    Returns ``(keep_mask, p_keep_per_bin)``. ``mock_dndz_values`` allows passing a
    dN/dz estimated on the full (multi-file) sample so that per-file calls use a
    common selection function. Bins where the target exceeds the mock are kept
    entirely and a warning is issued: downsampling cannot fill a shortage. Bins with no
    galaxies at all are included in that warning unless ``warn_empty`` is False.
    """
    z_edges = np.asarray(z_edges, dtype=np.float64)
    z_obs = np.asarray(z_obs, dtype=np.float64)
    mock = mock_dndz_values if mock_dndz_values is not None else mock_dndz(z_obs, z_edges, area_deg2, smooth_bins)
    zc = 0.5 * (z_edges[1:] + z_edges[:-1])
    tgt = np.asarray(target(zc), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_keep = np.where(mock > 0, tgt / mock, 0.0)
    short = tgt > mock * (1.0 + 1e-9)          # includes bins where the mock is empty
    if not warn_empty:
        short &= mock > 0
    if np.any(short):
        warnings.warn(
            f"target dN/dz exceeds the mock in {short.sum()} of {short.size} bins "
            f"(z ~ {zc[short].min():.2f}-{zc[short].max():.2f}); thinning cannot fill a shortage, "
            "those bins keep every galaxy.",
            RuntimeWarning,
            stacklevel=2,
        )
    p_keep = np.clip(p_keep, 0.0, 1.0)
    b = np.searchsorted(z_edges, z_obs, side="right") - 1
    b = np.where(z_obs == z_edges[-1], p_keep.size - 1, b)   # closed upper edge, like np.histogram
    inside = (b >= 0) & (b < p_keep.size) & np.isfinite(z_obs)
    p = np.zeros(z_obs.size)
    p[inside] = p_keep[b[inside]]
    keep = rng.random(z_obs.size) < p
    return keep, p_keep


# ---------------------------------------------------------------------------------
# Population thinning: match the *detected-line populations* to reference dN/dz
# ---------------------------------------------------------------------------------
def population_thinning_mask(
    data: Mapping[str, np.ndarray],
    line_set,
    targets: Mapping[str, Callable[[np.ndarray], np.ndarray]],
    area_deg2: float,
    rng: np.random.Generator,
    by: str = "line_true",
    z_column: str = "redshift_rsd",
    bin_width: float = 0.1,
    smooth_bins: float = 0.0,
    poly_degree: int | None = None,
    outside_range: str = "keep",
) -> tuple[np.ndarray, dict]:
    """Thin each emitter population to a reference ``dN/dz`` in bins of *true* redshift.

    Survey forecasts such as Wang et al. (2022) tabulate true emitters above the flux
    limit, before redshift identification. This function therefore thins the mock's
    detected-line populations (membership from ``data[by]``, default the brightest
    detected feature) as a function of ``data[z_column]`` (default the true redshift with
    peculiar velocity), *before* any assigned-redshift selection. Interlopers are then
    thinned consistently with the population they really belong to.

    ``targets`` maps feature names to callables ``dN/dz(z)`` (e.g. :class:`TargetDNDZ`).
    Galaxies whose population has no target, or that lie outside the target's redshift
    range, are kept when ``outside_range == "keep"`` and dropped when ``"drop"``.

    Returns ``(keep_mask, info)`` where ``info[feature]`` holds ``z_edges``, ``p_keep``,
    ``mock`` and ``target`` per bin.
    """
    from .lines import get_line_set

    line_set = get_line_set(line_set)
    member = np.asarray(data[by]).astype(np.int64)
    z = np.asarray(data[z_column], dtype=np.float64)
    keep = np.ones(z.size, dtype=bool)
    info = {}
    for name, target in targets.items():
        i = line_set.bit(name)
        sel = member == i
        if not np.any(sel):
            continue
        if getattr(target, "edges", None) is not None:
            edges = np.asarray(target.edges, dtype=np.float64)
        else:
            lo, hi = target.z_range if hasattr(target, "z_range") else (np.nanmin(z[sel]), np.nanmax(z[sel]))
            n = max(int(np.ceil((hi - lo) / bin_width - 1e-9)), 1)
            edges = lo + bin_width * np.arange(n + 1)
        zc = 0.5 * (edges[1:] + edges[:-1])
        mock = mock_dndz(z[sel], edges, area_deg2, smooth_bins, poly_degree)
        tgt = np.asarray(target(zc), dtype=np.float64)
        empty = (mock == 0) & (tgt > 0)
        if np.any(empty):   # not a shortage of the mock: nobody has this line as `by` there
            warnings.warn(
                f"{name}: {empty.sum()} of {empty.size} target bins (z {edges[:-1][empty].min():.2f}-"
                f"{edges[1:][empty].max():.2f}) contain no galaxies with {by} == {name!r}; "
                "the target cannot be applied there (another line is the brightest one).",
                RuntimeWarning, stacklevel=2)
        sub_keep, p_keep = downsample_to_dndz(z[sel], target, area_deg2, edges, rng,
                                              mock_dndz_values=mock, warn_empty=False)
        in_range = (z[sel] >= edges[0]) & (z[sel] <= edges[-1])
        if outside_range == "keep":
            sub_keep = np.where(in_range, sub_keep, True)
        elif outside_range != "drop":
            raise ValueError("outside_range must be 'keep' or 'drop'")
        keep[np.where(sel)[0]] = sub_keep
        info[name] = {"z_edges": edges, "z_center": zc, "p_keep": p_keep, "mock": mock,
                      "target": np.asarray(target(zc), dtype=np.float64),
                      "n_before": int(sel.sum()), "n_after": int(sub_keep.sum())}
    if outside_range == "drop":
        no_target = ~np.isin(member, line_set.index_of(list(targets)))
        keep[no_target] = False
    return keep, info

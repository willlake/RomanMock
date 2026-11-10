"""Line doublets that the Roman grism may or may not separate: when [O III] λ4959 is a line of its own.

The two lines of [O III] λλ4959, 5007 are 47.94 (1 + z) Å apart, 96–185 Å inside the grism bandpass,
with a flux ratio fixed by atomic physics (λ5007 / λ4959 = 2.98). In a slitless grism the image of a line
along the dispersion is the galaxy itself, so whether the two lines separate depends on the galaxy's size.
:class:`DoubletRule` models the image of one line as a face-on exponential disk of the galaxy's half-light
radius ``r50`` projected on the dispersion axis (``f(x) = |x| K1(|x| / h)``, ``h = r50 / 1.678``, 98.6 Å per
arcsec for 10.85 Å per 0.11" pixel), smoothed by a Gaussian of the 2-pixel resolution element of a point
source. The doublet is resolved when the profile dips between the two lines by at least ``min_dip`` (20 %)
of the weak line's peak; the largest resolved half-light radius is then 0.35" at z = 1.0, 0.55" at
z = 1.94 and 0.74" at z = 2.85 (:meth:`DoubletRule.resolved_radius`). A dip of 10 % or 30 % moves it by
about ±15 %. Derivation and caveats: ``Projects/Roman/Diffsky_line_rescaling_and_OIII_doublet.md``, Part 2.

``mode`` of the rule:

* ``"per_galaxy"``: the weak line is a feature of its own where the doublet is resolved for the galaxy's
  size and redshift, and its flux is added to the strong line where the doublet is blended;
* ``"merged"``: added in every galaxy (a doublet never confirms a redshift);
* ``"separated"``: never (each line is a feature of its own, as without a rule).

The rule acts on the feature fluxes before detection (:meth:`DoubletRule.apply`), and only in galaxies
whose two lines are both inside the bandpass: elsewhere at most one of them can be detected anyway. A
merged doublet is detected at the strong line's rest wavelength, the weak line then has no flux. The size
is a :class:`~romanmock.sizes.SizeSpec` (default the column ``r50_disk_2d`` in proper kpc, converted with
the angular-diameter distance at ``redshift_true``, interpolated on a redshift grid for speed); a galaxy
without a valid size counts as blended.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import numpy as np

from .lines import get_line_set
from .sizes import SizeSpec, arcsec_per_kpc_proper

GRISM_DISPERSION = 10.85     #: Å per pixel of the Roman grism (first order), nominal
GRISM_PIXEL = 0.11           #: arcsec per pixel of the WFI
POINT_SOURCE_FWHM = 2.0      #: pixels: the resolution element of a point source
DEFAULT_MIN_DIP = 0.2        #: resolved: the profile dips between the lines by >= 20 % of the weak line's peak
OIII_RATIO = 2.98            #: [O III] λ5007 / λ4959, set by atomic physics
DOUBLET_MODES = ("per_galaxy", "merged", "separated")
_Z_GRID = (0.0, 6.0, 0.05)   #: redshifts of the table of the largest resolved radius (start, stop, step)


def doublet_profile(r50: float, separation: float, ratio: float = OIII_RATIO, dispersion: float = GRISM_DISPERSION,
                    pixel: float = GRISM_PIXEL, point_fwhm: float = POINT_SOURCE_FWHM, step: float = 0.5,
                    half_width: float = 3000.0) -> tuple[np.ndarray, np.ndarray]:
    """The image of a doublet along the dispersion for a galaxy of half-light radius ``r50`` [arcsec].

    Returns ``x`` [Å from the strong line, observed frame] and the profile: an exponential disk projected on
    one axis, ``2 |x| K1(|x| / h)`` normalised to 1 at its centre, for the strong line, plus a copy
    ``1 / ratio`` as bright and ``separation`` [Å] to the blue for the weak line, smoothed by a Gaussian of
    FWHM ``point_fwhm`` pixels.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.special import k1

    x = np.arange(-half_width, half_width, step)
    u = np.abs(x) / (max(float(r50), 1e-4) / 1.678 * dispersion / pixel)      # in disk scale lengths
    line = np.where(u > 1e-9, u * k1(np.maximum(u, 1e-9)), 1.0)
    profile = line + np.interp(x + separation, x, line, left=0.0, right=0.0) / ratio
    return x, gaussian_filter1d(profile, point_fwhm * dispersion / np.sqrt(8.0 * np.log(2.0)) / step)


def doublet_dip(r50: float, separation: float, **kwargs) -> float:
    """Depth of the dip between the two lines as a share of the weak line's peak (0 when the weak line makes
    no peak of its own), for a galaxy of half-light radius ``r50`` [arcsec]; ``kwargs`` as
    :func:`doublet_profile`."""
    x, profile = doublet_profile(r50, separation, **kwargs)
    seg = profile[np.searchsorted(x, -1.6 * separation):np.searchsorted(x, 0.0)]
    d = np.diff(seg)
    peaks = np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0)) + 1
    if peaks.size == 0:
        return 0.0
    p = peaks[np.argmax(seg[peaks])]
    return float(1.0 - seg[p:].min() / seg[p])


@lru_cache(maxsize=32)
def _resolved_radius_table(rest_separation: float, ratio: float, min_dip: float, dispersion: float, pixel: float,
                           point_fwhm: float, grid: tuple = _Z_GRID) -> tuple[np.ndarray, np.ndarray]:
    """The largest resolved half-light radius [arcsec] on a grid of redshifts (bisection on the radius: the
    dip shrinks as the galaxy grows). Computed once per process and parameter set (~2 s)."""
    z = np.arange(grid[0], grid[1] + 1e-9, grid[2])
    kw = dict(ratio=ratio, dispersion=dispersion, pixel=pixel, point_fwhm=point_fwhm, step=1.0)
    r_max = np.empty(z.size)
    for k, zz in enumerate(z):
        lo, hi = 0.0, 3.0
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if doublet_dip(mid, rest_separation * (1.0 + zz), **kw) >= min_dip else (lo, mid)
        r_max[k] = lo
    return z, r_max


_ARCSEC_PER_KPC: dict = {}


def arcsec_per_kpc_interpolated(z, cosmology) -> np.ndarray:
    """``arcsec_per_kpc_proper`` interpolated on a redshift grid (0.05 < z < 10, steps of 0.0002 in log(1+z)):
    the same to 2e-6 and fast for millions of galaxies; computed directly below z = 0.05."""
    key = repr(cosmology)
    if key not in _ARCSEC_PER_KPC:
        grid = np.expm1(np.arange(np.log1p(0.05), np.log1p(10.0) + 1e-12, 0.0002))
        _ARCSEC_PER_KPC[key] = (np.log1p(grid), np.log(arcsec_per_kpc_proper(grid, cosmology)))
    lz, lv = _ARCSEC_PER_KPC[key]
    z = np.asarray(z, dtype=np.float64)
    out = np.exp(np.interp(np.log1p(np.maximum(z, 0.0)), lz, lv))
    low = z < 0.05
    if low.any():
        out[low] = arcsec_per_kpc_proper(z[low], cosmology)
    return out


class DoubletRule:
    """When the weak line of a doublet ([O III] λ4959 by default) is a feature of its own; see the module.

    Parameters
    ----------
    mode : {"per_galaxy", "merged", "separated"}
    strong, weak : str
        Feature names of the two lines in the line set.
    ratio : float
        Flux ratio strong / weak, for the profile model.
    min_dip : float
        Resolved when the profile dips between the lines by at least this share of the weak line's peak.
    size : SizeSpec, float, str or dict
        Half-light radius of the line emission (``mode="per_galaxy"`` only): a column in proper kpc
        (default ``"r50_disk_2d"``), a dict as :meth:`SizeSpec.parse` takes it, or a number in arcsec.
    dispersion, pixel, point_fwhm : float
        Å per pixel, arcsec per pixel, and the point-source resolution element in pixels.
    """

    type_name = "doublet"

    def __init__(self, mode: str = "per_galaxy", strong: str = "OIII_5007", weak: str = "OIII_4959",
                 ratio: float = OIII_RATIO, min_dip: float = DEFAULT_MIN_DIP, size="r50_disk_2d",
                 dispersion: float = GRISM_DISPERSION, pixel: float = GRISM_PIXEL, point_fwhm: float = POINT_SOURCE_FWHM):
        if mode not in DOUBLET_MODES:
            raise ValueError(f"doublet mode {mode!r}; known: {list(DOUBLET_MODES)}")
        if not 0.0 < float(min_dip) < 1.0:
            raise ValueError("min_dip is a share of the weak line's peak, between 0 and 1")
        if strong == weak:
            raise ValueError("strong and weak are two different features")
        self.mode, self.strong, self.weak = mode, str(strong), str(weak)
        self.ratio, self.min_dip = float(ratio), float(min_dip)
        self.dispersion, self.pixel, self.point_fwhm = float(dispersion), float(pixel), float(point_fwhm)
        self.size = SizeSpec.parse(size) if mode == "per_galaxy" else None

    # ---- what it needs --------------------------------------------------------------
    @property
    def required_columns(self) -> list[str]:
        """Columns of the catalog the rule reads (the size column and its redshift column)."""
        return [] if self.size is None or self.size.is_fixed else list(self.size.required_columns)

    def check(self, line_set) -> tuple[int, int]:
        """Indices of the strong and the weak line in ``line_set`` (KeyError when one is missing)."""
        names = get_line_set(line_set).names
        missing = [n for n in (self.strong, self.weak) if n not in names]
        if missing:
            raise KeyError(f"the doublet rule needs the features {missing}, which the line set {names} lacks")
        return names.index(self.strong), names.index(self.weak)

    # ---- the resolution criterion ---------------------------------------------------
    def rest_separation(self, line_set) -> float:
        ls = get_line_set(line_set)
        s, w = self.check(ls)
        return float(abs(ls.rest_waves[s] - ls.rest_waves[w]))

    def resolved_radius(self, z, line_set="roman_transitions_paa") -> np.ndarray:
        """The largest half-light radius [arcsec] with a resolved doublet at redshift ``z``."""
        zt, r_max = _resolved_radius_table(self.rest_separation(line_set), self.ratio, self.min_dip, self.dispersion,
                                           self.pixel, self.point_fwhm)
        return np.interp(np.asarray(z, dtype=np.float64), zt, r_max)

    def angular_size(self, data: Mapping[str, np.ndarray] | None, cosmology=None) -> np.ndarray:
        """Half-light radius [arcsec] of each galaxy of ``data`` (NaN where not valid), as the size spec gives it,
        with the kpc -> arcsec conversion interpolated (:func:`arcsec_per_kpc_interpolated`)."""
        spec = self.size
        if spec.is_fixed:
            n = len(next(iter(data.values()))) if data else 1
            return np.full(n, float(spec.fixed))
        raw = np.asarray(data[spec.column], dtype=np.float64)
        if spec.unit == "kpc":
            if cosmology is None:
                raise ValueError("a size in kpc needs the cosmology for the angular-diameter distance")
            raw = raw * arcsec_per_kpc_interpolated(np.asarray(data[spec.redshift_column], dtype=np.float64), cosmology)
        valid = np.isfinite(raw) & (raw >= 0)
        size = np.where(valid, np.maximum(raw * spec.scale, spec.floor), np.nan)
        if spec.fallback is not None:
            size = np.where(valid, size, spec.fallback)
        return size

    def resolved(self, z_rsd, r50, line_set="roman_transitions_paa") -> np.ndarray:
        """True where a galaxy of half-light radius ``r50`` [arcsec] at ``z_rsd`` has a resolved doublet (always
        for ``"separated"``, never for ``"merged"``; NaN sizes count as blended)."""
        z_rsd = np.asarray(z_rsd, dtype=np.float64)
        if self.mode == "separated":
            return np.ones(z_rsd.shape, dtype=bool)
        if self.mode == "merged":
            return np.zeros(z_rsd.shape, dtype=bool)
        r50 = np.broadcast_to(np.asarray(r50, dtype=np.float64), z_rsd.shape)
        with np.errstate(invalid="ignore"):
            return r50 < self.resolved_radius(z_rsd, line_set)

    # ---- the rule -------------------------------------------------------------------
    def apply(self, flux: np.ndarray, z_rsd: np.ndarray, line_set, wave_range, data: Mapping | None = None,
              cosmology=None) -> tuple[np.ndarray, np.ndarray]:
        """The feature fluxes with the doublet merged where the rule says so, and the mask of those galaxies.

        ``flux`` is ``(n, n_features)`` in the order of ``line_set``; the weak line's flux is added to the strong
        line's, and set to 0, in the galaxies whose two lines are both inside ``wave_range`` (strictly, as the
        detection tests it) and whose doublet is not resolved. ``data`` holds the size columns
        (:attr:`required_columns`) for ``mode="per_galaxy"``.
        """
        ls = get_line_set(line_set)
        s, w = self.check(ls)
        flux = np.asarray(flux, dtype=np.float64)
        z_rsd = np.asarray(z_rsd, dtype=np.float64)
        merged = np.zeros(z_rsd.shape, dtype=bool)
        if self.mode == "separated" or flux.shape[0] == 0:
            return flux, merged
        lo, hi = float(wave_range[0]), float(wave_range[1])
        ws, ww = ls.rest_waves[s] * (1.0 + z_rsd), ls.rest_waves[w] * (1.0 + z_rsd)
        both = (ws > lo) & (ws < hi) & (ww > lo) & (ww < hi)
        if self.mode == "merged":
            merged = both
        elif both.any():
            rows = np.flatnonzero(both)
            sub = {c: np.asarray(data[c])[rows] for c in self.required_columns} if self.required_columns else {}
            r50 = self.angular_size(sub, cosmology) if self.required_columns else np.full(rows.size, float(self.size.fixed))
            merged[rows] = ~self.resolved(z_rsd[rows], r50, ls)
        if merged.any():
            flux = flux.copy()
            flux[merged, s] += flux[merged, w]
            flux[merged, w] = 0.0
        return flux, merged

    # ---- description ----------------------------------------------------------------
    def to_config(self) -> dict:
        cfg = {"type": self.type_name, "mode": self.mode, "strong": self.strong, "weak": self.weak}
        if self.mode == "per_galaxy":
            cfg.update(ratio=self.ratio, min_dip=self.min_dip, size=self.size.to_config(), dispersion=self.dispersion,
                       pixel=self.pixel, point_fwhm=self.point_fwhm)
        return cfg

    def describe(self) -> str:
        if self.mode == "separated":
            return f"{self.weak} a line of its own"
        if self.mode == "merged":
            return f"{self.weak} added to {self.strong} in every galaxy"
        return (f"{self.weak} a line of its own where the doublet is resolved (dip >= {100 * self.min_dip:g} %, size "
                f"{self.size.describe()}), added to {self.strong} where blended")

    def __repr__(self) -> str:
        return f"DoubletRule({self.describe()})"


def build_doublets(cfg) -> DoubletRule | None:
    """A doublet rule from ``None`` (no rule), a mode (``"per_galaxy"``, ``"merged"``, ``"separated"``), a config
    dict (``{"type": "doublet", "mode": ..., ...}``) or a :class:`DoubletRule`."""
    if cfg is None or cfg is False:
        return None
    if isinstance(cfg, DoubletRule):
        return cfg
    if isinstance(cfg, str):
        return DoubletRule(mode=cfg)
    cfg = dict(cfg)
    kind = cfg.pop("type", "doublet")
    if kind != DoubletRule.type_name:
        raise KeyError(f"unknown doublet rule type {kind!r}; known: ['doublet']")
    return DoubletRule(**cfg)

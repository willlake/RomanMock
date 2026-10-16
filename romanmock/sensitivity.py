"""Sensitivity (flux limit versus wavelength and source size) and detection models.

Two small protocols keep the first-stage physics replaceable (plan comment 6):

* :class:`SensitivityModel` -- ``flux_limit(wave_obs)`` in erg/s/cm^2, ``inf`` outside
  the bandpass. :class:`ConstantFluxLimit` is the plan's initial model;
  :class:`TabulatedFluxLimit` is a wavelength-dependent replacement;
  :class:`RomanGrismSensitivity` is the Roman WFI grism model (wavelength, source size,
  exposure time and background, from the NASA WFI technical page) whose thresholds are
  S/N values. A model may also define ``secondary_flux_limit(wave_obs)``, the lower limit
  that applies to the other lines of a galaxy once one of its lines passes ``flux_limit``
  (see below). A model with ``uses_size = True`` takes the angular size of each galaxy as a
  second argument, ``flux_limit(wave_obs, size)``, and tells through ``size_spec`` /
  ``required_columns`` where the sizes come from (:mod:`romanmock.sizes`).
* :class:`DetectionModel` -- ``detect(flux, wave_obs, rng, size=None)`` returning a
  :class:`Detection` (boolean mask, measured flux, S/N). :class:`ThresholdDetection` is
  the deterministic ``F > F_thres`` rule; :class:`GaussianNoiseDetection` adds flux
  noise and a S/N cut as a first step towards detection scatter.

Two-tier detection: a galaxy enters the catalog only if at least one line exceeds the
*primary* limit inside the bandpass (default 1.0e-16 erg/s/cm^2). For such a galaxy every
other line inside the bandpass is detected if it exceeds the *secondary* limit (default
4.0e-17). A galaxy with no primary line has no detected line, however many of its lines
exceed the secondary limit. Setting the secondary limit equal to the primary one (or to
``None``) recovers the single-threshold rule. With :class:`RomanGrismSensitivity` the two
limits are S/N thresholds (5 and 2 by default) times the 5-sigma noise of each line.

Register additional models in :data:`SENSITIVITY_MODELS` / :data:`DETECTION_MODELS`
so they can be selected by name from a configuration file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .roman import (
    DEFAULT_LINE_FLUX_LIMIT,
    DEFAULT_PRIMARY_SNR,
    DEFAULT_SECONDARY_LINE_FLUX_LIMIT,
    DEFAULT_SECONDARY_SNR,
    DEFAULT_SNR_AT_LIMIT,
    GRISM_WAVE_RANGE_ANGSTROM,
    WFI_GRISM_5SIGMA_POINT,
    WFI_GRISM_5SIGMA_R50_0P3,
    WFI_GRISM_REFERENCE_BACKGROUND,
    WFI_GRISM_REFERENCE_EXPOSURE_S,
    WFI_GRISM_REFERENCE_SIZE_ARCSEC,
    WFI_GRISM_SENSITIVITY_WAVE_ANGSTROM,
)
from .sizes import SizeSpec


@runtime_checkable
class SensitivityModel(Protocol):
    wave_range: tuple[float, float]

    def flux_limit(self, wave_obs: np.ndarray) -> np.ndarray:
        """Minimum detectable line flux (erg/s/cm^2) at observed wavelength(s) [A]."""
        ...


class ConstantFluxLimit:
    """Constant flux thresholds inside a bandpass, infinite outside.

    ``flux_limit`` is the primary threshold. ``secondary_flux_limit`` is the threshold for
    the other lines of a galaxy that has a line above the primary one: ``"default"`` uses
    :data:`~romanmock.roman.DEFAULT_SECONDARY_LINE_FLUX_LIMIT` (4.0e-17), capped at the
    primary limit; ``None`` disables the second tier (secondary = primary); a number must
    satisfy ``0 < secondary <= flux_limit``.
    """

    uses_size = False

    def __init__(
        self,
        flux_limit: float = DEFAULT_LINE_FLUX_LIMIT,
        wave_range: tuple[float, float] = GRISM_WAVE_RANGE_ANGSTROM,
        secondary_flux_limit: float | str | None = "default",
    ):
        if flux_limit <= 0:
            raise ValueError("flux_limit must be positive")
        lo, hi = wave_range
        if not lo < hi:
            raise ValueError("wave_range must be (min, max) with min < max")
        self.flux_limit_value = float(flux_limit)
        self.wave_range = (float(lo), float(hi))
        if secondary_flux_limit is None:
            secondary = self.flux_limit_value
        elif isinstance(secondary_flux_limit, str):
            if secondary_flux_limit != "default":
                raise ValueError("secondary_flux_limit must be a number, None or 'default'")
            secondary = min(DEFAULT_SECONDARY_LINE_FLUX_LIMIT, self.flux_limit_value)
        else:
            secondary = float(secondary_flux_limit)
            if not 0 < secondary <= self.flux_limit_value:
                raise ValueError(f"secondary_flux_limit ({secondary:.3g}) must be positive and not exceed "
                                 f"flux_limit ({self.flux_limit_value:.3g})")
        self.secondary_flux_limit_value = secondary

    @property
    def two_tier(self) -> bool:
        return self.secondary_flux_limit_value < self.flux_limit_value

    def _inside(self, wave_obs):
        wave_obs = np.asarray(wave_obs, dtype=np.float64)
        lo, hi = self.wave_range
        return (wave_obs > lo) & (wave_obs < hi)

    def flux_limit(self, wave_obs, size=None):
        return np.where(self._inside(wave_obs), self.flux_limit_value, np.inf)

    def secondary_flux_limit(self, wave_obs, size=None):
        return np.where(self._inside(wave_obs), self.secondary_flux_limit_value, np.inf)

    def to_config(self) -> dict:
        return {"type": "constant", "flux_limit": self.flux_limit_value, "wave_range": list(self.wave_range),
                "secondary_flux_limit": self.secondary_flux_limit_value if self.two_tier else None}

    def describe(self) -> str:
        return f"F > {self.flux_limit_value:.2e}" + (
            f" (other lines > {self.secondary_flux_limit_value:.2e})" if self.two_tier else "")

    def __repr__(self):
        second = f", secondary {self.secondary_flux_limit_value:.3g}" if self.two_tier else ""
        return f"ConstantFluxLimit({self.flux_limit_value:.3g}{second} erg/s/cm2, {self.wave_range} A)"


class TabulatedFluxLimit:
    """Wavelength-dependent flux limit given on a grid (linear interpolation in log F).

    Outside ``wave_range`` (default: the tabulated extent) the limit is infinite.
    ``secondary_ratio`` (``0 < r <= 1``, default ``None`` = no second tier) sets the
    secondary limit to ``r`` times the tabulated one, e.g. 0.4 for 4e-17 at 1e-16.
    """

    uses_size = False

    def __init__(self, wave: np.ndarray, flux_limit: np.ndarray, wave_range=None, secondary_ratio: float | None = None):
        wave = np.asarray(wave, dtype=np.float64)
        flim = np.asarray(flux_limit, dtype=np.float64)
        if wave.ndim != 1 or wave.shape != flim.shape or wave.size < 2:
            raise ValueError("wave and flux_limit must be 1-D arrays of equal length >= 2")
        if np.any(np.diff(wave) <= 0):
            raise ValueError("wave must be strictly increasing")
        if np.any(flim <= 0):
            raise ValueError("flux limits must be positive")
        self.wave = wave
        self.flim = flim
        self.wave_range = tuple(map(float, wave_range)) if wave_range is not None else (float(wave[0]), float(wave[-1]))
        if secondary_ratio is not None and not 0 < secondary_ratio <= 1:
            raise ValueError("secondary_ratio must satisfy 0 < ratio <= 1")
        self.secondary_ratio = 1.0 if secondary_ratio is None else float(secondary_ratio)

    @property
    def two_tier(self) -> bool:
        return self.secondary_ratio < 1.0

    def secondary_flux_limit(self, wave_obs, size=None):
        return self.flux_limit(wave_obs) * self.secondary_ratio

    @classmethod
    def from_file(cls, path, wave_range=None, wave_unit: str = "angstrom", secondary_ratio: float | None = None):
        """Load a two-column text file (wavelength, flux limit). ``wave_unit`` may be
        ``"angstrom"`` or ``"micron"``."""
        arr = np.loadtxt(Path(path))
        wave, flim = arr[:, 0], arr[:, 1]
        if wave_unit == "micron":
            wave = wave * 1.0e4
        elif wave_unit != "angstrom":
            raise ValueError("wave_unit must be 'angstrom' or 'micron'")
        return cls(wave, flim, wave_range=wave_range, secondary_ratio=secondary_ratio)

    def flux_limit(self, wave_obs, size=None):
        wave_obs = np.asarray(wave_obs, dtype=np.float64)
        lo, hi = self.wave_range
        inside = (wave_obs > lo) & (wave_obs < hi)
        val = 10.0 ** np.interp(wave_obs, self.wave, np.log10(self.flim))
        return np.where(inside, val, np.inf)

    def to_config(self) -> dict:
        return {"type": "tabulated", "wave": self.wave.tolist(), "flux_limit": self.flim.tolist(),
                "wave_range": list(self.wave_range), "secondary_ratio": self.secondary_ratio if self.two_tier else None}

    def describe(self) -> str:
        return f"F > tabulated limit ({self.flim.min():.2e}-{self.flim.max():.2e})" + (
            f" (other lines > {self.secondary_ratio:g} x limit)" if self.two_tier else "")

    def __repr__(self):
        return f"TabulatedFluxLimit({self.wave.size} points, {self.wave_range} A)"


class RomanGrismSensitivity:
    """Roman WFI grism line sensitivity: wavelength, source size, exposure time, background.

    The NASA WFI technical page tabulates the 5-sigma line-flux limit of the grism at ten
    observed wavelengths for one hour at twice the minimum zodiacal background, for a point
    source and for a source with ``r50 = 0.3"`` (:mod:`romanmock.roman`). The 5-sigma flux
    of a line of a galaxy of angular half-light size ``r`` at observed wavelength ``lambda`` is

    ``F_5sigma(lambda, r) = F_5sigma,point(lambda) * [1 + (r / r_c(lambda))^2]^p * (t / 1 h)^(-1/2) * (b / 2)^(1/2)``

    with the point-source row interpolated linearly in log F; the size factor has one scale
    ``r_c(lambda)`` per tabulated wavelength, fixed so that ``r = 0.3"`` reproduces the
    extended-source row exactly. ``r_c`` (0.12-0.14") is an empirical transition scale set by
    the ratio of the two rows, not a measured PSF width. ``p`` = ``size_exponent``, 1/2 by
    default: the background noise grows with the square root of the number of pixels under
    the line, which grows as ``r^2`` when the line image spreads in both detector directions,
    so the limit tends to ``F ∝ r`` for large sizes, the large-size scaling the WFI page
    quotes. Between 0 and 0.3" only the two ends are tabulated, and beyond 0.3" the model
    follows that asymptote; other exponents are for sensitivity tests, not equally supported.
    Wavelengths between the bandpass edges and the first / last tabulated point take the
    nearest tabulated value (no throughput roll-off).

    The thresholds are S/N values: a line is above the primary limit when
    ``S/N = 5 F / F_5sigma >= snr`` (5 by default), and the other lines of such a galaxy are
    detected when ``S/N >= secondary_snr`` (2; ``None`` = single tier). ``size`` is a number
    in arcsec for every galaxy (0.3 by default; 0 = point source) or the column the size
    of each galaxy is read from (``"r50_disk_2d"``; :class:`~romanmock.sizes.SizeSpec`).
    ``exposure_time`` in seconds and ``background`` in units of the minimum zodiacal
    background (2 = the reference of the tables, ~1.4 = a typical HLWAS field).
    """

    uses_size = True

    def __init__(
        self,
        snr: float = DEFAULT_PRIMARY_SNR,
        secondary_snr: float | None = DEFAULT_SECONDARY_SNR,
        size=WFI_GRISM_REFERENCE_SIZE_ARCSEC,
        exposure_time: float = WFI_GRISM_REFERENCE_EXPOSURE_S,
        background: float = WFI_GRISM_REFERENCE_BACKGROUND,
        size_exponent: float = 0.5,
        wave_range: tuple[float, float] = GRISM_WAVE_RANGE_ANGSTROM,
        wave=WFI_GRISM_SENSITIVITY_WAVE_ANGSTROM,
        flux_5sigma_point=WFI_GRISM_5SIGMA_POINT,
        flux_5sigma_extended=WFI_GRISM_5SIGMA_R50_0P3,
        reference_size: float = WFI_GRISM_REFERENCE_SIZE_ARCSEC,
        reference_exposure_time: float = WFI_GRISM_REFERENCE_EXPOSURE_S,
        reference_background: float = WFI_GRISM_REFERENCE_BACKGROUND,
    ):
        if snr <= 0:
            raise ValueError("snr must be positive")
        if secondary_snr is None:
            secondary_snr = snr
        if not 0 < secondary_snr <= snr:
            raise ValueError(f"secondary_snr ({secondary_snr:g}) must be positive and not exceed snr ({snr:g})")
        if exposure_time <= 0 or background <= 0:
            raise ValueError("exposure_time and background must be positive")
        if size_exponent <= 0:
            raise ValueError("size_exponent must be positive")
        lo, hi = wave_range
        if not lo < hi:
            raise ValueError("wave_range must be (min, max) with min < max")
        wave = np.asarray(wave, dtype=np.float64)
        f_pt = np.asarray(flux_5sigma_point, dtype=np.float64)
        f_ext = np.asarray(flux_5sigma_extended, dtype=np.float64)
        if wave.ndim != 1 or wave.shape != f_pt.shape or wave.shape != f_ext.shape or wave.size < 2:
            raise ValueError("wave, flux_5sigma_point and flux_5sigma_extended must be 1-D arrays of equal length >= 2")
        if np.any(np.diff(wave) <= 0) or np.any(f_pt <= 0) or np.any(f_ext <= f_pt):
            raise ValueError("wave must increase and the extended-source limits must exceed the point-source ones")
        if reference_size <= 0:
            raise ValueError("reference_size must be positive")
        self.snr = float(snr)
        self.secondary_snr = float(secondary_snr)
        self.size_spec = SizeSpec.parse(size)
        self.exposure_time = float(exposure_time)
        self.background = float(background)
        self.size_exponent = float(size_exponent)
        self.wave_range = (float(lo), float(hi))
        self.wave, self.flux_5sigma_point, self.flux_5sigma_extended = wave, f_pt, f_ext
        self.reference_size = float(reference_size)
        self.reference_exposure_time = float(reference_exposure_time)
        self.reference_background = float(reference_background)
        # size scale per tabulated wavelength: (1 + (r_ref / r_c)^2)^p = F_ext / F_pt
        ratio = f_ext / f_pt
        self.size_scale = self.reference_size / np.sqrt(ratio ** (1.0 / self.size_exponent) - 1.0)
        # exposure time and background: F_lim ~ t^(-1/2) b^(1/2)
        self.depth_factor = float(np.sqrt(self.reference_exposure_time / self.exposure_time)
                                  * np.sqrt(self.background / self.reference_background))

    # ---- what the model needs from the catalog -------------------------------------
    @property
    def required_columns(self) -> list[str]:
        return self.size_spec.required_columns

    def angular_size(self, data=None, cosmology=None):
        """Angular size [arcsec] of the galaxies (a float for a fixed size); see :class:`~romanmock.sizes.SizeSpec`."""
        return self.size_spec.angular_size(data, cosmology)

    @property
    def two_tier(self) -> bool:
        return self.secondary_snr < self.snr

    @property
    def snr_at_limit(self) -> float:
        """S/N of a line exactly at the primary limit (= ``snr``)."""
        return self.snr

    # ---- the limits ----------------------------------------------------------------
    def _resolve_size(self, size):
        if size is None:
            if not self.size_spec.is_fixed:
                raise ValueError(f"RomanGrismSensitivity takes the size from the column {self.size_spec.column!r}: "
                                 "pass size=... (romanmock.detection.detect_lines(size=...))")
            return self.size_spec.fixed
        return np.asarray(size, dtype=np.float64)

    def flux_5sigma(self, wave_obs, size=None):
        """5-sigma line flux [erg/s/cm^2] at ``wave_obs`` [A] for a source of angular size ``size``
        [arcsec] (``None``: the fixed size of the model), for the model's exposure time and
        background; ``inf`` outside the bandpass."""
        wave_obs = np.asarray(wave_obs, dtype=np.float64)
        size = self._resolve_size(size)
        lo, hi = self.wave_range
        inside = (wave_obs > lo) & (wave_obs < hi)
        f_pt = 10.0 ** np.interp(wave_obs, self.wave, np.log10(self.flux_5sigma_point))
        r_c = np.interp(wave_obs, self.wave, self.size_scale)
        growth = (1.0 + (size / r_c) ** 2) ** self.size_exponent
        return np.where(inside, f_pt * growth * self.depth_factor, np.inf)

    def flux_limit(self, wave_obs, size=None):
        """Primary limit: the flux with ``S/N = snr``."""
        return self.flux_5sigma(wave_obs, size) * (self.snr / 5.0)

    def secondary_flux_limit(self, wave_obs, size=None):
        """Secondary limit: the flux with ``S/N = secondary_snr``."""
        return self.flux_5sigma(wave_obs, size) * (self.secondary_snr / 5.0)

    def snr_of(self, flux, wave_obs, size=None):
        """S/N of lines of flux ``flux`` at ``wave_obs`` (0 outside the bandpass)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            f5 = self.flux_5sigma(wave_obs, size)
            return np.where(np.isfinite(f5), 5.0 * np.asarray(flux, dtype=np.float64) / f5, 0.0)

    def nominal_flux_limit(self, wave: float = 15_000.0) -> float:
        """The primary limit at one wavelength for the fixed size, or the reference size (0.3") for
        a size column: a representative number, used for labels and for the lookup of reference dN/dz."""
        size = self.size_spec.fixed if self.size_spec.is_fixed else self.reference_size
        return float(self.flux_limit(np.array([wave]), size)[0])

    def nominal_secondary_limit(self, wave: float = 15_000.0) -> float | None:
        if not self.two_tier:
            return None
        return self.nominal_flux_limit(wave) * self.secondary_snr / self.snr

    # ---- bookkeeping ---------------------------------------------------------------
    def to_config(self) -> dict:
        cfg = {"type": "roman_grism", "snr": self.snr, "secondary_snr": self.secondary_snr if self.two_tier else None,
               "size": self.size_spec.to_config(), "exposure_time": self.exposure_time, "background": self.background,
               "size_exponent": self.size_exponent, "wave_range": list(self.wave_range)}
        default_table = (np.array_equal(self.wave, WFI_GRISM_SENSITIVITY_WAVE_ANGSTROM)
                         and np.array_equal(self.flux_5sigma_point, WFI_GRISM_5SIGMA_POINT)
                         and np.array_equal(self.flux_5sigma_extended, WFI_GRISM_5SIGMA_R50_0P3)
                         and self.reference_size == WFI_GRISM_REFERENCE_SIZE_ARCSEC
                         and self.reference_exposure_time == WFI_GRISM_REFERENCE_EXPOSURE_S
                         and self.reference_background == WFI_GRISM_REFERENCE_BACKGROUND)
        if not default_table:
            cfg.update(wave=self.wave.tolist(), flux_5sigma_point=self.flux_5sigma_point.tolist(),
                       flux_5sigma_extended=self.flux_5sigma_extended.tolist(), reference_size=self.reference_size,
                       reference_exposure_time=self.reference_exposure_time, reference_background=self.reference_background)
        return cfg

    def describe(self) -> str:
        hours = self.exposure_time / 3600.0
        return (f"S/N >= {self.snr:g}" + (f" (other lines S/N >= {self.secondary_snr:g})" if self.two_tier else "")
                + f", size {self.size_spec.describe()}, {hours:g} h, {self.background:g} x min zodi")

    def __repr__(self):
        return f"RomanGrismSensitivity({self.describe()}; {self.wave_range} A)"


SENSITIVITY_MODELS = {"constant": ConstantFluxLimit, "tabulated": TabulatedFluxLimit, "roman_grism": RomanGrismSensitivity}


def build_sensitivity(cfg) -> SensitivityModel:
    """Build a sensitivity model from a config dict (or pass a model through)."""
    if isinstance(cfg, SensitivityModel):
        return cfg
    cfg = dict(cfg or {})
    kind = cfg.pop("type", "constant")
    if kind == "constant":
        return ConstantFluxLimit(**cfg)
    if kind == "tabulated":
        if "file" in cfg:
            path = cfg.pop("file")
            return TabulatedFluxLimit.from_file(path, **cfg)
        return TabulatedFluxLimit(cfg["wave"], cfg["flux_limit"], wave_range=cfg.get("wave_range"),
                                  secondary_ratio=cfg.get("secondary_ratio"))
    try:
        return SENSITIVITY_MODELS[kind](**cfg)
    except KeyError:
        raise KeyError(f"unknown sensitivity model {kind!r}; known: {sorted(SENSITIVITY_MODELS)}")


# ---------------------------------------------------------------------------------
# helpers that work for every sensitivity model
# ---------------------------------------------------------------------------------
def uses_size(sensitivity) -> bool:
    """Does the model take a source size (``flux_limit(wave_obs, size)``)?"""
    return bool(getattr(sensitivity, "uses_size", False))


def needs_size(sensitivity) -> bool:
    """Does the model take the size of each galaxy from the catalog (a size column)? A model with a
    fixed size keeps that size whatever sizes are passed to the helpers below."""
    spec = getattr(sensitivity, "size_spec", None)
    return uses_size(sensitivity) and spec is not None and not spec.is_fixed


def primary_flux_limit(sensitivity, wave_obs, size=None) -> np.ndarray:
    """``sensitivity.flux_limit`` at ``wave_obs``, with the sizes when the model takes them from the catalog."""
    if size is not None and needs_size(sensitivity):
        return sensitivity.flux_limit(wave_obs, size)
    return sensitivity.flux_limit(wave_obs)


def secondary_flux_limit(sensitivity, wave_obs, primary_limit=None, size=None) -> np.ndarray:
    """Secondary limit of ``sensitivity`` at ``wave_obs``, never above the primary limit.
    Models without a ``secondary_flux_limit`` method have a single tier."""
    primary = primary_flux_limit(sensitivity, wave_obs, size) if primary_limit is None else primary_limit
    method = getattr(sensitivity, "secondary_flux_limit", None)
    if method is None:
        return primary
    second = method(wave_obs, size) if (size is not None and needs_size(sensitivity)) else method(wave_obs)
    return np.minimum(second, primary)


def nominal_flux_limit(sensitivity) -> float:
    """One number for the primary limit of a model: the constant value, the model's own
    ``nominal_flux_limit()``, or the limit in the middle of the bandpass."""
    if hasattr(sensitivity, "flux_limit_value"):
        return float(sensitivity.flux_limit_value)
    if hasattr(sensitivity, "nominal_flux_limit"):
        return float(sensitivity.nominal_flux_limit())
    lo, hi = sensitivity.wave_range
    return float(np.asarray(sensitivity.flux_limit(np.array([0.5 * (lo + hi)]))).ravel()[0])


def nominal_secondary_limit(sensitivity) -> float | None:
    """One number for the secondary limit of a model, ``None`` for a single tier."""
    if not getattr(sensitivity, "two_tier", False):
        return None
    if hasattr(sensitivity, "secondary_flux_limit_value"):
        return float(sensitivity.secondary_flux_limit_value)
    if hasattr(sensitivity, "nominal_secondary_limit"):
        return sensitivity.nominal_secondary_limit()
    lo, hi = sensitivity.wave_range
    return float(np.asarray(sensitivity.secondary_flux_limit(np.array([0.5 * (lo + hi)]))).ravel()[0])


def describe_sensitivity(sensitivity) -> str:
    """Short text of the limits of a model or of its config dict (``"F > 1.00e-16 (...)"``)."""
    if isinstance(sensitivity, dict):
        try:
            sensitivity = build_sensitivity(sensitivity)
        except Exception:                         # pragma: no cover - unknown model in a settings dict
            return str(sensitivity)
    if hasattr(sensitivity, "describe"):
        return sensitivity.describe()
    return repr(sensitivity)


#: a noisy detection model wants the base catalog at least this much deeper than the requested limit
NOISY_BASE_MARGIN = 1.5

#: raw sizes [arcsec] of the containment check: the sizes of the base catalog's galaxies before
#: ``scale`` and ``floor``, from a point source to the largest disks of the mock (~80 kpc)
CONTAINMENT_SIZES_ARCSEC = (0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0)


def limits_contained(requested, base, n_wave: int = 4001, sizes=CONTAINMENT_SIZES_ARCSEC) -> tuple[str | None, float]:
    """Is every galaxy ``requested`` can select in a base catalog built with ``base``?

    The primary limits are compared on a grid of wavelengths inside the requested bandpass
    and, when either model takes the sizes of the galaxies from the catalog, on a grid of raw
    sizes of the same galaxy: each model's own :class:`~romanmock.sizes.SizeSpec` (``scale``,
    ``floor``) is applied to the grid, so that two specs that assign different angular sizes
    to the same galaxy are compared for that galaxy. Two models reading the size from
    different quantities (column, unit, redshift column) cannot be compared and are refused,
    as is a requested ``size_exponent`` below the base one, whose limits cross for sizes
    beyond any finite grid (the limit grows as ``r^(2p)``). Returns ``(message, min_ratio)``:
    ``message`` is ``None`` when the requested limits are nowhere deeper than the base ones
    and the bandpass is inside the base one, else it says where; ``min_ratio`` is the
    smallest requested / base limit.
    """
    lo, hi = requested.wave_range
    wave = np.linspace(lo, hi, n_wave)[1:-1]
    req_spec = requested.size_spec if needs_size(requested) else None
    base_spec = base.size_spec if needs_size(base) else None
    if req_spec is not None and base_spec is not None:
        if not req_spec.same_source(base_spec):
            return (f"the requested sizes come from {req_spec.describe()} and the sizes of the base catalog from "
                    f"{base_spec.describe()}: the limits cannot be compared galaxy by galaxy; rebuild the base catalog "
                    "with the same size column, a fixed size or a constant flux limit", 0.0)
        p_req, p_base = getattr(requested, "size_exponent", None), getattr(base, "size_exponent", None)
        if p_req is not None and p_base is not None and p_req < p_base - 1e-12:
            return (f"requested size_exponent {p_req:g} is below the base catalog's {p_base:g}: the requested limit is "
                    "deeper than the base one for large galaxies", 0.0)
    with_size = req_spec is not None or base_spec is not None
    raw_sizes = np.asarray(sizes, dtype=np.float64)
    if with_size:
        # the raw sizes a galaxy of the base catalog can have: the grid, also through each scale
        for spec in (req_spec, base_spec):
            if spec is not None:
                raw_sizes = np.concatenate([raw_sizes, np.asarray(sizes) / spec.scale])
        raw_sizes = np.unique(raw_sizes)
    min_ratio = np.inf
    for raw in (raw_sizes if with_size else (None,)):
        if raw is None:
            size_req = size_base = None
        else:
            size_req = float(req_spec.transform(raw)) if req_spec is not None else None
            size_base = float(base_spec.transform(raw)) if base_spec is not None else None
        req = primary_flux_limit(requested, wave, size_req)
        ref = primary_flux_limit(base, wave, size_base)
        inside = np.isfinite(req)
        if np.any(inside & ~np.isfinite(ref)):
            return (f"requested bandpass {requested.wave_range} extends beyond the bandpass of the base catalog "
                    f"{tuple(base.wave_range)}", 0.0)
        if not inside.any():
            continue
        ratio = req[inside] / ref[inside]
        k = int(np.argmin(ratio))
        min_ratio = min(min_ratio, float(ratio[k]))
        if ratio[k] < 1.0 - 1e-9:
            where = f"at {wave[inside][k]:.0f} A"
            if size_req is not None:
                where += f" for a size of {size_req:g} arcsec"
                if size_base is not None and size_base != size_req:
                    where += f" ({size_base:g} arcsec in the base catalog)"
            elif size_base is not None:
                where += f" for a galaxy of {size_base:g} arcsec in the base catalog"
            return (f"requested flux limit {req[inside][k]:.3g} is deeper than the base catalog limit "
                    f"{ref[inside][k]:.3g} ({where})", float(ratio[k]))
    return None, float(min_ratio)


def apply_two_tier(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Combine per-line masks ``(n, n_lines)``: a line is detected if it passes the primary
    cut, or if it passes the secondary cut and any line of the same galaxy passes the
    primary one."""
    anchored = primary.any(axis=-1, keepdims=True)
    return primary | (anchored & secondary)


# ---------------------------------------------------------------------------------
# Detection models
# ---------------------------------------------------------------------------------
@dataclass
class Detection:
    """Result of a detection model for an ``(n, n_features)`` flux array."""

    detected: np.ndarray   # bool
    flux_measured: np.ndarray  # erg/s/cm^2 (equal to input flux for noiseless models)
    snr: np.ndarray        # nominal or realised signal-to-noise ratio (0 outside bandpass)
    primary: np.ndarray | None = None   # bool: lines passing the primary limit (None: same as detected)


@runtime_checkable
class DetectionModel(Protocol):
    sensitivity: SensitivityModel

    def detect(self, flux: np.ndarray, wave_obs: np.ndarray, rng: np.random.Generator) -> Detection:
        ...


def _snr_at_limit(sensitivity, value) -> float:
    """S/N of a line at the primary limit: the given value, else the model's own (``snr_at_limit``),
    else the project default (6.5, the HLSS reference depth)."""
    if value is not None:
        return float(value)
    return float(getattr(sensitivity, "snr_at_limit", DEFAULT_SNR_AT_LIMIT))


class ThresholdDetection:
    """Deterministic rule of the plan: detected iff ``F > F_lim(lambda_obs[, size])`` and the
    line falls inside the bandpass; with a two-tier sensitivity, the other lines of a
    galaxy with such a line are detected iff ``F > F_lim,secondary(lambda_obs[, size])``.
    ``snr`` is the nominal ``snr_at_limit * F / F_lim`` (primary limit); ``snr_at_limit``
    defaults to the value of the sensitivity model (the ``snr`` of
    :class:`RomanGrismSensitivity`, where it makes ``snr`` the estimated S/N of the line)
    or 6.5."""

    def __init__(self, sensitivity: SensitivityModel | None = None, snr_at_limit: float | None = None):
        self.sensitivity = sensitivity if sensitivity is not None else ConstantFluxLimit()
        self.snr_at_limit = _snr_at_limit(self.sensitivity, snr_at_limit)

    def detect(self, flux, wave_obs, rng=None, size=None) -> Detection:
        flux = np.asarray(flux, dtype=np.float64)
        flim = primary_flux_limit(self.sensitivity, wave_obs, size)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(np.isfinite(flim), self.snr_at_limit * flux / flim, 0.0)
        primary = flux > flim
        detected = apply_two_tier(primary, flux > secondary_flux_limit(self.sensitivity, wave_obs, flim, size))
        return Detection(detected=detected, flux_measured=flux, snr=snr, primary=primary)

    def to_config(self) -> dict:
        return {"type": "threshold", "snr_at_limit": self.snr_at_limit}


class GaussianNoiseDetection:
    """Flux noise ``sigma = F_lim / snr_at_limit`` added to each line; detected iff the
    measured flux exceeds ``snr_threshold * sigma`` (and the line is in the bandpass).

    With ``snr_threshold == snr_at_limit`` (the default) this reduces to the threshold rule on
    average but introduces detection scatter near the limit (plan comment 6). With a
    two-tier sensitivity the other lines of a galaxy with a primary detection need
    ``S/N > snr_threshold * F_lim,secondary / F_lim``; pure noise can then pass the
    lower cut, as it would in real spectra. With :class:`RomanGrismSensitivity`
    ``sigma`` is the 1-sigma noise of the line (``F_5sigma / 5``).
    """

    def __init__(
        self,
        sensitivity: SensitivityModel | None = None,
        snr_at_limit: float | None = None,
        snr_threshold: float | None = None,
    ):
        self.sensitivity = sensitivity if sensitivity is not None else ConstantFluxLimit()
        self.snr_at_limit = _snr_at_limit(self.sensitivity, snr_at_limit)
        self.snr_threshold = self.snr_at_limit if snr_threshold is None else float(snr_threshold)

    def detect(self, flux, wave_obs, rng: np.random.Generator, size=None) -> Detection:
        flux = np.asarray(flux, dtype=np.float64)
        flim = primary_flux_limit(self.sensitivity, wave_obs, size)
        inside = np.isfinite(flim)
        sigma = np.where(inside, flim / self.snr_at_limit, np.inf)
        noise = rng.standard_normal(flux.shape) * np.where(inside, sigma, 0.0)
        flux_meas = flux + noise
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(inside, flux_meas / sigma, 0.0)
        primary = inside & (snr > self.snr_threshold)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(inside, secondary_flux_limit(self.sensitivity, wave_obs, flim, size) / flim, 1.0)
        detected = apply_two_tier(primary, inside & (snr > self.snr_threshold * ratio))
        return Detection(detected=detected, flux_measured=flux_meas, snr=snr, primary=primary)

    def to_config(self) -> dict:
        return {"type": "gaussian_noise", "snr_at_limit": self.snr_at_limit, "snr_threshold": self.snr_threshold}


DETECTION_MODELS = {"threshold": ThresholdDetection, "gaussian_noise": GaussianNoiseDetection}


def build_detection(cfg, sensitivity: SensitivityModel) -> DetectionModel:
    if isinstance(cfg, DetectionModel):
        return cfg
    cfg = dict(cfg or {})
    kind = cfg.pop("type", "threshold")
    try:
        cls = DETECTION_MODELS[kind]
    except KeyError:
        raise KeyError(f"unknown detection model {kind!r}; known: {sorted(DETECTION_MODELS)}")
    return cls(sensitivity=sensitivity, **cfg)

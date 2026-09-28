"""Sensitivity (flux limit versus wavelength) and detection models.

Two small protocols keep the first-stage physics replaceable (plan comment 6):

* :class:`SensitivityModel` -- ``flux_limit(wave_obs)`` in erg/s/cm^2, ``inf`` outside
  the bandpass. :class:`ConstantFluxLimit` is the plan's initial model;
  :class:`TabulatedFluxLimit` is a wavelength-dependent replacement. A model may also
  define ``secondary_flux_limit(wave_obs)``, the lower limit that applies to the other
  lines of a galaxy once one of its lines passes ``flux_limit`` (see below).
* :class:`DetectionModel` -- ``detect(flux, wave_obs, rng)`` returning a
  :class:`Detection` (boolean mask, measured flux, S/N). :class:`ThresholdDetection` is
  the deterministic ``F > F_thres`` rule; :class:`GaussianNoiseDetection` adds flux
  noise and a S/N cut as a first step towards detection scatter.

Two-tier detection: a galaxy enters the catalog only if at least one line exceeds the
*primary* limit inside the bandpass (default 1.0e-16 erg/s/cm^2). For such a galaxy every
other line inside the bandpass is detected if it exceeds the *secondary* limit (default
4.0e-17). A galaxy with no primary line has no detected line, however many of its lines
exceed the secondary limit. Setting the secondary limit equal to the primary one (or to
``None``) recovers the single-threshold rule.

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
    DEFAULT_SECONDARY_LINE_FLUX_LIMIT,
    DEFAULT_SNR_AT_LIMIT,
    GRISM_WAVE_RANGE_ANGSTROM,
)


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

    def flux_limit(self, wave_obs):
        return np.where(self._inside(wave_obs), self.flux_limit_value, np.inf)

    def secondary_flux_limit(self, wave_obs):
        return np.where(self._inside(wave_obs), self.secondary_flux_limit_value, np.inf)

    def to_config(self) -> dict:
        return {"type": "constant", "flux_limit": self.flux_limit_value, "wave_range": list(self.wave_range),
                "secondary_flux_limit": self.secondary_flux_limit_value if self.two_tier else None}

    def __repr__(self):
        second = f", secondary {self.secondary_flux_limit_value:.3g}" if self.two_tier else ""
        return f"ConstantFluxLimit({self.flux_limit_value:.3g}{second} erg/s/cm2, {self.wave_range} A)"


class TabulatedFluxLimit:
    """Wavelength-dependent flux limit given on a grid (linear interpolation in log F).

    Outside ``wave_range`` (default: the tabulated extent) the limit is infinite.
    ``secondary_ratio`` (``0 < r <= 1``, default ``None`` = no second tier) sets the
    secondary limit to ``r`` times the tabulated one, e.g. 0.4 for 4e-17 at 1e-16.
    """

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

    def secondary_flux_limit(self, wave_obs):
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

    def flux_limit(self, wave_obs):
        wave_obs = np.asarray(wave_obs, dtype=np.float64)
        lo, hi = self.wave_range
        inside = (wave_obs > lo) & (wave_obs < hi)
        val = 10.0 ** np.interp(wave_obs, self.wave, np.log10(self.flim))
        return np.where(inside, val, np.inf)

    def to_config(self) -> dict:
        return {"type": "tabulated", "wave": self.wave.tolist(), "flux_limit": self.flim.tolist(),
                "wave_range": list(self.wave_range), "secondary_ratio": self.secondary_ratio if self.two_tier else None}

    def __repr__(self):
        return f"TabulatedFluxLimit({self.wave.size} points, {self.wave_range} A)"


SENSITIVITY_MODELS = {"constant": ConstantFluxLimit, "tabulated": TabulatedFluxLimit}


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


def secondary_flux_limit(sensitivity, wave_obs, primary_limit=None) -> np.ndarray:
    """Secondary limit of ``sensitivity`` at ``wave_obs``, never above the primary limit.
    Models without a ``secondary_flux_limit`` method have a single tier."""
    primary = sensitivity.flux_limit(wave_obs) if primary_limit is None else primary_limit
    method = getattr(sensitivity, "secondary_flux_limit", None)
    if method is None:
        return primary
    return np.minimum(method(wave_obs), primary)


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


class ThresholdDetection:
    """Deterministic rule of the plan: detected iff ``F > F_lim(lambda_obs)`` and the
    line falls inside the bandpass; with a two-tier sensitivity, the other lines of a
    galaxy with such a line are detected iff ``F > F_lim,secondary(lambda_obs)``.
    ``snr`` is the nominal ``snr_at_limit * F / F_lim`` (primary limit)."""

    def __init__(self, sensitivity: SensitivityModel | None = None, snr_at_limit: float = DEFAULT_SNR_AT_LIMIT):
        self.sensitivity = sensitivity if sensitivity is not None else ConstantFluxLimit()
        self.snr_at_limit = float(snr_at_limit)

    def detect(self, flux, wave_obs, rng=None) -> Detection:
        flux = np.asarray(flux, dtype=np.float64)
        flim = self.sensitivity.flux_limit(wave_obs)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(np.isfinite(flim), self.snr_at_limit * flux / flim, 0.0)
        primary = flux > flim
        detected = apply_two_tier(primary, flux > secondary_flux_limit(self.sensitivity, wave_obs, flim))
        return Detection(detected=detected, flux_measured=flux, snr=snr, primary=primary)

    def to_config(self) -> dict:
        return {"type": "threshold", "snr_at_limit": self.snr_at_limit}


class GaussianNoiseDetection:
    """Flux noise ``sigma = F_lim / snr_at_limit`` added to each line; detected iff the
    measured flux exceeds ``snr_threshold * sigma`` (and the line is in the bandpass).

    With ``snr_threshold == snr_at_limit`` this reduces to the threshold rule on
    average but introduces detection scatter near the limit (plan comment 6). With a
    two-tier sensitivity the other lines of a galaxy with a primary detection need
    ``S/N > snr_threshold * F_lim,secondary / F_lim``; pure noise can then pass the
    lower cut, as it would in real spectra.
    """

    def __init__(
        self,
        sensitivity: SensitivityModel | None = None,
        snr_at_limit: float = DEFAULT_SNR_AT_LIMIT,
        snr_threshold: float = DEFAULT_SNR_AT_LIMIT,
    ):
        self.sensitivity = sensitivity if sensitivity is not None else ConstantFluxLimit()
        self.snr_at_limit = float(snr_at_limit)
        self.snr_threshold = float(snr_threshold)

    def detect(self, flux, wave_obs, rng: np.random.Generator) -> Detection:
        flux = np.asarray(flux, dtype=np.float64)
        flim = self.sensitivity.flux_limit(wave_obs)
        inside = np.isfinite(flim)
        sigma = np.where(inside, flim / self.snr_at_limit, np.inf)
        noise = rng.standard_normal(flux.shape) * np.where(inside, sigma, 0.0)
        flux_meas = flux + noise
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(inside, flux_meas / sigma, 0.0)
        primary = inside & (snr > self.snr_threshold)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(inside, secondary_flux_limit(self.sensitivity, wave_obs, flim) / flim, 1.0)
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

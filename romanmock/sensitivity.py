"""Sensitivity (flux limit versus wavelength) and detection models.

Two small protocols keep the first-stage physics replaceable (plan comment 6):

* :class:`SensitivityModel` -- ``flux_limit(wave_obs)`` in erg/s/cm^2, ``inf`` outside
  the bandpass. :class:`ConstantFluxLimit` is the plan's initial model;
  :class:`TabulatedFluxLimit` is a wavelength-dependent replacement.
* :class:`DetectionModel` -- ``detect(flux, wave_obs, rng)`` returning a
  :class:`Detection` (boolean mask, measured flux, S/N). :class:`ThresholdDetection` is
  the deterministic ``F > F_thres`` rule; :class:`GaussianNoiseDetection` adds flux
  noise and a S/N cut as a first step towards detection scatter.

Register additional models in :data:`SENSITIVITY_MODELS` / :data:`DETECTION_MODELS`
so they can be selected by name from a configuration file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .roman import DEFAULT_LINE_FLUX_LIMIT, DEFAULT_SNR_AT_LIMIT, GRISM_WAVE_RANGE_ANGSTROM


@runtime_checkable
class SensitivityModel(Protocol):
    wave_range: tuple[float, float]

    def flux_limit(self, wave_obs: np.ndarray) -> np.ndarray:
        """Minimum detectable line flux (erg/s/cm^2) at observed wavelength(s) [A]."""
        ...


class ConstantFluxLimit:
    """Constant flux threshold inside a bandpass, infinite outside."""

    def __init__(
        self,
        flux_limit: float = DEFAULT_LINE_FLUX_LIMIT,
        wave_range: tuple[float, float] = GRISM_WAVE_RANGE_ANGSTROM,
    ):
        if flux_limit <= 0:
            raise ValueError("flux_limit must be positive")
        lo, hi = wave_range
        if not lo < hi:
            raise ValueError("wave_range must be (min, max) with min < max")
        self.flux_limit_value = float(flux_limit)
        self.wave_range = (float(lo), float(hi))

    def flux_limit(self, wave_obs):
        wave_obs = np.asarray(wave_obs, dtype=np.float64)
        lo, hi = self.wave_range
        inside = (wave_obs > lo) & (wave_obs < hi)
        return np.where(inside, self.flux_limit_value, np.inf)

    def to_config(self) -> dict:
        return {"type": "constant", "flux_limit": self.flux_limit_value, "wave_range": list(self.wave_range)}

    def __repr__(self):
        return f"ConstantFluxLimit({self.flux_limit_value:.3g} erg/s/cm2, {self.wave_range} A)"


class TabulatedFluxLimit:
    """Wavelength-dependent flux limit given on a grid (linear interpolation in log F).

    Outside ``wave_range`` (default: the tabulated extent) the limit is infinite.
    """

    def __init__(self, wave: np.ndarray, flux_limit: np.ndarray, wave_range=None):
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

    @classmethod
    def from_file(cls, path, wave_range=None, wave_unit: str = "angstrom"):
        """Load a two-column text file (wavelength, flux limit). ``wave_unit`` may be
        ``"angstrom"`` or ``"micron"``."""
        arr = np.loadtxt(Path(path))
        wave, flim = arr[:, 0], arr[:, 1]
        if wave_unit == "micron":
            wave = wave * 1.0e4
        elif wave_unit != "angstrom":
            raise ValueError("wave_unit must be 'angstrom' or 'micron'")
        return cls(wave, flim, wave_range=wave_range)

    def flux_limit(self, wave_obs):
        wave_obs = np.asarray(wave_obs, dtype=np.float64)
        lo, hi = self.wave_range
        inside = (wave_obs > lo) & (wave_obs < hi)
        val = 10.0 ** np.interp(wave_obs, self.wave, np.log10(self.flim))
        return np.where(inside, val, np.inf)

    def to_config(self) -> dict:
        return {"type": "tabulated", "wave": self.wave.tolist(), "flux_limit": self.flim.tolist(),
                "wave_range": list(self.wave_range)}

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
        return TabulatedFluxLimit(cfg["wave"], cfg["flux_limit"], wave_range=cfg.get("wave_range"))
    try:
        return SENSITIVITY_MODELS[kind](**cfg)
    except KeyError:
        raise KeyError(f"unknown sensitivity model {kind!r}; known: {sorted(SENSITIVITY_MODELS)}")


# ---------------------------------------------------------------------------------
# Detection models
# ---------------------------------------------------------------------------------
@dataclass
class Detection:
    """Result of a detection model for an ``(n, n_features)`` flux array."""

    detected: np.ndarray   # bool
    flux_measured: np.ndarray  # erg/s/cm^2 (equal to input flux for noiseless models)
    snr: np.ndarray        # nominal or realised signal-to-noise ratio (0 outside bandpass)


@runtime_checkable
class DetectionModel(Protocol):
    sensitivity: SensitivityModel

    def detect(self, flux: np.ndarray, wave_obs: np.ndarray, rng: np.random.Generator) -> Detection:
        ...


class ThresholdDetection:
    """Deterministic rule of the plan: detected iff ``F > F_lim(lambda_obs)`` and the
    line falls inside the bandpass. ``snr`` is the nominal ``snr_at_limit * F / F_lim``."""

    def __init__(self, sensitivity: SensitivityModel | None = None, snr_at_limit: float = DEFAULT_SNR_AT_LIMIT):
        self.sensitivity = sensitivity if sensitivity is not None else ConstantFluxLimit()
        self.snr_at_limit = float(snr_at_limit)

    def detect(self, flux, wave_obs, rng=None) -> Detection:
        flux = np.asarray(flux, dtype=np.float64)
        flim = self.sensitivity.flux_limit(wave_obs)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(np.isfinite(flim), self.snr_at_limit * flux / flim, 0.0)
        detected = flux > flim
        return Detection(detected=detected, flux_measured=flux, snr=snr)

    def to_config(self) -> dict:
        return {"type": "threshold", "snr_at_limit": self.snr_at_limit}


class GaussianNoiseDetection:
    """Flux noise ``sigma = F_lim / snr_at_limit`` added to each line; detected iff the
    measured flux exceeds ``snr_threshold * sigma`` (and the line is in the bandpass).

    With ``snr_threshold == snr_at_limit`` this reduces to the threshold rule on
    average but introduces detection scatter near the limit (plan comment 6).
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
        detected = inside & (snr > self.snr_threshold)
        return Detection(detected=detected, flux_measured=flux_meas, snr=snr)

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

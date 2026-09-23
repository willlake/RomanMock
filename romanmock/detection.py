"""Step 1-2 of the plan: observed wavelengths, detectability, and the bitmask.

Bit ``i`` of ``bitmask`` is set when feature ``i`` of the :class:`~romanmock.lines.LineSet`
is detected. ``n_det = popcount(bitmask)``; galaxies with ``bitmask == 0`` are not
part of the emission-line-selected sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lines import LineSet, get_line_set
from .sensitivity import Detection, DetectionModel, ThresholdDetection

BITMASK_DTYPE = np.uint32


@dataclass
class DetectionResult:
    """Per-galaxy detection information (arrays of shape ``(n,)`` or ``(n, n_features)``)."""

    wave_obs: np.ndarray      # (n, nf) observed wavelength [A]
    flux: np.ndarray          # (n, nf) true flux [erg/s/cm^2]
    flux_measured: np.ndarray  # (n, nf) measured flux (== flux for noiseless models)
    snr: np.ndarray           # (n, nf)
    detected: np.ndarray      # (n, nf) bool
    bitmask: np.ndarray       # (n,) uint32
    n_det: np.ndarray         # (n,) uint8

    @property
    def n_features(self) -> int:
        return self.detected.shape[1]


def observed_wavelength(rest_waves, z_rsd) -> np.ndarray:
    """``lambda_obs = lambda_0 (1 + z_rsd)`` as an ``(n, n_features)`` array."""
    return np.asarray(rest_waves, dtype=np.float64)[None, :] * (1.0 + np.asarray(z_rsd, dtype=np.float64))[:, None]


def pack_bitmask(detected: np.ndarray) -> np.ndarray:
    """Pack an ``(n, n_features)`` boolean array into a uint32 bitmask (bit i = column i)."""
    detected = np.asarray(detected, dtype=bool)
    nf = detected.shape[1]
    weights = (np.uint64(1) << np.arange(nf, dtype=np.uint64)).astype(np.uint64)
    packed = (detected.astype(np.uint64) * weights[None, :]).sum(axis=1)
    return packed.astype(BITMASK_DTYPE)


def unpack_bitmask(bitmask: np.ndarray, n_features: int) -> np.ndarray:
    """Inverse of :func:`pack_bitmask`."""
    bm = np.asarray(bitmask).astype(np.uint64)
    bits = np.arange(n_features, dtype=np.uint64)
    return ((bm[:, None] >> bits[None, :]) & np.uint64(1)).astype(bool)


def popcount(bitmask: np.ndarray) -> np.ndarray:
    """Number of set bits (``N_det``)."""
    bm = np.asarray(bitmask).astype(np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(bm).astype(np.uint8)
    count = np.zeros(bm.shape, dtype=np.uint8)
    while np.any(bm):
        count += (bm & np.uint64(1)).astype(np.uint8)
        bm >>= np.uint64(1)
    return count


def detect_lines(
    flux: np.ndarray,
    z_rsd: np.ndarray,
    line_set="roman_transitions",
    detection_model: DetectionModel | None = None,
    rng: np.random.Generator | None = None,
    component_flux: dict | None = None,
) -> DetectionResult:
    """Apply the detection model to feature fluxes and build the bitmask.

    Parameters
    ----------
    flux : (n, n_features) array
        Observed feature fluxes in erg/s/cm^2 (see :func:`romanmock.fluxes.feature_fluxes`).
    z_rsd : (n,) array
        Redshift including peculiar velocity (diffsky ``redshift_obs``).
    line_set : LineSet or preset name
    detection_model : DetectionModel, optional
        Defaults to :class:`ThresholdDetection` with the Roman grism constant limit.
    rng : numpy Generator, optional
        Needed by stochastic detection models.
    component_flux : dict, optional
        ``{luminosity column: (n,) flux}`` from :func:`romanmock.fluxes.component_fluxes`;
        required when the line set has separated multi-component features.
    """
    line_set = get_line_set(line_set)
    flux = np.asarray(flux, dtype=np.float64)
    if flux.ndim != 2 or flux.shape[1] != len(line_set):
        raise ValueError(f"flux must have shape (n, {len(line_set)}); got {flux.shape}")
    model = detection_model if detection_model is not None else ThresholdDetection()
    rng = rng if rng is not None else np.random.default_rng()
    z_rsd = np.asarray(z_rsd, dtype=np.float64)
    wave_obs = observed_wavelength(line_set.rest_waves, z_rsd)

    if not line_set.has_separated_components:
        det: Detection = model.detect(flux, wave_obs, rng)
        detected, flux_measured, snr = det.detected, det.flux_measured, det.snr
    else:
        # Separated multi-component features (e.g. the plan's [O III] = 4959 + 5007 species)
        # are tested transition by transition: each component has its own observed
        # wavelength and bandpass test; the feature is detected when any component is.
        if component_flux is None:
            raise ValueError("line set has separated multi-component features; pass component_flux "
                             "(see romanmock.fluxes.component_fluxes)")
        cols, waves, owner = [], [], []
        for i, feat in enumerate(line_set):
            if feat.is_blend:
                cols.append(flux[:, i]); waves.append(feat.rest_wave); owner.append(i)
            else:
                for c in feat.components:
                    cols.append(np.asarray(component_flux[c], dtype=np.float64))
                    waves.append(line_set.component_wave(c)); owner.append(i)
        owner = np.asarray(owner)
        f_cols = np.stack(cols, axis=1)
        w_cols = observed_wavelength(np.asarray(waves), z_rsd)
        det = model.detect(f_cols, w_cols, rng)
        n, nf = flux.shape
        detected = np.zeros((n, nf), dtype=bool)
        flux_measured = np.zeros((n, nf))
        snr = np.zeros((n, nf))
        for i in range(nf):
            sel = owner == i
            detected[:, i] = det.detected[:, sel].any(axis=1)
            flux_measured[:, i] = det.flux_measured[:, sel].sum(axis=1)
            snr[:, i] = det.snr[:, sel].max(axis=1)
    bitmask = pack_bitmask(detected)
    return DetectionResult(
        wave_obs=wave_obs,
        flux=flux,
        flux_measured=flux_measured,
        snr=snr,
        detected=detected,
        bitmask=bitmask,
        n_det=popcount(bitmask),
    )

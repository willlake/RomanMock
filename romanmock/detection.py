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
    primary: np.ndarray | None = None   # (n, nf) bool: lines above the primary flux limit

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


def one_line_per_galaxy(primary: np.ndarray, snr: np.ndarray, prefer: int | None = None) -> np.ndarray:
    """One detected line per galaxy, the rule without a secondary limit (every detected galaxy a single-line galaxy):
    of the lines above the primary limit, line ``prefer`` when it is one of them (the target line of a catalog: every
    galaxy with it above the primary limit is a single-line galaxy of that line), else the one with the highest S/N."""
    score = np.where(primary, snr, -np.inf)
    if prefer is not None:
        score[:, prefer] = np.where(primary[:, prefer], np.inf, -np.inf)
    top = np.argmax(score, axis=-1)
    out = np.zeros(primary.shape, dtype=bool)
    rows = np.flatnonzero(primary.any(axis=-1))
    out[rows, top[rows]] = True
    return out


def detect_lines(
    flux: np.ndarray,
    z_rsd: np.ndarray,
    line_set="roman_transitions",
    detection_model: DetectionModel | None = None,
    rng: np.random.Generator | None = None,
    component_flux: dict | None = None,
    size=None,
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
        Defaults to :class:`ThresholdDetection` with the Roman grism constant limits
        (1.0e-16 for the first line of a galaxy, 4.0e-17 for its other lines).
    rng : numpy Generator, optional
        Needed by stochastic detection models.
    component_flux : dict, optional
        ``{luminosity column: (n,) flux}`` from :func:`romanmock.fluxes.component_fluxes`;
        required when the line set has separated multi-component features.
    size : float or (n,) array, optional
        Angular half-light size of each galaxy [arcsec], for sensitivity models that depend
        on it (:class:`~romanmock.sensitivity.RomanGrismSensitivity` with a size column;
        ``model.sensitivity.angular_size(data, cosmology)`` computes it). Ignored by the
        other models.

    A detection model with ``single_line`` (True or a line name) keeps one line per galaxy
    (:func:`one_line_per_galaxy`); one with ``primary_line`` (a line name) detects only the
    galaxies with that line above the primary limit.
    """
    line_set = get_line_set(line_set)
    flux = np.asarray(flux, dtype=np.float64)
    if flux.ndim != 2 or flux.shape[1] != len(line_set):
        raise ValueError(f"flux must have shape (n, {len(line_set)}); got {flux.shape}")
    model = detection_model if detection_model is not None else ThresholdDetection()
    rng = rng if rng is not None else np.random.default_rng()
    z_rsd = np.asarray(z_rsd, dtype=np.float64)
    wave_obs = observed_wavelength(line_set.rest_waves, z_rsd)
    kw = {}
    if size is not None:
        size = np.asarray(size, dtype=np.float64)
        if size.ndim == 1:
            if size.shape[0] != flux.shape[0]:
                raise ValueError(f"size must have one value per galaxy ({flux.shape[0]}); got {size.shape}")
            size = size[:, None]                       # broadcasts against (n, n_features) wavelengths
        elif size.ndim != 0:
            raise ValueError("size is a number or an (n,) array")
        kw["size"] = size

    if not line_set.has_separated_components:
        det: Detection = model.detect(flux, wave_obs, rng, **kw)
        detected, flux_measured, snr = det.detected, det.flux_measured, det.snr
        primary = det.primary if det.primary is not None else detected
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
        det = model.detect(f_cols, w_cols, rng, **kw)
        n, nf = flux.shape
        detected = np.zeros((n, nf), dtype=bool)
        primary = np.zeros((n, nf), dtype=bool)
        comp_primary = det.primary if det.primary is not None else det.detected
        flux_measured = np.zeros((n, nf))
        snr = np.zeros((n, nf))
        for i in range(nf):
            sel = owner == i
            detected[:, i] = det.detected[:, sel].any(axis=1)
            primary[:, i] = comp_primary[:, sel].any(axis=1)
            flux_measured[:, i] = det.flux_measured[:, sel].sum(axis=1)
            snr[:, i] = det.snr[:, sel].max(axis=1)
    need = getattr(model, "primary_line", None)
    if need:                                    # the sample of one line: the galaxies with it above the primary limit
        has = primary[:, line_set.names.index(need)][:, None]
        detected, primary = detected & has, primary & has
    rule = getattr(model, "single_line", False)
    if rule:                                    # no secondary limit: one line per galaxy
        detected = one_line_per_galaxy(primary, snr, None if rule is True else line_set.names.index(rule))
    bitmask = pack_bitmask(detected)
    return DetectionResult(
        wave_obs=wave_obs,
        flux=flux,
        flux_measured=flux_measured,
        snr=snr,
        detected=detected,
        bitmask=bitmask,
        n_det=popcount(bitmask),
        primary=primary,
    )

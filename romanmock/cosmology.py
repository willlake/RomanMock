"""Luminosity distances for the luminosity -> flux conversion.

The catalog's own cosmology is taken from OpenCosmo (``dataset.cosmology`` returns an
``astropy.cosmology`` object built from the file header). Two backends are offered:

* ``"astropy"`` (default): ``cosmology.luminosity_distance``; fast and exact.
* ``"dsps"``: ``dsps.cosmology.luminosity_distance_to_z``, the JAX routine diffsky uses
  internally for distance moduli. Agrees with astropy to ~5e-6 for this catalog and is
  provided for strict consistency checks against diffsky photometry.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u

MPC_IN_CM = u.Mpc.to(u.cm)


def cosmology_from_catalog(catalog):
    """Return the astropy cosmology stored in an OpenCosmo dataset/lightcone header."""
    return catalog.cosmology


def _dsps_luminosity_distance_mpc(z: np.ndarray, cosmology) -> np.ndarray:
    import jax
    from dsps.cosmology import luminosity_distance_to_z

    try:  # diffsky helper that maps an astropy cosmology onto (Om0, w0, wa, h)
        from diffsky.data_loaders.opencosmo_utils.utils import prep_cosmology_parameters

        cp = prep_cosmology_parameters(cosmology)
        Om0, w0, wa, h = cp.Om0, cp.w0, cp.wa, cp.h
    except ImportError:  # pragma: no cover - diffsky optional
        Om0, h = cosmology.Om0, cosmology.h
        w0 = getattr(cosmology, "w0", -1.0)
        wa = getattr(cosmology, "wa", 0.0)
    fn = jax.vmap(luminosity_distance_to_z, in_axes=(0, None, None, None, None))
    return np.asarray(fn(np.asarray(z, dtype=np.float64), Om0, w0, wa, h))


def luminosity_distance_cm(z, cosmology, backend: str = "astropy") -> np.ndarray:
    """Luminosity distance in cm for an array of redshifts."""
    z = np.asarray(z, dtype=np.float64)
    if backend == "astropy":
        d_mpc = cosmology.luminosity_distance(z).to_value(u.Mpc)
    elif backend == "dsps":
        d_mpc = _dsps_luminosity_distance_mpc(z, cosmology)
    else:
        raise ValueError(f"unknown distance backend {backend!r} (use 'astropy' or 'dsps')")
    return np.asarray(d_mpc) * MPC_IN_CM


def flux_conversion_factor(z, cosmology, backend: str = "astropy") -> np.ndarray:
    """Factor ``1 / (4 pi D_L^2)`` in cm^-2 turning erg/s into erg/s/cm^2."""
    d_cm = luminosity_distance_cm(z, cosmology, backend=backend)
    return 1.0 / (4.0 * np.pi * d_cm**2)

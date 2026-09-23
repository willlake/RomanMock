"""Emission-line luminosity (erg/s) -> observed flux (erg/s/cm^2).

The diffsky columns are dust-attenuated, continuum-subtracted line luminosities.
The observed flux is ``F = L / (4 pi D_L(z)^2)`` with ``D_L`` from the catalog
cosmology. By default ``D_L`` is evaluated at the cosmological redshift
(``redshift_true``); the peculiar-velocity Doppler correction to the flux is
negligible for the purposes of this mock, but the column is configurable.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from astropy import units as u

from .cosmology import flux_conversion_factor
from .lines import LineSet, get_line_set

FLUX_UNIT = u.erg / u.s / u.cm**2
DEFAULT_FLUX_SUFFIX = "_flux"


def luminosity_to_flux(luminosity, z, cosmology, backend: str = "astropy") -> np.ndarray:
    """Convert luminosity in erg/s to flux in erg/s/cm^2 at redshift ``z``.

    ``luminosity`` may be 1-D ``(n,)`` or 2-D ``(n, n_lines)``; ``z`` is ``(n,)``.
    """
    lum = np.asarray(luminosity, dtype=np.float64)
    factor = flux_conversion_factor(z, cosmology, backend=backend)
    if lum.ndim == 2:
        factor = factor[:, None]
    return lum * factor


def feature_luminosities(data: Mapping[str, np.ndarray], line_set: LineSet) -> np.ndarray:
    """Sum component luminosity columns into an ``(n, n_features)`` array."""
    n = len(next(iter(data.values())))
    out = np.zeros((n, len(line_set)), dtype=np.float64)
    for i, feat in enumerate(line_set):
        for col in feat.components:
            out[:, i] += np.asarray(data[col], dtype=np.float64)
    return out


def feature_fluxes(
    data: Mapping[str, np.ndarray],
    line_set: LineSet,
    z,
    cosmology,
    backend: str = "astropy",
) -> np.ndarray:
    """Observed flux of every feature: ``(n, n_features)`` in erg/s/cm^2."""
    lum = feature_luminosities(data, line_set)
    return luminosity_to_flux(lum, z, cosmology, backend=backend)


def component_fluxes(
    data: Mapping[str, np.ndarray],
    line_set: LineSet,
    z,
    cosmology,
    backend: str = "astropy",
) -> dict[str, np.ndarray]:
    """Observed flux of every component luminosity column, ``{column: (n,) array}``."""
    factor = flux_conversion_factor(z, cosmology, backend=backend)
    return {col: np.asarray(data[col], dtype=np.float64) * factor for col in line_set.required_columns}


def flux_column_names(line_set: LineSet, suffix: str = DEFAULT_FLUX_SUFFIX) -> list[str]:
    return [f"{feat.name}{suffix}" for feat in line_set]


def add_line_fluxes(
    dataset,
    line_set="roman_transitions",
    cosmology=None,
    redshift_column: str = "redshift_true",
    backend: str = "astropy",
    suffix: str = DEFAULT_FLUX_SUFFIX,
):
    """Return an OpenCosmo dataset/lightcone with one flux column per feature.

    The new columns are named ``<feature>{suffix}`` and carry the unit erg/s/cm^2.
    Only the luminosity and redshift columns are read from disk.
    """
    line_set = get_line_set(line_set)
    cosmology = cosmology if cosmology is not None else dataset.cosmology
    needed = line_set.required_columns + [redshift_column]
    data = dataset.select(*needed).get_data("numpy", wrap_single=True)
    flux = feature_fluxes(data, line_set, data[redshift_column], cosmology, backend=backend)
    names = flux_column_names(line_set, suffix)
    new_cols = {name: flux[:, i] * FLUX_UNIT for i, name in enumerate(names)}
    descriptions = {
        name: f"Observed line flux of {feat.label or feat.name} "
        f"(sum of {'+'.join(feat.components)}), dust attenuation included"
        for name, feat in zip(names, line_set)
    }
    return dataset.with_new_columns(descriptions=descriptions, **new_cols)

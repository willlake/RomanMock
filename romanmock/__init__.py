"""romanmock: build a Roman-like emission-line spectroscopic mock from a diffsky lightcone.

The pipeline follows the construction plan in ``Roman_Mock_Plan_and_Comments.pdf``:

1. convert dust-attenuated line luminosities (erg/s) to observed fluxes (erg/s/cm^2)
   and decide which lines are detectable  -> :mod:`romanmock.fluxes`, :mod:`romanmock.detection`
2. record detectable lines in a documented bitmask   -> :mod:`romanmock.lines`, :mod:`romanmock.detection`
3. assign a measured redshift (multi-line: exact; single-line: confusion matrix)
   -> :mod:`romanmock.confusion`, :mod:`romanmock.redshift`
4. construct analysis samples (footprint, redshift, quality, downsampling to a target dN/dz)
   -> :mod:`romanmock.selection`

Catalog access goes through OpenCosmo (``opencosmo``) and, when installed, the diffsky
loaders in ``diffsky.data_loaders.opencosmo_utils``.
"""

from ._version import __version__
from .lines import LineFeature, LineSet, get_line_set, LINE_SET_PRESETS
from .cosmology import luminosity_distance_cm, flux_conversion_factor
from .fluxes import luminosity_to_flux, feature_fluxes, add_line_fluxes
from .sensitivity import (
    ConstantFluxLimit,
    TabulatedFluxLimit,
    ThresholdDetection,
    GaussianNoiseDetection,
)
from .detection import detect_lines, DetectionResult, pack_bitmask, unpack_bitmask, popcount
from .confusion import ConfusionMatrix, WavelengthBinnedConfusion, FAIL
from .redshift import assign_redshifts, IdOutcome, RedshiftAssignment
from .config import RomanMockConfig
from .pipeline import RomanMockPipeline
from .catalog import BaseCatalog, Catalog, make_catalog, make_random
from .footprint import PatchFootprint


__all__ = [
    "BaseCatalog",
    "Catalog",
    "make_catalog",
    "make_random",
    "PatchFootprint",
    "LineFeature",
    "LineSet",
    "get_line_set",
    "LINE_SET_PRESETS",
    "luminosity_distance_cm",
    "flux_conversion_factor",
    "luminosity_to_flux",
    "feature_fluxes",
    "add_line_fluxes",
    "ConstantFluxLimit",
    "TabulatedFluxLimit",
    "ThresholdDetection",
    "GaussianNoiseDetection",
    "detect_lines",
    "DetectionResult",
    "pack_bitmask",
    "unpack_bitmask",
    "popcount",
    "ConfusionMatrix",
    "WavelengthBinnedConfusion",
    "FAIL",
    "assign_redshifts",
    "IdOutcome",
    "RedshiftAssignment",
    "RomanMockConfig",
    "RomanMockPipeline",
]

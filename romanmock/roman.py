"""Roman Wide Field Instrument constants used as defaults.

Only quantities the pipeline actually uses are kept here. Everything is configurable
at run time; these are the baseline values of the plan.
"""

# Roman WFI grism, first order: 1.00-1.93 micron (NASA WFI technical pages).
GRISM_WAVE_RANGE_ANGSTROM = (10_000.0, 19_300.0)

# Roman WFI prism: 0.75-1.80 micron.
PRISM_WAVE_RANGE_ANGSTROM = (7_500.0, 18_000.0)

# Constant line-flux threshold used by the initial model (erg/s/cm^2). This is the
# usual order of magnitude quoted for the Roman High Latitude Spectroscopic Survey
# and is meant to be replaced by a wavelength-dependent sensitivity curve.
DEFAULT_LINE_FLUX_LIMIT = 1.0e-16

# Nominal signal-to-noise ratio that a line exactly at the flux limit is assumed to
# have. Used only to translate a flux limit into a noise level for the optional
# Gaussian-noise detection model.
DEFAULT_SNR_AT_LIMIT = 6.5

"""Roman Wide Field Instrument constants used as defaults.

Only quantities the pipeline actually uses are kept here. Everything is configurable
at run time; these are the baseline values of the plan.
"""

# Roman WFI grism, first order: 1.00-1.93 micron (NASA WFI technical pages).
GRISM_WAVE_RANGE_ANGSTROM = (10_000.0, 19_300.0)

# Roman WFI prism: 0.75-1.80 micron.
PRISM_WAVE_RANGE_ANGSTROM = (7_500.0, 18_000.0)

# Constant line-flux threshold used by the initial model (erg/s/cm^2): the Roman HLSS
# reference depth, 1e-16 at 6.5 sigma (Wang et al. 2022; a tabulated column of their
# dN/dz tables). Meant to be replaced by a wavelength-dependent sensitivity curve.
DEFAULT_LINE_FLUX_LIMIT = 1.0e-16

# Threshold for the *other* lines of a galaxy that already has one line above
# DEFAULT_LINE_FLUX_LIMIT inside the bandpass (erg/s/cm^2). Once a galaxy is anchored by
# a primary line, its redshift is known and fainter lines can be confirmed at lower
# significance (4e-17 is 2.6 sigma for a 6.5 sigma limit of 1e-16).
DEFAULT_SECONDARY_LINE_FLUX_LIMIT = 4.0e-17

# Nominal signal-to-noise ratio that a line exactly at the flux limit is assumed to
# have. Used only to translate a flux limit into a noise level for the optional
# Gaussian-noise detection model.
DEFAULT_SNR_AT_LIMIT = 6.5

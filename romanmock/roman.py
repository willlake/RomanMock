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

# ---------------------------------------------------------------------------------------
# Roman WFI grism sensitivity (NASA WFI technical page, "Grism" section, v1.4 March 2026;
# https://science.nasa.gov/mission/roman-space-telescope/wfi-technical/)
# ---------------------------------------------------------------------------------------
# 5-sigma emission-line flux limits [erg/s/cm^2] for 1 hour of exposure at twice the minimum
# zodiacal background, for a point source (r50 = 0) and for a source with r50 = 0.3 arcsec,
# at ten observed wavelengths [Angstrom]. Typical HLWAS backgrounds are about 30 % lower.
WFI_GRISM_SENSITIVITY_WAVE_ANGSTROM = (10_500.0, 11_000.0, 12_000.0, 13_000.0, 14_000.0,
                                       15_000.0, 16_000.0, 17_000.0, 18_000.0, 19_000.0)
WFI_GRISM_5SIGMA_POINT = (5.9e-17, 4.7e-17, 3.6e-17, 3.2e-17, 3.0e-17, 3.1e-17, 3.3e-17, 3.7e-17, 3.4e-17, 4.7e-17)
WFI_GRISM_5SIGMA_R50_0P3 = (16.1e-17, 12.9e-17, 9.8e-17, 8.7e-17, 7.6e-17, 8.1e-17, 8.4e-17, 8.9e-17, 8.5e-17, 11.7e-17)
WFI_GRISM_REFERENCE_SIZE_ARCSEC = 0.3        # r50 of the extended-source row
WFI_GRISM_REFERENCE_EXPOSURE_S = 3600.0      # the tables are for 1 hour
WFI_GRISM_REFERENCE_BACKGROUND = 2.0         # ... at 2 x the minimum zodiacal background
# The limits scale as t^(-1/2) b^(1/2) with exposure time t and background b (same page).

# S/N thresholds of the S/N route of the catalog: a galaxy needs one line with S/N >= DEFAULT_PRIMARY_SNR;
# its other lines are detected with S/N >= DEFAULT_SECONDARY_SNR.
DEFAULT_PRIMARY_SNR = 5.0
DEFAULT_SECONDARY_SNR = 2.0

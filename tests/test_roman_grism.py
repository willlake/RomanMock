"""The Roman grism sensitivity (S/N route): WFI tables, scalings, sizes, the two-tier S/N rule."""
import numpy as np
import pytest

from romanmock import RomanGrismSensitivity, ConstantFluxLimit, ThresholdDetection, GaussianNoiseDetection, SizeSpec, detect_lines
from romanmock.detection import observed_wavelength, pack_bitmask
from romanmock.lines import ROMAN_TRANSITIONS
from romanmock.roman import (
    WFI_GRISM_5SIGMA_POINT,
    WFI_GRISM_5SIGMA_R50_0P3,
    WFI_GRISM_SENSITIVITY_WAVE_ANGSTROM,
)
from romanmock.sensitivity import build_sensitivity, limits_contained, nominal_flux_limit, nominal_secondary_limit
from romanmock.sizes import arcsec_per_kpc_proper

WAVE = np.array(WFI_GRISM_SENSITIVITY_WAVE_ANGSTROM)


def test_reproduces_the_wfi_tables():
    s = RomanGrismSensitivity()                       # 1 h, 2 x min zodi, S/N 5 / 2, size 0.3"
    np.testing.assert_allclose(s.flux_5sigma(WAVE, 0.0), WFI_GRISM_5SIGMA_POINT, rtol=1e-12)
    np.testing.assert_allclose(s.flux_5sigma(WAVE, 0.3), WFI_GRISM_5SIGMA_R50_0P3, rtol=1e-12)
    np.testing.assert_allclose(s.flux_limit(WAVE), WFI_GRISM_5SIGMA_R50_0P3, rtol=1e-12)        # S/N 5 = the 5 sigma row
    np.testing.assert_allclose(s.secondary_flux_limit(WAVE), 0.4 * np.asarray(WFI_GRISM_5SIGMA_R50_0P3), rtol=1e-12)
    assert 0.11 < s.size_scale.min() and s.size_scale.max() < 0.14          # r_c close to the PSF
    # the default reproduces the baseline 1e-16 of the flux-limit route near the band centre, at 6.5 sigma
    assert abs(RomanGrismSensitivity(snr=6.5).flux_limit(np.array([15000.0]))[0] / 1e-16 - 1.05) < 0.01
    # outside the bandpass: infinite; flat extrapolation to the bandpass edges
    lim = s.flux_limit(np.array([9999.0, 10200.0, 19200.0, 19300.0]))
    assert np.isinf(lim[0]) and np.isinf(lim[3])
    assert lim[1] == pytest.approx(WFI_GRISM_5SIGMA_R50_0P3[0]) and lim[2] == pytest.approx(WFI_GRISM_5SIGMA_R50_0P3[-1])


def test_scalings():
    ref = RomanGrismSensitivity(size=0.0).flux_5sigma(WAVE)
    np.testing.assert_allclose(RomanGrismSensitivity(size=0.0, exposure_time=4 * 3600.0).flux_5sigma(WAVE), ref / 2)
    np.testing.assert_allclose(RomanGrismSensitivity(size=0.0, background=1.4).flux_5sigma(WAVE), ref * np.sqrt(0.7))
    np.testing.assert_allclose(RomanGrismSensitivity(size=0.0, snr=6.5).flux_limit(WAVE), ref * 1.3)
    # the size growth: monotonic, reproduces 0.3" whatever the exponent
    for p in (0.5, 1.0):
        s = RomanGrismSensitivity(size_exponent=p)
        np.testing.assert_allclose(s.flux_5sigma(WAVE, 0.3), WFI_GRISM_5SIGMA_R50_0P3, rtol=1e-12)
        sizes = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
        f = s.flux_5sigma(np.full(sizes.size, 15000.0), sizes)
        assert np.all(np.diff(f) > 0)
    s = RomanGrismSensitivity(size_exponent=0.5)
    r_c = np.interp(15000.0, WAVE, s.size_scale)
    f0 = s.flux_5sigma(np.array([15000.0]), 0.0)[0]
    np.testing.assert_allclose(s.flux_5sigma(np.array([15000.0]), 1.0)[0], f0 * np.sqrt(1 + (1.0 / r_c) ** 2))


def test_snr_and_single_tier():
    s = RomanGrismSensitivity(size=0.0)
    f = s.flux_5sigma(WAVE)
    np.testing.assert_allclose(s.snr_of(2 * f, WAVE), 10.0)
    assert s.snr_of(np.array([1e-16]), np.array([9000.0]))[0] == 0.0
    single = RomanGrismSensitivity(secondary_snr=None)
    assert not single.two_tier and single.to_config()["secondary_snr"] is None
    np.testing.assert_allclose(single.secondary_flux_limit(WAVE), single.flux_limit(WAVE))
    with pytest.raises(ValueError):
        RomanGrismSensitivity(snr=5, secondary_snr=6)
    with pytest.raises(ValueError):
        RomanGrismSensitivity(exposure_time=0)
    assert nominal_flux_limit(s) == pytest.approx(s.flux_limit(np.array([15000.0]))[0])
    assert nominal_secondary_limit(s) == pytest.approx(0.4 * nominal_flux_limit(s))
    assert nominal_secondary_limit(single) is None
    column = RomanGrismSensitivity(size="r50_disk_2d")
    assert nominal_flux_limit(column) == pytest.approx(WFI_GRISM_5SIGMA_R50_0P3[5])    # reference size 0.3" at 1.5 um


def test_size_spec(cosmo):
    assert SizeSpec.parse(0.3).is_fixed and SizeSpec.parse(0.3).required_columns == []
    spec = SizeSpec.parse("r50_disk_2d")
    assert spec.required_columns == ["r50_disk_2d", "redshift_true"] and spec.unit == "kpc"
    assert SizeSpec.parse({"column": "r50", "unit": "arcsec"}).required_columns == ["r50"]
    with pytest.raises(ValueError):
        SizeSpec.parse(-1.0)
    with pytest.raises(KeyError):
        SizeSpec.parse({"column": "r50", "units": "kpc"})
    z = np.array([0.5, 1.5, 2.5])
    data = {"r50_disk_2d": np.array([2.0, np.nan, 4.0]), "redshift_true": z}
    size = spec.angular_size(data, cosmo)
    expected = np.array([2.0, 0.0, 4.0]) * arcsec_per_kpc_proper(z, cosmo)
    np.testing.assert_allclose(size, expected)
    assert 0.2 < size[0] < 0.4                                           # 2 kpc at z = 0.5 is about 0.3"
    floor = SizeSpec.parse({"column": "r50_disk_2d", "scale": 2.0, "floor": 0.1}).angular_size(data, cosmo)
    np.testing.assert_allclose(floor, np.maximum(2 * expected, 0.1))
    # a fixed size ignores the data; a column spec needs them
    assert SizeSpec.parse(0.0).angular_size() == 0.0
    with pytest.raises(ValueError):
        spec.angular_size()
    assert SizeSpec.parse(spec.to_config()) == spec
    assert build_sensitivity(RomanGrismSensitivity(size={"column": "r50_disk_2d", "scale": 0.8}).to_config()).size_spec.scale == 0.8


def test_two_tier_snr_rule_with_sizes(cosmo):
    """A galaxy with one line of S/N >= 5 and no other above 2 is single-line; with other lines of
    S/N >= 2 it is multi-line; S/N >= 2 lines alone give nothing. Sizes make the thresholds per galaxy."""
    s = RomanGrismSensitivity(snr=5.0, secondary_snr=2.0, size="r50_disk_2d")
    n = 6
    z = np.full(n, 1.5)                                                   # Ha, Hb, [O III] in the band
    wave = observed_wavelength(ROMAN_TRANSITIONS.rest_waves, z)
    size = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    f5 = s.flux_5sigma(wave, size[:, None])
    flux = np.zeros((n, len(ROMAN_TRANSITIONS)))
    ha, oiii = ROMAN_TRANSITIONS.bit("Ha"), ROMAN_TRANSITIONS.bit("OIII_5007")
    flux[0, ha] = 1.2 * f5[0, ha]                                         # S/N 6: single line
    flux[1, ha], flux[1, oiii] = 1.2 * f5[1, ha], 0.5 * f5[1, oiii]       # S/N 6 + 2.5: multi-line
    flux[2, ha], flux[2, oiii] = 0.9 * f5[2, ha], 0.9 * f5[2, oiii]       # S/N 4.5 + 4.5: nothing
    flux[3, ha], flux[3, oiii] = 1.2 * f5[3, ha], 0.3 * f5[3, oiii]       # S/N 6 + 1.5: single line
    flux[4] = flux[1]                                                     # same fluxes, a 1" galaxy: fainter than its noise
    flux[5] = np.where(np.isfinite(f5[1]), flux[1] * f5[5] / f5[1], 0.0)   # scaled to its own noise: multi-line again
    det = detect_lines(flux, z, ROMAN_TRANSITIONS, ThresholdDetection(s), size=size)
    np.testing.assert_array_equal(det.n_det, [1, 2, 0, 1, 0, 2])
    np.testing.assert_allclose(det.snr[1, [ha, oiii]], [6.0, 2.5])
    np.testing.assert_allclose(det.snr[4, [ha, oiii]], 5 * flux[1, [ha, oiii]] / f5[4, [ha, oiii]])
    # the same with a hand-made per-galaxy constant limit
    expect = np.zeros_like(det.detected)
    for i in range(n):
        d = detect_lines(flux[i:i + 1], z[i:i + 1], ROMAN_TRANSITIONS,
                         ThresholdDetection(ConstantFluxLimit(f5[i, ha], secondary_flux_limit=0.4 * f5[i, ha])))
        expect[i] = d.detected[0]
    np.testing.assert_array_equal(pack_bitmask(expect), det.bitmask)
    # without sizes a column model refuses; a fixed-size model does not need them
    with pytest.raises(ValueError, match="size"):
        detect_lines(flux, z, ROMAN_TRANSITIONS, ThresholdDetection(s))
    fixed = detect_lines(flux, z, ROMAN_TRANSITIONS, ThresholdDetection(RomanGrismSensitivity(size=0.0)))
    np.testing.assert_array_equal(fixed.n_det[:4], [1, 2, 0, 1])
    with pytest.raises(ValueError, match="one value per galaxy"):
        detect_lines(flux, z, ROMAN_TRANSITIONS, ThresholdDetection(s), size=size[:3])


def test_gaussian_noise_uses_the_line_noise():
    rng = np.random.default_rng(5)
    n = 40000
    z = np.full(n, 1.5)
    s = RomanGrismSensitivity(size=0.3)
    wave = observed_wavelength(ROMAN_TRANSITIONS.rest_waves, z)
    flux = np.zeros((n, len(ROMAN_TRANSITIONS)))
    flux[:, 0] = s.flux_limit(wave[:, 0])                                 # exactly S/N 5: detected half of the time
    model = GaussianNoiseDetection(s)
    assert model.snr_at_limit == 5.0 and model.snr_threshold == 5.0
    det = detect_lines(flux, z, ROMAN_TRANSITIONS, model, rng)
    assert 0.48 < det.detected[:, 0].mean() < 0.52
    # the realised S/N scatters by one around 5
    assert abs(det.snr[:, 0].std() - 1.0) < 0.03


def test_config_roundtrip_and_containment():
    s = RomanGrismSensitivity(snr=6.5, secondary_snr=2.6, size="r50_disk_2d", exposure_time=7200.0, background=1.4)
    back = build_sensitivity(s.to_config())
    assert back.to_config() == s.to_config()
    np.testing.assert_allclose(back.flux_limit(WAVE, 0.5), s.flux_limit(WAVE, 0.5))
    assert "S/N >= 6.5" in s.describe() and "r50_disk_2d" in s.describe()
    custom = RomanGrismSensitivity(wave=[10000.0, 20000.0], flux_5sigma_point=[1e-17, 2e-17], flux_5sigma_extended=[2e-17, 4e-17])
    assert build_sensitivity(custom.to_config()).to_config() == custom.to_config()
    # containment against a base catalog
    assert limits_contained(RomanGrismSensitivity(), ConstantFluxLimit(1e-17, secondary_flux_limit=None))[0] is None
    assert limits_contained(RomanGrismSensitivity(size="r50_disk_2d"), RomanGrismSensitivity(size=0.0))[0] is None
    assert limits_contained(RomanGrismSensitivity(size=0.3, snr=4.0), RomanGrismSensitivity(size=0.3))[0] is not None
    msg, ratio = limits_contained(RomanGrismSensitivity(size="r50_disk_2d"), ConstantFluxLimit(1e-16))
    assert "deeper" in msg and "size" in msg and ratio < 1
    msg, _ = limits_contained(RomanGrismSensitivity(wave_range=(9000.0, 19300.0)), RomanGrismSensitivity())
    assert "bandpass" in msg
    # a fixed-size base keeps its size: the point-source base contains the 0.3" request, not the reverse
    assert limits_contained(ConstantFluxLimit(1e-16), RomanGrismSensitivity(size=0.0))[0] is None
    assert limits_contained(ConstantFluxLimit(1e-16), RomanGrismSensitivity(size=0.3))[0] is not None


def test_pipeline_arrays_with_size_column(synthetic, cosmo):
    from romanmock import RomanMockConfig, RomanMockPipeline

    rng = np.random.default_rng(2)
    data = dict(synthetic)
    data["r50_disk_2d"] = 10 ** rng.normal(0.5, 0.25, len(data["redshift_true"]))
    cfg = RomanMockConfig.from_dict({"catalog": {"rest_waves_from_catalog": False}, "output": {"store_snr": True},
                                     "sensitivity": {"type": "roman_grism", "size": "r50_disk_2d"}})
    pipe = RomanMockPipeline(cfg)
    assert "r50_disk_2d" in pipe.input_columns and pipe.base_columns == ["gal_id", "ra", "dec", "redshift_true", "r50_disk_2d"]
    out = pipe.process_arrays(data, cosmo, np.random.default_rng(0))
    size = pipe.angular_sizes(data, cosmo)
    wave = observed_wavelength(ROMAN_TRANSITIONS.rest_waves, data["redshift_obs"])
    flux = pipe.compute_fluxes(data, cosmo)[0]
    snr = pipe.sensitivity.snr_of(flux, wave, size[:, None])
    primary = snr > 5.0
    expect = primary | (primary.any(axis=1)[:, None] & (snr > 2.0))
    np.testing.assert_array_equal(out["line_bitmask"], pack_bitmask(expect))
    np.testing.assert_allclose(out["Ha_snr"], snr[:, 0])
    assert pipe.scientific_config()["sensitivity"]["size"] == {"column": "r50_disk_2d", "unit": "kpc", "redshift_column": "redshift_true"}
    # a fixed size needs no column and detects more (the galaxies are mostly larger than 0.3")
    cfg.sensitivity = {"type": "roman_grism", "size": 0.3}
    pipe2 = RomanMockPipeline(cfg)
    assert "r50_disk_2d" not in pipe2.input_columns
    out2 = pipe2.process_arrays(data, cosmo, np.random.default_rng(0))
    assert (out2["line_bitmask"] > 0).sum() > (out["line_bitmask"] > 0).sum() > 0

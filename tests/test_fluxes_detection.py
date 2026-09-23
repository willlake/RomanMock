import numpy as np
import pytest
from astropy import units as u

from romanmock import (
    ConstantFluxLimit,
    GaussianNoiseDetection,
    TabulatedFluxLimit,
    ThresholdDetection,
    detect_lines,
    feature_fluxes,
    luminosity_to_flux,
    pack_bitmask,
    popcount,
    unpack_bitmask,
)
from romanmock.cosmology import luminosity_distance_cm
from romanmock.detection import observed_wavelength
from romanmock.lines import ROMAN_TRANSITIONS


def test_luminosity_to_flux_matches_astropy(cosmo):
    z = np.array([0.5, 1.0, 2.0])
    L = np.full(3, 1e42)
    dl = cosmo.luminosity_distance(z).to_value(u.cm)
    expected = L / (4 * np.pi * dl**2)
    np.testing.assert_allclose(luminosity_to_flux(L, z, cosmo), expected, rtol=1e-12)
    np.testing.assert_allclose(luminosity_distance_cm(z, cosmo), dl)


@pytest.mark.skipif(pytest.importorskip("dsps", reason="dsps not installed") is None, reason="dsps")
def test_dsps_backend_agrees_with_astropy(cosmo):
    z = np.linspace(0.1, 3.9, 20)
    a = luminosity_distance_cm(z, cosmo, backend="astropy")
    b = luminosity_distance_cm(z, cosmo, backend="dsps")
    np.testing.assert_allclose(a, b, rtol=1e-4)


def test_feature_fluxes_sums_components(synthetic, cosmo):
    z = synthetic["redshift_true"]
    flux = feature_fluxes(synthetic, ROMAN_TRANSITIONS, z, cosmo)
    i = ROMAN_TRANSITIONS.bit("OII_3727")
    expected = luminosity_to_flux(synthetic["O_II_3726"] + synthetic["O_II_3729"], z, cosmo)
    np.testing.assert_allclose(flux[:, i], expected)
    assert flux.shape == (len(z), len(ROMAN_TRANSITIONS))


def test_bitmask_roundtrip_and_popcount():
    rng = np.random.default_rng(0)
    det = rng.random((1000, 9)) < 0.3
    bm = pack_bitmask(det)
    assert bm.dtype == np.uint32
    np.testing.assert_array_equal(unpack_bitmask(bm, 9), det)
    np.testing.assert_array_equal(popcount(bm), det.sum(axis=1))
    # documented example from the plan: BM=3 has two bits, BM=4 one bit
    np.testing.assert_array_equal(popcount(np.array([3, 4], dtype=np.uint32)), [2, 1])


def test_constant_flux_limit_bandpass():
    s = ConstantFluxLimit(1e-16, (10000.0, 19300.0))
    lim = s.flux_limit(np.array([9000.0, 12000.0, 20000.0]))
    assert np.isinf(lim[0]) and lim[1] == 1e-16 and np.isinf(lim[2])


def test_tabulated_flux_limit_interpolates():
    s = TabulatedFluxLimit([10000.0, 20000.0], [1e-16, 1e-15])
    mid = s.flux_limit(np.array([15000.0]))[0]
    assert 1e-16 < mid < 1e-15
    assert np.isinf(s.flux_limit(np.array([9999.0]))[0])


def test_threshold_detection_rule(cosmo):
    z = np.array([1.5, 1.5, 0.2])  # Ha at 16412 A (in), Ha (in, faint), Ha at 7878 A (out)
    flux = np.zeros((3, len(ROMAN_TRANSITIONS)))
    flux[:, 0] = [5e-16, 5e-17, 5e-15]
    det = detect_lines(flux, z, ROMAN_TRANSITIONS, ThresholdDetection(ConstantFluxLimit(1e-16)))
    np.testing.assert_array_equal(det.bitmask, [1, 0, 0])
    np.testing.assert_array_equal(det.n_det, [1, 0, 0])
    wave = observed_wavelength(ROMAN_TRANSITIONS.rest_waves, z)
    assert abs(wave[0, 0] - ROMAN_TRANSITIONS["Ha"].rest_wave * 2.5) < 1e-9


def test_gaussian_noise_detection_scatters_near_limit():
    rng = np.random.default_rng(3)
    n = 20000
    z = np.full(n, 1.5)
    flux = np.zeros((n, len(ROMAN_TRANSITIONS)))
    flux[:, 0] = 1e-16  # exactly at the limit -> detected ~50% of the time
    model = GaussianNoiseDetection(ConstantFluxLimit(1e-16), snr_at_limit=6.5, snr_threshold=6.5)
    det = detect_lines(flux, z, ROMAN_TRANSITIONS, model, rng)
    frac = det.detected[:, 0].mean()
    assert 0.47 < frac < 0.53
    # lines outside the bandpass are never detected, even with noise
    assert not det.detected[:, ROMAN_TRANSITIONS.bit("SIII_9532")].any()

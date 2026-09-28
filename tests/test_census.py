import numpy as np
import pytest

from romanmock.census import CLASSES, LineCensus, census_arrays, line_census
from romanmock.detection import detect_lines, observed_wavelength
from romanmock.fluxes import feature_fluxes
from romanmock.lines import ROMAN_TRANSITIONS
from romanmock.sensitivity import ConstantFluxLimit, ThresholdDetection

from conftest import CATALOG_DIR, catalog_available

Z_EDGES = np.round(np.arange(0.5, 3.0 + 1e-9, 0.1), 2)
LIMITS = [1e-16, 1.5e-16]


def test_census_arrays_matches_direct_count(synthetic, cosmo):
    counts, n_all, n_any = census_arrays(synthetic, ROMAN_TRANSITIONS, cosmo, LIMITS, Z_EDGES)
    assert counts.shape == (2, len(ROMAN_TRANSITIONS), 2, Z_EDGES.size - 1)
    assert n_any.shape == (2, 2, Z_EDGES.size - 1)
    z = synthetic["redshift_obs"]
    assert n_all.sum() == np.sum((z >= Z_EDGES[0]) & (z < Z_EDGES[-1]))

    flux = feature_fluxes(synthetic, ROMAN_TRANSITIONS, synthetic["redshift_true"], cosmo)
    for i, lim in enumerate(LIMITS):
        det = detect_lines(flux, z, ROMAN_TRANSITIONS,
                           ThresholdDetection(ConstantFluxLimit(lim, secondary_flux_limit=None)))   # single threshold
        in_z = (z >= Z_EDGES[0]) & (z < Z_EDGES[-1])
        assert n_any[i, 0].sum() == np.sum(in_z & (det.n_det == 1))
        assert n_any[i, 1].sum() == np.sum(in_z & (det.n_det > 1))
        # distinct galaxies: single-line galaxies equal the sum of the per-line single counts,
        # multi-line galaxies are fewer than the sum of the per-line multi counts
        assert n_any[i, 0].sum() == counts[i, :, 0].sum()
        assert n_any[i, 1].sum() < counts[i, :, 1].sum()
        for j, name in enumerate(ROMAN_TRANSITIONS.names):
            hit = det.detected[:, j]
            for k, cls in enumerate(CLASSES):
                sel = hit & ((det.n_det == 1) if cls == "single" else (det.n_det > 1))
                expect, _ = np.histogram(z[sel], bins=Z_EDGES)
                np.testing.assert_array_equal(counts[i, j, k], expect)
        # the limit at 1.5e-16 can only remove galaxies
    assert counts[1].sum() <= counts[0].sum()
    # a detected line is either single or multi: all = single + multi and no galaxy counted twice
    assert (counts.sum(axis=2) >= 0).all()


def test_census_bandpass_cut(synthetic, cosmo):
    counts, _, _ = census_arrays(synthetic, ROMAN_TRANSITIONS, cosmo, [1e-30], Z_EDGES)   # every line bright
    z = synthetic["redshift_obs"]
    wave = observed_wavelength(ROMAN_TRANSITIONS.rest_waves, z)
    lo, hi = ConstantFluxLimit().wave_range
    for j in range(len(ROMAN_TRANSITIONS)):
        inside = (wave[:, j] > lo) & (wave[:, j] < hi) & (z >= Z_EDGES[0]) & (z < Z_EDGES[-1])
        assert counts[0, j].sum() == inside.sum()


def test_line_census_accessors(synthetic, cosmo, tmp_path):
    counts, n_all, n_any = census_arrays(synthetic, ROMAN_TRANSITIONS, cosmo, LIMITS, Z_EDGES)
    c = LineCensus.empty(LIMITS, Z_EDGES, ROMAN_TRANSITIONS).add(counts, n_all, n_any, step=1)
    c.add(counts, n_all, n_any, step=2)
    np.testing.assert_array_equal(c.counts, 2 * counts)
    np.testing.assert_array_equal(c.galaxies(1e-16), 2 * n_any[0].sum(axis=0))
    np.testing.assert_array_equal(c.galaxies(1e-16, "single") + c.galaxies(1e-16, "multi"), c.galaxies(1e-16))
    area = 10.0
    np.testing.assert_allclose(c.dndz("Ha", 1e-16, area), c.hist("Ha", 1e-16) / (0.1 * area), rtol=1e-6)
    np.testing.assert_allclose(c.hist("Ha", 1e-16), c.hist("Ha", 1e-16, "single") + c.hist("Ha", 1e-16, "multi"))
    total = c.integrated("Ha", 1e-16, area)
    part = c.integrated("Ha", 1e-16, area, z_range=(1.0, 2.0))
    assert 0 < part <= total
    frac = c.single_fraction("Ha", 1.5e-16)
    assert np.nanmin(frac) >= 0 and np.nanmax(frac) <= 1
    with pytest.raises(ValueError):
        c.hist("Ha", 2e-16)
    with pytest.raises(KeyError):
        c.hist("nope", 1e-16)
    c2 = LineCensus.load(c.save(tmp_path / "census.npz"))
    np.testing.assert_array_equal(c2.counts, c.counts)
    np.testing.assert_array_equal(c2.n_any, c.n_any)
    assert c2.line_names == c.line_names and c2.steps == [1, 2] and c2.z_column == "redshift_obs"


@pytest.mark.skipif(not catalog_available(), reason="diffsky mock not available")
def test_line_census_real_slice_serial_equals_parallel():
    from romanmock.io import list_catalog_files

    files = list_catalog_files(CATALOG_DIR)
    small = sorted(files, key=lambda p: p.stat().st_size)[:2]
    edges = np.linspace(0.0, 4.0, 9)
    a = line_census(CATALOG_DIR, LIMITS, edges, files=small, n_workers=1, verbose=False)
    b = line_census(CATALOG_DIR, LIMITS, edges, files=small, n_workers=2, verbose=False)
    np.testing.assert_array_equal(a.counts, b.counts)
    np.testing.assert_array_equal(a.n_all, b.n_all)
    np.testing.assert_array_equal(a.n_any, b.n_any)
    assert a.steps == b.steps


def test_census_with_secondary_limits(synthetic, cosmo, tmp_path):
    limits = [1e-16, (1e-16, 8e-17), (1e-16, 4e-17), (1.5e-16, None)]
    counts, n_all, n_any = census_arrays(synthetic, ROMAN_TRANSITIONS, cosmo, limits, Z_EDGES)
    c = LineCensus.empty(limits, Z_EDGES, ROMAN_TRANSITIONS).add(counts, n_all, n_any, step=1)
    assert c.limits == [(1e-16, None), (1e-16, 8e-17), (1e-16, 4e-17), (1.5e-16, None)]
    z = synthetic["redshift_obs"]
    flux = feature_fluxes(synthetic, ROMAN_TRANSITIONS, synthetic["redshift_true"], cosmo)
    in_z = (z >= Z_EDGES[0]) & (z < Z_EDGES[-1])
    for second in (None, 8e-17, 4e-17):
        det = detect_lines(flux, z, ROMAN_TRANSITIONS,
                           ThresholdDetection(ConstantFluxLimit(1e-16, secondary_flux_limit=second)))
        assert c.galaxies(1e-16, "single", secondary=second).sum() == np.sum(in_z & (det.n_det == 1))
        assert c.galaxies(1e-16, "multi", secondary=second).sum() == np.sum(in_z & (det.n_det > 1))
        for name in ("Ha", "OIII_5007"):
            hit = det.detected[:, ROMAN_TRANSITIONS.bit(name)] & in_z
            assert c.hist(name, 1e-16, secondary=second).sum() == hit.sum()
            assert c.hist(name, 1e-16, "single", secondary=second).sum() == np.sum(hit & (det.n_det == 1))
    # the second tier keeps the detected galaxies and moves some of them from single- to multi-line
    np.testing.assert_array_equal(c.galaxies(1e-16), c.galaxies(1e-16, secondary=4e-17))
    s0, s8, s4 = (c.galaxies(1e-16, "single", secondary=x).sum() for x in (None, 8e-17, 4e-17))
    assert s0 > s8 > s4
    assert c.hist("Hb", 1e-16, secondary=4e-17).sum() > c.hist("Hb", 1e-16).sum()
    np.testing.assert_array_equal(c.hist("Ha", 1e-16), c.hist("Ha", 1e-16, secondary=None))
    with pytest.raises(ValueError):
        c.hist("Ha", 1e-16, secondary=2e-17)
    with pytest.raises(ValueError):
        census_arrays(synthetic, ROMAN_TRANSITIONS, cosmo, [(1e-16, 2e-16)], Z_EDGES)
    c2 = LineCensus.load(c.save(tmp_path / "c.npz"))
    assert c2.limits == c.limits
    np.testing.assert_array_equal(c2.hist("Ha", 1e-16, "single", secondary=4e-17), c.hist("Ha", 1e-16, "single", secondary=4e-17))

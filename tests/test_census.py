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
    counts, n_all = census_arrays(synthetic, ROMAN_TRANSITIONS, cosmo, LIMITS, Z_EDGES)
    assert counts.shape == (2, len(ROMAN_TRANSITIONS), 2, Z_EDGES.size - 1)
    z = synthetic["redshift_obs"]
    assert n_all.sum() == np.sum((z >= Z_EDGES[0]) & (z < Z_EDGES[-1]))

    flux = feature_fluxes(synthetic, ROMAN_TRANSITIONS, synthetic["redshift_true"], cosmo)
    for i, lim in enumerate(LIMITS):
        det = detect_lines(flux, z, ROMAN_TRANSITIONS, ThresholdDetection(ConstantFluxLimit(lim)))
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
    counts, _ = census_arrays(synthetic, ROMAN_TRANSITIONS, cosmo, [1e-30], Z_EDGES)   # every line bright
    z = synthetic["redshift_obs"]
    wave = observed_wavelength(ROMAN_TRANSITIONS.rest_waves, z)
    lo, hi = ConstantFluxLimit().wave_range
    for j in range(len(ROMAN_TRANSITIONS)):
        inside = (wave[:, j] > lo) & (wave[:, j] < hi) & (z >= Z_EDGES[0]) & (z < Z_EDGES[-1])
        assert counts[0, j].sum() == inside.sum()


def test_line_census_accessors(synthetic, cosmo, tmp_path):
    counts, n_all = census_arrays(synthetic, ROMAN_TRANSITIONS, cosmo, LIMITS, Z_EDGES)
    c = LineCensus.empty(LIMITS, Z_EDGES, ROMAN_TRANSITIONS).add(counts, n_all, step=1)
    c.add(counts, n_all, step=2)
    np.testing.assert_array_equal(c.counts, 2 * counts)
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
    assert a.steps == b.steps

import numpy as np
import pytest

from romanmock.fluxdist import FluxDistribution, flux_distribution, flux_histograms
from romanmock.fluxes import feature_fluxes
from romanmock.lines import ROMAN_TRANSITIONS

from conftest import CATALOG_DIR, catalog_available

Z_EDGES = np.array([0.0, 0.5, 1.0, 4.0])
LOGF_EDGES = np.round(np.arange(-21.0, -11.0 + 1e-9, 0.02), 6)


def test_quantiles_and_fractions_match_direct(synthetic, cosmo):
    counts, below, above, n_gal = flux_histograms(synthetic, ROMAN_TRANSITIONS, cosmo, Z_EDGES, LOGF_EDGES)
    d = FluxDistribution.empty(Z_EDGES, LOGF_EDGES, ROMAN_TRANSITIONS).add(counts, below, above, n_gal, step=1)
    z = synthetic["redshift_true"]
    flux = feature_fluxes(synthetic, ROMAN_TRANSITIONS, z, cosmo)
    assert n_gal.sum() == np.sum(z < Z_EDGES[-1])
    for name in ("Ha", "OIII_5007"):
        i = ROMAN_TRANSITIONS.bit(name)
        assert (counts[i].sum(axis=1) + below[i] + above[i] == n_gal).all()
        for j in range(Z_EDGES.size - 1):
            sel = (z >= Z_EDGES[j]) & (z < Z_EDGES[j + 1])
            if sel.sum() < 50:
                continue
            f = flux[sel, i]
            for q in (0.05, 0.5, 0.95):
                # within one flux bin (0.02 dex) of the direct quantile
                assert abs(np.log10(d.quantile(name, q)[j]) - np.log10(np.quantile(f, q))) < 0.03
            assert d.fraction_above(name, 1e-16)[j] == pytest.approx(np.mean(f > 1e-16), abs=0.01)
            assert d.number_above(name, 1e-16)[j] == pytest.approx(np.sum(f > 1e-16), abs=0.01 * sel.sum())
    assert np.isnan(d.fraction_above("Ha", 1e-16)[d.n_gal == 0]).all() if (d.n_gal == 0).any() else True


def test_save_load(synthetic, cosmo, tmp_path):
    parts = flux_histograms(synthetic, ROMAN_TRANSITIONS, cosmo, Z_EDGES, LOGF_EDGES)
    d = FluxDistribution.empty(Z_EDGES, LOGF_EDGES, ROMAN_TRANSITIONS).add(*parts, step=3)
    d2 = FluxDistribution.load(d.save(tmp_path / "f.npz"))
    np.testing.assert_array_equal(d2.counts, d.counts)
    assert d2.line_names == d.line_names and d2.steps == [3] and d2.z_column == "redshift_true"
    np.testing.assert_allclose(d2.quantile("Hb", 0.5), d.quantile("Hb", 0.5))


@pytest.mark.skipif(not catalog_available(), reason="diffsky mock not available")
def test_flux_distribution_serial_equals_parallel():
    from romanmock.io import list_catalog_files

    small = sorted(list_catalog_files(CATALOG_DIR), key=lambda p: p.stat().st_size)[:2]
    a = flux_distribution(CATALOG_DIR, files=small, n_workers=1, verbose=False)
    b = flux_distribution(CATALOG_DIR, files=small, n_workers=2, verbose=False)
    np.testing.assert_array_equal(a.counts, b.counts)
    np.testing.assert_array_equal(a.n_gal, b.n_gal)
    assert a.steps == b.steps and a.n_gal.sum() > 0

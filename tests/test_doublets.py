"""Tests of romanmock.doublets (array level)."""
import numpy as np
import pytest

from romanmock.doublets import DoubletRule, arcsec_per_kpc_interpolated, build_doublets, doublet_dip
from romanmock.lines import get_line_set
from romanmock.sizes import SizeSpec

LS = get_line_set("roman_transitions_paa")
BAND = (10000.0, 19300.0)
S, W = LS.names.index("OIII_5007"), LS.names.index("OIII_4959")


def test_resolved_radius_and_dip():
    rule = DoubletRule()
    r = rule.resolved_radius([1.0, 1.94, 2.85], LS)
    np.testing.assert_allclose(r, [0.354, 0.551, 0.738], atol=0.005)       # Projects/Roman/...OIII_doublet.md, Table 5
    assert np.all(np.diff(rule.resolved_radius(np.linspace(1, 2.9, 20), LS)) > 0)   # wider separation, larger galaxies
    sep = rule.rest_separation(LS) * 2.2
    dips = [doublet_dip(x, sep) for x in (0.1, 0.2, 0.3, 0.4, 0.5)]          # falls through the threshold
    assert all(a > b for a, b in zip(dips, dips[1:])) and dips[0] > 0.9 and dips[-1] < 0.05
    assert doublet_dip(0.9, sep) < 0.01         # blended: what is left is the cusp of the noiseless profile
    looser = DoubletRule(min_dip=0.1).resolved_radius(1.94, LS)
    assert looser > r[1] and DoubletRule(min_dip=0.3).resolved_radius(1.94, LS) < r[1]


def _fluxes(n):
    flux = np.full((n, len(LS)), 1e-17)
    flux[:, S], flux[:, W] = 3e-16, 1e-16
    return flux


def test_apply_merges_only_blended_doublets_with_both_lines_in_band():
    z = np.array([1.5, 1.5, 1.005, 2.87, 0.5, 2.0])          # both in band, both, only 5007, only 4959, neither, both
    r50 = np.array([0.2, 1.0, 1.0, 1.0, 1.0, 1.0])            # arcsec: small (resolved at 1.5) or large (blended)
    flux = _fluxes(z.size)
    rule = DoubletRule(size=0.3)                              # a fixed size: per_galaxy needs no column
    out, merged = DoubletRule(mode="per_galaxy", size={"column": "r", "unit": "arcsec"}).apply(flux, z, LS, BAND, {"r": r50})
    np.testing.assert_array_equal(merged, [False, True, False, False, False, True])
    np.testing.assert_allclose(out[merged, S], 4e-16)
    np.testing.assert_array_equal(out[merged, W], 0.0)
    np.testing.assert_array_equal(out[~merged], flux[~merged])
    assert flux[1, W] == 1e-16                                 # the input is not modified
    np.testing.assert_allclose(out.sum(axis=1), flux.sum(axis=1))   # flux is moved, not created
    # a fixed 0.3": resolved at z 1.5 and 2 (largest resolved 0.46" and 0.56")
    assert not rule.apply(flux, z, LS, BAND)[1].any()
    merged_all = build_doublets("merged").apply(flux, z, LS, BAND)[1]
    np.testing.assert_array_equal(merged_all, [True, True, False, False, False, True])
    out_sep, m_sep = build_doublets("separated").apply(flux, z, LS, BAND)
    assert not m_sep.any() and out_sep is flux


def test_sizes_in_kpc_match_sizespec_and_invalid_sizes_count_as_blended(cosmo):
    rng = np.random.default_rng(3)
    data = {"r50_disk_2d": rng.uniform(1, 8, 2000), "redshift_true": rng.uniform(1.0, 2.9, 2000)}
    rule = DoubletRule()
    assert rule.required_columns == ["r50_disk_2d", "redshift_true"]
    np.testing.assert_allclose(rule.angular_size(data, cosmo), SizeSpec.parse("r50_disk_2d").angular_size(data, cosmo), rtol=1e-5)
    data["r50_disk_2d"][:3] = [np.nan, -1.0, np.inf]
    size = rule.angular_size(data, cosmo)
    assert np.isnan(size[:3]).all()
    assert not rule.resolved(data["redshift_true"][:3], size[:3], LS).any()
    z = np.linspace(0.06, 5, 1000)
    from romanmock.sizes import arcsec_per_kpc_proper
    np.testing.assert_allclose(arcsec_per_kpc_interpolated(z, cosmo), arcsec_per_kpc_proper(z, cosmo), rtol=5e-6)


def test_config_round_trip_and_errors():
    for cfg in ("per_galaxy", "merged", "separated", {"mode": "per_galaxy", "min_dip": 0.3, "size": 0.25}):
        rule = build_doublets(cfg)
        assert build_doublets(rule.to_config()).to_config() == rule.to_config()
    assert build_doublets(None) is None and "added to OIII_5007" in DoubletRule().describe()
    with pytest.raises(ValueError, match="mode"):
        DoubletRule(mode="sometimes")
    with pytest.raises(KeyError, match="OIII_4959"):
        DoubletRule().check(get_line_set("plan_species"))
    with pytest.raises(KeyError, match="unknown doublet rule"):
        build_doublets({"type": "triplet"})

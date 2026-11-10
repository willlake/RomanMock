"""Tests of romanmock.rescaling (array level)."""
import numpy as np
import pytest
from scipy.special import ndtri

from romanmock.lines import get_line_set
from romanmock.rescaling import (DEFAULT_SEED, GalacticusLike, LineFactors, build_rescaling, calzetti_k,
                                 galaxy_normal)

LS = get_line_set("roman_transitions_paa")


def _splitmix_normal(i, seed):
    """Plain-Python splitmix64 of galaxy_normal, to check the vectorised uint64 arithmetic."""
    m = (1 << 64) - 1
    x = (i + seed * 0x9E3779B97F4A7C15 + 0x9E3779B97F4A7C15) & m
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & m
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & m
    x ^= x >> 31
    return float(ndtri(((x >> 11) + 0.5) * 2.0**-53))


def test_galaxy_normal_matches_reference_and_is_normal():
    ids = np.array([0, 1, 252887674393380, 264982310683042, 2**62 + 12345], dtype=np.int64)
    got = galaxy_normal(ids, seed=7)
    want = [_splitmix_normal(int(i), 7) for i in ids]
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    eps = galaxy_normal(np.arange(10**6) + 252887674393380)
    assert np.isfinite(eps).all()
    assert abs(eps.mean()) < 0.005 and abs(eps.std() - 1) < 0.005
    assert abs(np.corrcoef(eps[:-1], eps[1:])[0, 1]) < 0.005                # neighbouring ids are independent
    np.testing.assert_array_equal(galaxy_normal(ids), galaxy_normal(ids.copy()))
    assert not np.allclose(galaxy_normal(ids, seed=1), galaxy_normal(ids, seed=2))


def test_calzetti_values():
    np.testing.assert_allclose(calzetti_k([6564.6, 4862.7, 18756.0]), [3.326, 4.598, 0.587], atol=2e-3)


def test_galacticus_like_offsets_roman_transitions_paa():
    constant = GalacticusLike(galaxy_dust=())                              # the first version: constant dust for every line
    off = dict(zip(LS.names, constant.offsets(LS)))
    k = dict(zip(LS.names, calzetti_k(LS.rest_waves)))
    for n in ["Hb", "OIII_5007", "OIII_4959", "OII_3727", "SIII_9532", "SIII_9069", "PaA"]:
        assert off[n] == pytest.approx(-0.2 * (k[n] / k["Ha"] - 1) + GalacticusLike.DEFAULT_DELTA.get(n, 0.0), abs=1e-12)
    assert off["Ha"] == 0.0
    assert off["OIII_4363"] == 0.0 and off["SIII_6312"] == 0.0           # no counterpart in Galacticus: left alone
    assert constant.rescaled_features(LS) == ["Ha", "Hb", "OIII_5007", "OIII_4959", "OII_3727", "SIII_9532", "SIII_9069", "PaA"]
    want = {"Ha": 1.0, "Hb": 0.919, "OIII_5007": 1.323, "OIII_4959": 1.315, "OII_3727": 1.651, "SIII_9532": 1.199,
            "SIII_9069": 1.175, "OIII_4363": 1.0, "SIII_6312": 1.0, "PaA": 1.287}
    for n, v in off.items():
        assert 10 ** v == pytest.approx(want[n], abs=6e-4)
    # the default: [S III] and Paα take their dust per galaxy; the constant part is delta alone
    model = GalacticusLike()
    assert model.galaxy_features(LS) == ["SIII_9532", "SIII_9069", "PaA"] and model.required_features(LS) == ["Ha", "PaA"]
    off = dict(zip(LS.names, model.offsets(LS)))
    assert off["SIII_9532"] == 0.0 and off["SIII_9069"] == 0.0 and off["PaA"] == pytest.approx(-0.055)
    typical = dict(zip(LS.names, 10 ** model.typical_offsets(LS)))       # at diffsky's median dust, 0.35 mag
    for n, v in {"SIII_9532": 1.191, "SIII_9069": 1.168, "PaA": 1.309, "OIII_5007": 1.323, "Hb": 0.919}.items():
        assert typical[n] == pytest.approx(v, abs=6e-4)
    sig = dict(zip(LS.names, model.sigmas(LS)))
    assert sig == {"Ha": 0, "Hb": 0, "OIII_5007": 0.19, "OIII_4959": 0.19, "OII_3727": 0.11, "SIII_9532": 0, "SIII_9069": 0,
                   "OIII_4363": 0, "SIII_6312": 0, "PaA": 0}


def test_apply_keeps_ha_zeros_doublets_and_shares_the_deviate():
    rng = np.random.default_rng(3)
    n = 5000
    flux = 10 ** rng.normal(-16, 0.5, (n, len(LS)))
    flux[0, LS.names.index("OII_3727")] = 0.0
    flux[1, LS.names.index("Hb")] = -1e-18
    flux[2, LS.names.index("PaA")] = np.nan
    ids = rng.integers(0, 2**60, n)
    model = GalacticusLike()
    new = model.apply(flux, LS, ids)
    assert new[0, LS.names.index("OII_3727")] == 0.0 and new[1, LS.names.index("Hb")] == -1e-18
    assert np.isnan(new[2, LS.names.index("PaA")])
    j = {name: LS.names.index(name) for name in LS.names}
    np.testing.assert_array_equal(new[:, j["Ha"]], flux[:, j["Ha"]])
    np.testing.assert_array_equal(new[:, j["OIII_4363"]], flux[:, j["OIII_4363"]])
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.log10(new / flux)
    off = model.offsets(LS)
    eps = galaxy_normal(ids)
    np.testing.assert_allclose(d[3:, j["OIII_5007"]], off[j["OIII_5007"]] + 0.19 * eps[3:], atol=1e-12)
    np.testing.assert_allclose(d[3:, j["OII_3727"]], off[j["OII_3727"]] + 0.11 * eps[3:], atol=1e-12)
    # the [O III] doublet moves together: its ratio changes only by the dust term between 4959 and 5007
    np.testing.assert_allclose(d[3:, j["OIII_5007"]] - d[3:, j["OIII_4959"]], off[j["OIII_5007"]] - off[j["OIII_4959"]], atol=1e-12)
    # the deviate is a property of the galaxy: chunks give the same result
    np.testing.assert_array_equal(np.concatenate([model.apply(flux[:2000], LS, ids[:2000]), model.apply(flux[2000:], LS, ids[2000:])]), new)


def test_apply_columns_matches_apply_and_passes_other_columns():
    rng = np.random.default_rng(4)
    n = 1000
    flux = 10 ** rng.normal(-16, 0.5, (n, len(LS)))
    ids = rng.integers(0, 2**60, n)
    data = {f"{name}_flux": flux[:, k] for k, name in enumerate(LS.names)}
    data.update(gal_id=ids, redshift_rsd=rng.uniform(0.5, 3, n))
    model = GalacticusLike()
    out = model.apply_columns(data, LS)
    np.testing.assert_allclose(np.stack([out[f"{name}_flux"] for name in LS.names], axis=1), model.apply(flux, LS, ids), rtol=1e-14)
    assert out["redshift_rsd"] is data["redshift_rsd"]
    part = model.apply_columns({"Ha_flux": data["Ha_flux"], "OII_3727_flux": data["OII_3727_flux"], "gal_id": ids}, LS)
    np.testing.assert_array_equal(part["OII_3727_flux"], out["OII_3727_flux"])  # a subset of the columns is enough


def test_scatter_needs_gal_id_and_constant_factors_do_not():
    flux = np.full((3, len(LS)), 1e-16)
    with pytest.raises(ValueError, match="gal_id"):
        GalacticusLike().apply(flux, LS)
    out = GalacticusLike(sigma={}, galaxy_dust=()).apply(flux, LS)
    np.testing.assert_allclose(out[0], 1e-16 * 10 ** GalacticusLike(sigma={}, galaxy_dust=()).offsets(LS))


def test_galaxy_dust_keeps_dust_free_galaxies_and_swaps_the_reddening():
    model = GalacticusLike(sigma={})
    j = {n: LS.names.index(n) for n in LS.names}
    a_d = np.array([0.0, 0.1, 0.3455, 1.0, 2.0])                        # diffsky A(Hα) of five galaxies
    flux = np.full((5, len(LS)), 1e-16)
    flux[:, j["PaA"]] = 1e-16 * 10 ** (model.paa_ha_intrinsic + 0.4 * a_d * (1 - model.diffsky_dust_ratio["PaA"]))
    flux[:, j["SIII_9532"]] = 0.35e-16 * 10 ** (0.4 * a_d * (1 - model.diffsky_dust_ratio["SIII_9532"]))
    np.testing.assert_allclose(model.galaxy_a_ha(flux[:, j["Ha"]], flux[:, j["PaA"]]), a_d, atol=1e-12)
    new = model.apply(flux, LS)
    np.testing.assert_allclose(new[:, j["Ha"]], flux[:, j["Ha"]])
    # a galaxy without dust keeps its dust-free [S III]/Hα; Paα/Hα takes only the case-B delta
    assert new[0, j["SIII_9532"]] == pytest.approx(flux[0, j["SIII_9532"]])
    assert np.log10(new[0, j["PaA"]] / new[0, j["Ha"]]) == pytest.approx(model.paa_ha_intrinsic - 0.055)
    # every galaxy ends with the Calzetti reddening of the Galacticus A(Hα) at its dust quantile
    k = dict(zip(LS.names, calzetti_k(LS.rest_waves)))
    a_g = model.galacticus_a_ha(a_d)
    np.testing.assert_allclose(np.log10(new[:, j["PaA"]] / new[:, j["Ha"]]),
                               model.paa_ha_intrinsic - 0.055 + 0.4 * a_g * (1 - k["PaA"] / k["Ha"]), atol=1e-12)
    np.testing.assert_allclose(np.log10(new[:, j["SIII_9532"]] / 0.35e-16), 0.4 * a_g * (1 - k["SIII_9532"] / k["Ha"]), atol=1e-12)
    assert model.galacticus_a_ha(0.3455) == pytest.approx(0.8928)       # diffsky's median dust -> Galacticus's median
    with pytest.raises(KeyError, match="PaA_flux"):                     # the dust of a galaxy is read from its Paα/Hα
        model.apply_columns({"Ha_flux": flux[:, 0], "SIII_9532_flux": flux[:, j["SIII_9532"]]}, LS)


def test_build_and_round_trip():
    assert build_rescaling(None) is None
    model = build_rescaling("galacticus_like")
    assert isinstance(model, GalacticusLike) and model.seed == DEFAULT_SEED
    again = build_rescaling(model.to_config())
    np.testing.assert_array_equal(again.offsets(LS), model.offsets(LS))
    np.testing.assert_array_equal(again.sigmas(LS), model.sigmas(LS))
    explicit = build_rescaling({"type": "line_factors", "log_offsets": {"OII_3727": 0.2}, "scatter": {"OII_3727": 0.1}, "seed": 3})
    assert isinstance(explicit, LineFactors) and explicit.offsets(LS)[LS.names.index("OII_3727")] == 0.2
    assert build_rescaling(explicit.to_config()).to_config() == explicit.to_config()
    with pytest.raises(KeyError, match="unknown rescaling model"):
        build_rescaling({"type": "nope"})
    with pytest.raises(KeyError, match="unknown features"):
        LineFactors({"NII_6585": 0.1}).offsets(LS)
    with pytest.raises(KeyError, match="no counterpart"):
        GalacticusLike(delta={"OIII_4363": 0.1})


def test_max_log_boost():
    model = GalacticusLike()
    off = dict(zip(LS.names, model.offsets(LS)))
    assert model.max_log_boost(LS, n_sigma=5) == pytest.approx(off["OIII_5007"] + 5 * 0.19)

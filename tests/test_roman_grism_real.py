"""The S/N route on the real mock: a base catalog with the size column, make_catalog and assign agree."""
import warnings

import numpy as np
import pytest

from romanmock import BaseCatalog, RomanMockConfig, RomanMockPipeline
from romanmock.detection import observed_wavelength, pack_bitmask

from conftest import CATALOG_DIR, catalog_available

pytestmark = pytest.mark.skipif(not catalog_available(), reason="diffsky mock not available")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from romanmock.io import list_catalog_files, step_from_filename

    tmp = tmp_path_factory.mktemp("roman_grism")
    files = sorted(list_catalog_files(CATALOG_DIR), key=lambda p: p.stat().st_size)
    steps = [step_from_filename(files[8])]
    cfg = RomanMockConfig.from_dict({
        "catalog": {"path": CATALOG_DIR, "steps": steps},
        # the size column is stored for step 2; the base itself is cut at the point-source limit, the deepest
        # case of the model, so that every size chosen afterwards is contained
        "base": {"dir": str(tmp / "base"), "columns": ["gal_id", "ra", "dec", "redshift_true", "r50_disk_2d"]},
        "output": {"dir": str(tmp / "out"), "passthrough_columns": ["gal_id", "ra", "dec", "redshift_true"], "store_snr": True},
        "sensitivity": {"type": "roman_grism", "snr": 5.0, "secondary_snr": None, "size": 0.0},
        "seed": 3,
    })
    pipe = RomanMockPipeline(cfg)
    base_files = pipe.build_base(verbose=False)
    return tmp, steps, pipe, base_files


def test_base_stores_the_size_column(built):
    import opencosmo as oc

    tmp, steps, pipe, base_files = built
    base = oc.open(*base_files)
    assert "r50_disk_2d" in base.columns and "redshift_true" in base.columns
    assert BaseCatalog(tmp / "base").description["columns"] == ["gal_id", "ra", "dec", "redshift_true", "r50_disk_2d"]


def test_make_catalog_snr_route(built):
    tmp, steps, pipe, base_files = built
    b = BaseCatalog(tmp / "base")
    cat = b.make_catalog(snr=5.0, secondary_snr=2.0, size="r50_disk_2d", columns="all", outcomes=None, n_workers=1, verbose=False)
    assert "S/N >= 5" in repr(cat) and cat.settings["sensitivity"]["type"] == "roman_grism"
    # the rule by hand: S/N from the fluxes, the sizes from r50_disk_2d at redshift_true
    sens = b._sensitivity(snr=5.0, secondary_snr=2.0, size="r50_disk_2d")
    size = sens.angular_size({"r50_disk_2d": cat["r50_disk_2d"], "redshift_true": cat["redshift_true"]}, b.cosmology)
    wave = observed_wavelength(b.line_set.rest_waves, cat["redshift_rsd"])
    flux = np.stack([cat[c] for c in b.flux_columns], axis=1)
    snr = sens.snr_of(flux, wave, size[:, None])
    primary = snr > 5.0
    expect = primary | (primary.any(axis=1)[:, None] & (snr > 2.0))
    assert primary.any(axis=1).all()
    np.testing.assert_array_equal(cat["line_bitmask"], pack_bitmask(expect))
    single, multi = (cat["n_det"] == 1).sum(), (cat["n_det"] > 1).sum()
    assert single + multi == len(cat) > 0
    # one size for all: the point source (the base selection) keeps every galaxy, 0.3" fewer, 1" fewer still; the
    # sizes of the mock (a 4.7 kpc disk at this slice's z ~ 0.22 spans 1.3") give fewer than the point source
    n_point = len(b.make_catalog(snr=5.0, size=0.0, outcomes=None, n_workers=1, verbose=False))
    n_0p3 = len(b.make_catalog(snr=5.0, size=0.3, outcomes=None, n_workers=1, verbose=False))
    n_big = len(b.make_catalog(snr=5.0, size=1.0, outcomes=None, n_workers=1, verbose=False))
    assert n_point == b.n_galaxies > n_0p3 > n_big > 0
    assert n_point > len(cat) > 0
    # the two routes cannot be mixed, and the sizes cannot be deeper than the base
    with pytest.raises(ValueError, match="two ways"):
        b.make_catalog(flux_limit=1e-16, snr=5.0, n_workers=1, verbose=False)
    with pytest.raises(ValueError, match="deeper"):
        b.make_catalog(snr=4.0, size=0.0, n_workers=1, verbose=False)
    # the flux-limit route still works on the same base catalog
    old = b.make_catalog(flux_limit=1e-16, secondary_flux_limit=4e-17, outcomes=None, n_workers=1, verbose=False)
    assert old.settings["sensitivity"]["type"] == "constant" and len(old) > 0


def test_assign_equals_make_catalog(built):
    import opencosmo as oc

    tmp, steps, pipe, base_files = built
    cfg = RomanMockConfig.from_dict({
        "source": "base", "base": {"dir": str(tmp / "base")},
        "output": {"dir": str(tmp / "assigned"), "passthrough_columns": ["gal_id"], "store_snr": True},
        "sensitivity": {"type": "roman_grism", "snr": 5.0, "secondary_snr": 2.0, "size": "r50_disk_2d"},
        "seed": 3,
    })
    files = RomanMockPipeline(cfg).run(verbose=False)
    d = oc.open(*files).select("gal_id", "line_bitmask", "redshift_assigned", "id_outcome", "Ha_snr").get_data("numpy", wrap_single=True)
    cat = BaseCatalog(tmp / "base").make_catalog(snr=5.0, secondary_snr=2.0, size="r50_disk_2d", outcomes=None, seed=3,
                                                 columns=["gal_id", "line_bitmask", "redshift_assigned", "id_outcome"],
                                                 n_workers=1, verbose=False)
    assert len(cat) == len(d["gal_id"]) > 0
    order_a, order_b = np.argsort(d["gal_id"]), np.argsort(cat["gal_id"])
    for c in ["gal_id", "line_bitmask", "id_outcome"]:
        np.testing.assert_array_equal(np.asarray(d[c])[order_a], cat[c][order_b], err_msg=c)
    np.testing.assert_allclose(np.asarray(d["redshift_assigned"])[order_a], cat["redshift_assigned"][order_b], equal_nan=True)
    assert np.all(np.asarray(d["Ha_snr"]) >= 0)
    # a config that asks for a size column the base does not have is refused up front
    bad = RomanMockConfig.from_dict({**cfg.to_dict(), "sensitivity": {"type": "roman_grism", "size": "r50_bulge_2d"}})
    with pytest.raises(KeyError, match="r50_bulge_2d"):
        RomanMockPipeline(bad)


def test_noisy_detection_warns_and_fallback_is_counted(built):
    """A noisy detection model close to the base limit warns in make_catalog as in the pipeline; the
    galaxies that took a fallback size are counted (none in the mock)."""
    tmp, steps, pipe, base_files = built
    b = BaseCatalog(tmp / "base")
    kw = dict(outcomes=None, n_workers=1, verbose=False)
    with pytest.warns(UserWarning, match="noisy detection"):
        noisy = b.make_catalog(snr=5.0, size=0.0, detection={"type": "gaussian_noise"}, **kw)
    assert len(noisy) > 0
    with warnings.catch_warnings(record=True) as seen:                   # 0.3" is 2.6 x shallower: the base is deep enough
        warnings.simplefilter("always")
        b.make_catalog(snr=5.0, size=0.3, detection={"type": "gaussian_noise"}, **kw)
    assert not [w for w in seen if "noisy detection" in str(w.message)]
    cat = b.make_catalog(snr=5.0, size={"column": "r50_disk_2d", "fallback": 0.3}, **kw)
    assert cat.info["n_fallback_size"] == 0 and cat.settings["sensitivity"]["size"]["fallback"] == 0.3
    assert len(cat) == len(b.make_catalog(snr=5.0, size="r50_disk_2d", **kw))


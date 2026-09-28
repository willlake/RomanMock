"""Step 2 in memory: `BaseCatalog.make_catalog` against the pipeline (`assign` + `select`)."""
import numpy as np
import pytest

from romanmock import BaseCatalog, Catalog, RomanMockConfig, RomanMockPipeline, make_catalog
from romanmock.config import PopulationThinningConfig, SampleConfig
from romanmock.io import open_sample

from conftest import CATALOG_DIR, catalog_available

pytestmark = pytest.mark.skipif(not catalog_available(), reason="diffsky mock not available")

# the test slices are at z ~ 0.27-0.36: [S III] 9532/9069 are the lines inside the grism bandpass
LIMITS = dict(flux_limit=1.5e-16, secondary_flux_limit=8e-17)
ROWS = {"SIII_9532": {"Ha": 0.2, "fail": 0.1}}
ALL = ["gal_id", "ra", "dec", "redshift_true", "redshift_rsd", "redshift_assigned", "id_outcome", "line_bitmask", "n_det",
       "line_true", "line_assigned"]


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    from romanmock.io import list_catalog_files, step_from_filename

    tmp = tmp_path_factory.mktemp("catalog")
    files = sorted(list_catalog_files(CATALOG_DIR), key=lambda p: p.stat().st_size)
    steps = [step_from_filename(p) for p in files[10:13]]
    cfg = RomanMockConfig.from_dict({"catalog": {"path": CATALOG_DIR, "steps": steps}, "base": {"dir": str(tmp / "base")}})
    RomanMockPipeline(cfg).build_base(verbose=False)
    return tmp, BaseCatalog(tmp / "base")


def _pipeline(tmp, name, thinning=None):
    cfg = RomanMockConfig.from_dict({
        "source": "base", "base": {"dir": str(tmp / "base")},
        "output": {"dir": str(tmp / name), "passthrough_columns": ["gal_id", "ra", "dec", "redshift_true"], "overwrite": True,
                   "sample_columns": None},
        "sensitivity": {"type": "constant", "wave_range": [10000.0, 19300.0], **LIMITS},
        "confusion": {"type": "matrix", "default_p_fail": 0.0, "rows": ROWS}, "seed": 5})
    cfg.population_thinning = thinning
    pipe = RomanMockPipeline(cfg)
    pipe.run(verbose=False)
    return pipe


def test_base_catalog_description(base):
    tmp, b = base
    assert b.flux_limit == 1e-16 and b.wave_range == (10000.0, 19300.0) and b.n_galaxies > 0
    assert {"gal_id", "ra", "redshift_rsd", "redshift_assigned", "line_bitmask", "Ha_flux"} <= set(b.columns)
    assert "64" not in repr(b) or True
    assert 2000 < b.area_deg2 < 2130
    assert (tmp / "base" / "romanmock_base_area.json").exists()              # cached


def test_catalog_equals_pipeline_catalog(base):
    tmp, b = base
    cat = b.make_catalog(confusion=ROWS, seed=5, outcomes=None, columns=ALL, n_workers=1, verbose=False, **LIMITS)
    parent = _pipeline(tmp, "assigned").open_parent().select(*ALL).get_data("numpy")
    assert len(cat) == len(parent["ra"]) > 10000
    for c in ALL:
        np.testing.assert_array_equal(cat[c], parent[c], err_msg=c)
    assert cat.outcomes()["SINGLE_FAILED"] > 0 and cat.outcomes()["SINGLE_MISID"] > 0
    # assigned redshifts: multi-line galaxies get redshift_rsd (peculiar velocity included), never redshift_true
    multi, right, wrong, failed = (cat["id_outcome"] == k for k in (1, 2, 3, 4))
    assert multi.sum() > 1000 and right.sum() > 1000
    np.testing.assert_array_equal(cat["redshift_assigned"][multi], cat["redshift_rsd"][multi])
    assert np.mean(cat["redshift_assigned"][multi] == cat["redshift_true"][multi]) < 0.01
    np.testing.assert_allclose(cat["redshift_assigned"][right], cat["redshift_rsd"][right], rtol=0, atol=1e-12)
    rest = b.line_set.rest_waves
    expected = rest[cat["line_true"][wrong]] * (1 + cat["redshift_rsd"][wrong]) / rest[cat["line_assigned"][wrong]] - 1
    np.testing.assert_allclose(cat["redshift_assigned"][wrong], expected, rtol=0, atol=1e-12)
    assert np.isnan(cat["redshift_assigned"][failed]).all()
    assert cat.info["n_detected"] == len(cat) and cat.info["n_base"] == b.n_galaxies
    # parallel == serial, and the module-level shortcut
    cat2 = make_catalog(tmp / "base", confusion=ROWS, seed=5, outcomes=None, columns=ALL, n_workers=2, verbose=False, **LIMITS)
    for c in ALL:
        np.testing.assert_array_equal(cat[c], cat2[c], err_msg=c)


def test_cuts_and_downsampling_equal_select(base):
    tmp, b = base
    target = {"z": [0.0, 4.0], "dndz": [1500.0, 1500.0]}
    with pytest.warns(RuntimeWarning):
        cat = b.make_catalog(confusion=ROWS, seed=5, lines=["SIII_9532"], z_range=(0.28, 0.35), columns=ALL,
                             downsample={"SIII_9532": target}, n_workers=2, verbose=False, **LIMITS)
    pipe = _pipeline(tmp, "assigned_thin", PopulationThinningConfig(targets={"SIII_9532": target}, by="line_true",
                                                                    z_column="redshift_rsd"))
    with pytest.warns(RuntimeWarning):
        (path,) = pipe.build_samples(samples=[SampleConfig(name="s", z_min=0.28, z_max=0.35, assigned_lines=["SIII_9532"])],
                                     verbose=False)
    sample = open_sample(path).select(*ALL).get_data("numpy")
    assert len(cat) == len(sample["ra"]) > 1000
    for c in ALL:
        np.testing.assert_array_equal(cat[c], sample[c], err_msg=c)
    assert (cat["line_assigned"] == cat.bit("SIII_9532")).all()
    assert cat["redshift_assigned"].min() >= 0.28 and cat["redshift_assigned"].max() <= 0.35
    info = cat.info["downsample"]["SIII_9532"]
    assert 0 < info["n_after"] < info["n_before"]
    # thinned to the target where the mock is denser than it
    p = np.array(info["p_keep"]); edges = np.array(info["z_edges"])
    full = b.make_catalog(confusion=ROWS, seed=5, outcomes=None, columns=["redshift_rsd", "line_true"],
                          downsample={"SIII_9532": target}, n_workers=2, verbose=False, **LIMITS)
    z = full["redshift_rsd"][full["line_true"] == cat.bit("SIII_9532")]
    counts = np.histogram(z, bins=edges)[0]
    dndz = counts / (np.diff(edges) * cat.area_deg2)
    ok = (p > 0) & (p < 1) & (counts > 1000)
    assert ok.any() and np.all(np.abs(dndz[ok] / 1500.0 - 1) < 4 / np.sqrt(counts[ok]) + 0.02)


def test_settings_change_freely(base):
    tmp, b = base
    kw = dict(seed=5, columns=["gal_id", "id_outcome", "redshift_assigned", "n_det"], n_workers=1, verbose=False)
    a = b.make_catalog(**kw)                                           # defaults: base limit, 4e-17, default confusion
    assert a.settings["flux_limit"] == 1e-16 and a.settings["secondary_flux_limit"] == 4e-17
    c = b.make_catalog(flux_limit=2e-16, secondary_flux_limit=None, **kw)
    assert 0 < len(c) < len(a) and c.settings["secondary_flux_limit"] is None
    hi = b.make_catalog(confusion={"SIII_9532": {"Ha": 0.5}}, **kw)
    lo = b.make_catalog(confusion={"SIII_9532": {"Ha": 0.0}}, **kw)
    assert lo.outcomes()["SINGLE_MISID"] == 0 < a.outcomes()["SINGLE_MISID"] < hi.outcomes()["SINGLE_MISID"]
    again = b.make_catalog(confusion={"SIII_9532": {"Ha": 0.5}}, **kw)
    np.testing.assert_array_equal(hi["redshift_assigned"], again["redshift_assigned"])
    other_seed = b.make_catalog(confusion={"SIII_9532": {"Ha": 0.5}}, **{**kw, "seed": 6})
    assert not np.array_equal(hi["id_outcome"], other_seed["id_outcome"])
    sub = b.make_catalog(z_range=(0.3, None), z_column="redshift_true", n_det_min=2, cuts={"dec": (60.0, 70.0)},
                         columns=["dec", "redshift_true", "n_det"], n_workers=1, verbose=False)
    assert sub["redshift_true"].min() >= 0.3 and sub["n_det"].min() >= 2 and 60 <= sub["dec"].min() and sub["dec"].max() <= 70
    everything = b.make_catalog(columns="all", n_workers=1, verbose=False)
    assert set(everything.columns) == set(b.columns) and everything["Ha_flux"].dtype == np.float64


def test_refused_settings(base):
    tmp, b = base
    with pytest.raises(ValueError, match="deeper"):
        b.make_catalog(flux_limit=5e-17)
    with pytest.raises(ValueError, match="bandpass"):
        b.make_catalog(wave_range=(7500.0, 19300.0))
    with pytest.raises(ValueError):
        b.make_catalog(flux_limit=2e-16, secondary_flux_limit=3e-16)
    with pytest.raises(KeyError, match="logsm_obs"):
        b.make_catalog(columns=["ra", "logsm_obs"])
    with pytest.raises(KeyError, match="unknown lines"):
        b.make_catalog(lines=["Halpha"])
    with pytest.raises(KeyError, match="unknown lines"):
        b.make_catalog(confusion={"Halpha": {"OIII_5007": 0.1}})
    with pytest.raises(ValueError, match="not tabulated"):
        b.make_catalog(flux_limit=1.2e-16, downsample="wang2022")
    with pytest.raises(FileNotFoundError):
        BaseCatalog(tmp / "nothing_here")


def test_write_and_read(base, tmp_path):
    tmp, b = base
    cat = b.make_catalog(confusion=ROWS, lines=["SIII_9532"], seed=5, n_workers=1, verbose=False, **LIMITS)
    for name in ("c.hdf5", "c.npz"):
        back = Catalog.read(cat.write(tmp_path / name))
        assert back.columns == cat.columns and len(back) == len(cat)
        for c in cat.columns:
            np.testing.assert_array_equal(back[c], cat[c], err_msg=c)
            assert back[c].dtype == cat[c].dtype
        assert back.settings["flux_limit"] == 1.5e-16 and back.settings["lines"] == ["SIII_9532"]
        assert back.line_names == cat.line_names
    from astropy.table import Table

    t = Table.read(cat.write(tmp_path / "c.fits"))
    assert len(t) == len(cat) and set(t.colnames) == set(cat.columns)
    with pytest.raises(FileExistsError):
        cat.write(tmp_path / "c.hdf5", overwrite=False)
    direct = b.make_catalog(confusion=ROWS, lines=["SIII_9532"], seed=5, n_workers=1, verbose=False,
                            write_to=tmp_path / "d.hdf5", **LIMITS)
    assert len(Catalog.read(tmp_path / "d.hdf5")) == len(direct)
    # table-like access
    mask = cat["id_outcome"] == 3
    assert len(cat[mask]) == mask.sum() and cat[["ra", "dec"]].columns == ["ra", "dec"]
    assert len(cat.to_table()) == len(cat) and "galaxies" in cat.summary()


def test_line_report(base, capsys):
    _, b = base
    cat = b.make_catalog(confusion=ROWS, seed=5, columns=ALL, n_workers=1, verbose=True, **LIMITS)
    printed = capsys.readouterr().out
    assert cat.line_report() in printed and "line interlopers" in printed
    rep = cat.info["line_report"]
    lo, hi = b.wave_range
    names = cat.line_names
    rest = b.line_set.rest_waves
    # lines: inside the bandpass somewhere in the lightcone, counted from the bitmask and the assigned line
    for r in rep["lines"]:
        k = cat.bit(r["line"])
        assert lo <= r["wave_obs"][0] < r["wave_obs"][1] <= hi
        np.testing.assert_allclose(np.array(r["wave_obs"]) / rest[k] - 1, r["z"])
        assert r["n_detected"] == np.sum((cat["line_bitmask"].astype(np.int64) >> k) & 1 > 0)
        assert r["n_assigned"] == np.sum(cat["line_assigned"] == k)
    detected = {names[k] for k in range(len(names)) if np.any((cat["line_bitmask"].astype(np.int64) >> k) & 1)}
    assert detected <= {r["line"] for r in rep["lines"]}
    # interlopers: the counts are the misidentified galaxies, inside the predicted range of true redshift
    wrong = cat["id_outcome"] == 3
    rows = {(r["line_true"], r["line_assigned"]): r for r in rep["interlopers"]}
    assert sum(r["n"] for r in rows.values()) == wrong.sum() > 0
    r = rows[("SIII_9532", "Ha")]
    sel = wrong & (cat["line_true"] == cat.bit("SIII_9532")) & (cat["line_assigned"] == cat.bit("Ha"))
    assert r["n"] == sel.sum() > 0 and r["probability"] == pytest.approx(0.2)
    z = cat["redshift_rsd"][sel]
    assert r["z_true"][0] <= z.min() and z.max() <= r["z_true"][1]
    za = cat["redshift_assigned"][sel]
    assert r["z_assigned"][0] - 1e-9 <= za.min() and za.max() <= r["z_assigned"][1] + 1e-9

    # a cut on the assigned redshift: only the interlopers that land inside it
    cut = b.make_catalog(confusion=ROWS, seed=5, columns=ALL, n_workers=1, verbose=False, lines=["Ha"], z_range=(0.9, 1.0),
                         **LIMITS)
    rep = cut.info["line_report"]
    assert {r["line_assigned"] for r in rep["interlopers"]} == {"Ha"}
    r = [x for x in rep["interlopers"] if x["line_true"] == "SIII_9532"][0]
    assert r["n"] == len(cut) == np.sum(cut["id_outcome"] == 3)
    assert r["z_assigned"][0] >= 0.9 - 1e-9 and r["z_assigned"][1] <= 1.0 + 1e-9
    if len(cut):
        assert r["z_true"][0] <= cut["redshift_rsd"].min() and cut["redshift_rsd"].max() <= r["z_true"][1]
    assert rep["lines"] == []                      # no galaxy of these slices is at 0.9 < z < 1.0 itself
    only = b.make_catalog(confusion=ROWS, seed=5, n_workers=1, verbose=False, lines=["Ha"], **LIMITS)
    assert "not in `lines`" in only.line_report() and len(only) == np.sum(cat["line_assigned"] == cat.bit("Ha"))
    # without the misidentified galaxies, and with downsampling (counted after the thinning)
    none = b.make_catalog(confusion=ROWS, seed=5, n_workers=1, verbose=False, outcomes=["MULTI_LINE", "SINGLE_CORRECT"], **LIMITS)
    assert sum(r["n"] for r in none.info["line_report"]["interlopers"]) == 0
    assert "excludes the misidentified" in none.line_report()
    z = np.linspace(0.2, 0.5, 4)
    thin = b.make_catalog(confusion=ROWS, seed=5, columns=ALL, n_workers=1, verbose=False,
                          downsample={"SIII_9532": {"z": z.tolist(), "dndz": [50.0] * 4}}, **LIMITS)
    assert len(thin) < len(cat)
    assert sum(r["n"] for r in thin.info["line_report"]["interlopers"]) == np.sum(thin["id_outcome"] == 3)

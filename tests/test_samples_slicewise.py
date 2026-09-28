"""Stage B built slice by slice: passes over the parent catalog, one output file per slice."""
import numpy as np
import pytest
import yaml

from romanmock import RomanMockConfig, RomanMockPipeline
from romanmock.config import DownsampleConfig, PopulationThinningConfig, SampleConfig
from romanmock.io import open_sample, sample_files
from romanmock.samples import process_slice, thinning_keep

from conftest import CATALOG_DIR, catalog_available

pytestmark = pytest.mark.skipif(not catalog_available(), reason="diffsky mock not available")

COLS = ["ra", "dec", "redshift_true", "redshift_rsd", "redshift_assigned", "id_outcome"]


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    from romanmock.io import list_catalog_files, step_from_filename

    tmp = tmp_path_factory.mktemp("slicewise")
    files = sorted(list_catalog_files(CATALOG_DIR), key=lambda p: p.stat().st_size)
    steps = [step_from_filename(p) for p in files[10:13]]
    cfg = RomanMockConfig.from_dict({
        "catalog": {"path": CATALOG_DIR, "steps": steps, "rest_waves_from_catalog": False},
        "output": {"dir": str(tmp), "overwrite": True},
        "sensitivity": {"type": "constant", "flux_limit": 1e-16, "secondary_flux_limit": None, "wave_range": [10000.0, 19300.0]},
        "seed": 11,
    })
    pipe = RomanMockPipeline(cfg, aux_data=None)
    pipe.run(verbose=False)
    return tmp, pipe


def _data(path, cols=COLS):
    return open_sample(path).select(*cols).get_data("numpy")


def test_plain_sample_matches_whole_catalog_mask(parent):
    from romanmock.selection import SampleDefinition, sample_mask

    tmp, pipe = parent
    # the test slices are at z ~ 0.27-0.36, where [S III] 9532/9069 are the lines inside the grism bandpass
    scfg = SampleConfig(name="ha", z_min=0.2, z_max=3.0, assigned_lines=["SIII_9532"])
    (out,) = pipe.build_samples(samples=[scfg], verbose=False)
    assert out == tmp / "samples" / "ha" and out.is_dir()
    assert all(f.name.startswith("lc_cores-") and f.name.endswith(".ha.hdf5") for f in sample_files(out))
    whole = pipe.open_parent()
    d = whole.select(*COLS, "line_assigned", "line_bitmask", "n_det").get_data("numpy")
    mask = sample_mask(d, pipe.line_set, SampleDefinition(name="ha", z_min=0.2, z_max=3.0, assigned_lines=["SIII_9532"]))
    assert mask.sum() > 10000
    s = _data(out)
    assert sorted(open_sample(out).columns) == sorted([*COLS, "redshift"])
    for c in COLS:
        np.testing.assert_array_equal(s[c], d[c][mask], err_msg=c)
    meta = yaml.safe_load((tmp / "samples" / "ha_metadata.yaml").read_text())
    assert meta["n_galaxies"] == mask.sum() and meta["layout"] == "slices" and len(meta["files"]) == len(sample_files(out))


def test_serial_parallel_and_single_layout_agree(parent):
    tmp, pipe = parent
    pipe.config.population_thinning = PopulationThinningConfig(
        targets={"SIII_9532": {"z": [0.0, 4.0], "dndz": [2000.0, 2000.0]}}, by="line_true", z_column="redshift_rsd")
    want = {"a": 500.0, "b": 5.0}
    samples = [SampleConfig(name="a", z_min=0.28, z_max=0.36, assigned_lines=["SIII_9532"],
                            downsample=DownsampleConfig(target={"z": [0.0, 4.0], "dndz": [want["a"]] * 2}, nbins=8, smooth_bins=0.0)),
               # Ha is outside the bandpass here: this sample consists of [S III] galaxies misidentified as Ha.
               # It has no z range, so its bins come from a range pass
               SampleConfig(name="b", assigned_lines=["Ha"],
                            downsample=DownsampleConfig(target={"z": [0.0, 4.0], "dndz": [want["b"]] * 2}, nbins=5, smooth_bins=0.0))]
    try:
        with pytest.warns(RuntimeWarning):
            out1 = pipe.build_samples(samples=samples, verbose=False)
        d1 = [_data(p) for p in out1]
        m1 = [yaml.safe_load((tmp / "samples" / f"{n}_metadata.yaml").read_text()) for n in "ab"]
        with pytest.warns(RuntimeWarning):
            out2 = pipe.build_samples(samples=samples, verbose=False, n_workers=2)
        d2 = [_data(p) for p in out2]                  # read now: the other layout replaces these directories
        pipe.config.output.sample_layout = "single"
        with pytest.warns(RuntimeWarning):
            out3 = pipe.build_samples(samples=samples, verbose=False)
        assert [p.name for p in out3] == ["a.hdf5", "b.hdf5"] and not (tmp / "samples" / "a").exists()
        for a, b, p3, meta in zip(d1, d2, out3, m1):
            assert len(a["ra"]) == meta["n_galaxies"] > 0
            c = _data(p3)
            for col in COLS:
                np.testing.assert_array_equal(a[col], b[col], err_msg=col)
                np.testing.assert_array_equal(np.sort(a[col]), np.sort(c[col]), err_msg=col)
            # downsampled to the target wherever the mock is dense enough
            p = np.array(meta["p_keep"]); edges = np.array(meta["z_edges"])
            counts = np.histogram(a["redshift_assigned"], bins=edges)[0]
            dndz = counts / (np.diff(edges) * meta["area_deg2"])
            thinned = (p > 0) & (p < 1) & (counts > 100)
            assert thinned.any(), (meta["sample"]["name"], p, counts)
            tol = 4.0 / np.sqrt(counts[thinned]) + 0.02
            assert np.all(np.abs(dndz[thinned] / want[meta["sample"]["name"]] - 1.0) < tol)
        info = m1[0]["population_thinning"]["SIII_9532"]
        assert 0 < info["n_after"] < info["n_before"]
    finally:
        pipe.config.population_thinning = None
        pipe.config.output.sample_layout = "slices"


def test_thinning_is_reproducible_per_slice(parent):
    tmp, pipe = parent
    f = pipe.parent_files()[0]
    plan = {"seed": 3, "line_set": pipe.line_set.to_config(), "columns": ["redshift_assigned", "redshift_rsd", "line_true"],
            "samples": [], "compress_index": True,
            "thinning": {"by": "line_true", "z_column": "redshift_rsd", "outside_range": "drop",
                         "populations": {"Ha": {"bit": 5, "edges": np.array([0.0, 2.0, 4.0]), "p_keep": np.array([0.5, 0.0])}}}}
    import opencosmo as oc

    d = oc.open(f).select(*plan["columns"]).get_data("numpy")
    from romanmock.io import step_from_filename

    step = step_from_filename(f)
    k1, k2 = thinning_keep(d, plan, step), thinning_keep(d, plan, step)
    np.testing.assert_array_equal(k1, k2)
    ha = d["line_true"] == 5        # [S III] 9532, the dominant line of these slices
    assert not k1[~ha].any()                                   # populations without a target are dropped
    low = ha & (d["redshift_rsd"] < 2.0)
    assert low.sum() > 1000
    assert abs(k1[low].mean() - 0.5) < 0.05
    assert not np.array_equal(k1, thinning_keep(d, plan, step + 1))
    res = process_slice(plan, str(f), {"pop_hist": ["Ha"]})
    assert res["pop_n"]["Ha"] == ha.sum() and res["pop_hist"]["Ha"].sum() == np.isfinite(d["redshift_rsd"][ha]).sum()


def test_existing_sample_is_protected(parent):
    tmp, pipe = parent
    scfg = SampleConfig(name="keepme", assigned_lines=["SIII_9532"])
    pipe.build_samples(samples=[scfg], verbose=False)
    pipe.config.output.overwrite = False
    try:
        with pytest.raises(FileExistsError):
            pipe.build_samples(samples=[scfg], verbose=False)
    finally:
        pipe.config.output.overwrite = True
    with pytest.raises(ValueError, match="unique"):
        pipe.build_samples(samples=[scfg, scfg], verbose=False)

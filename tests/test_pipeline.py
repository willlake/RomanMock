import numpy as np
import pytest

from romanmock import RomanMockConfig, RomanMockPipeline
from romanmock.lines import PLAN_SPECIES
from romanmock.redshift import IdOutcome

from conftest import CATALOG_DIR, catalog_available


def test_process_arrays_synthetic(synthetic, cosmo):
    cfg = RomanMockConfig.from_dict({
        "line_set": "plan_species",
        "confusion": {"type": "matrix", "rows": {"OIII": {"Ha": 0.3}}},
        "output": {"store_snr": True, "store_wave_obs": True},
    })
    pipe = RomanMockPipeline(cfg)
    assert pipe.line_set is PLAN_SPECIES
    out = pipe.process_arrays(synthetic, cosmo, np.random.default_rng(0))
    assert set(pipe.flux_columns) <= set(out)
    assert "Ha_snr" in out and "Ha_wave_obs" in out
    assert out["line_bitmask"].dtype == np.uint32
    assert (out["n_det"] <= len(PLAN_SPECIES)).all()
    assert (out["id_outcome"] == IdOutcome.SINGLE_MISID).any()
    # deterministic given the seed
    out2 = pipe.process_arrays(synthetic, cosmo, np.random.default_rng(0))
    np.testing.assert_array_equal(out["z_obs"][np.isfinite(out["z_obs"])], out2["z_obs"][np.isfinite(out2["z_obs"])])


@pytest.mark.skipif(not catalog_available(), reason="diffsky mock not available")
def test_run_on_real_slice(tmp_path):
    import opencosmo as oc

    from romanmock.io import list_catalog_files, step_from_filename

    files = list_catalog_files(CATALOG_DIR)
    small = min(files, key=lambda p: p.stat().st_size)
    cfg = RomanMockConfig.from_dict({
        "catalog": {"path": CATALOG_DIR, "rest_waves_from_catalog": False},
        "output": {"dir": str(tmp_path), "overwrite": True, "keep_undetected": True,
                   "passthrough_columns": ["gal_id", "ra", "dec", "redshift_true", "redshift_obs", "logsm_obs"]},
        "sensitivity": {"type": "constant", "flux_limit": 1e-17, "wave_range": [3000.0, 25000.0]},
        "confusion": {"type": "identity", "p_fail": 0.1},
    })
    pipe = RomanMockPipeline(cfg, aux_data=None)
    outputs = pipe.run(files=[small], verbose=False)
    assert len(outputs) == 1 and outputs[0].exists()
    back = oc.open(outputs[0])
    assert "line_bitmask" in back.columns and "z_obs" in back.columns
    assert str(back.units["Ha_flux"]) in ("erg / (s cm2)", "erg / (cm2 s)")
    d = back.select("z_obs", "id_outcome", "redshift_obs").get_data("numpy", wrap_single=True)
    ok = d["id_outcome"] == IdOutcome.MULTI_LINE
    np.testing.assert_allclose(d["z_obs"][ok], d["redshift_obs"][ok])
    assert (tmp_path / f"{cfg.output.tag}_metadata.yaml").exists()


@pytest.mark.skipif(not catalog_available(), reason="diffsky mock not available")
def test_parallel_run_matches_serial(tmp_path):
    import opencosmo as oc

    from romanmock.io import list_catalog_files

    files = sorted(list_catalog_files(CATALOG_DIR), key=lambda p: p.stat().st_size)[:2]
    base = {
        "catalog": {"path": CATALOG_DIR, "rest_waves_from_catalog": False},
        "sensitivity": {"type": "constant", "flux_limit": 1e-17, "wave_range": [3000.0, 25000.0]},
        "confusion": {"type": "matrix", "rows": {"Ha": {"OIII_5007": 0.2, "fail": 0.1}}},
    }
    serial = RomanMockPipeline(RomanMockConfig.from_dict({**base, "output": {"dir": str(tmp_path / "serial")}}), aux_data=None)
    parallel = RomanMockPipeline(RomanMockConfig.from_dict({**base, "output": {"dir": str(tmp_path / "par")}}), aux_data=None)
    out_s = serial.run(files=files, verbose=False)
    out_p = parallel.run(files=files, verbose=False, n_workers=2)
    for a, b in zip(out_s, out_p):
        da = oc.open(a).select("gal_id", "line_bitmask", "z_obs").get_data("numpy", wrap_single=True)
        db = oc.open(b).select("gal_id", "line_bitmask", "z_obs").get_data("numpy", wrap_single=True)
        np.testing.assert_array_equal(da["gal_id"], db["gal_id"])
        np.testing.assert_array_equal(da["line_bitmask"], db["line_bitmask"])
        np.testing.assert_array_equal(np.nan_to_num(da["z_obs"]), np.nan_to_num(db["z_obs"]))


@pytest.mark.skipif(not catalog_available(), reason="diffsky mock not available")
def test_empty_slice_writes_marker(tmp_path):
    from romanmock.io import list_catalog_files

    small = min(list_catalog_files(CATALOG_DIR), key=lambda p: p.stat().st_size)
    cfg = RomanMockConfig.from_dict({
        "catalog": {"path": CATALOG_DIR, "rest_waves_from_catalog": False},
        "output": {"dir": str(tmp_path)},
        "sensitivity": {"type": "constant", "flux_limit": 1e-5, "wave_range": [10000.0, 19300.0]},  # nothing passes
    })
    pipe = RomanMockPipeline(cfg, aux_data=None)
    outputs = pipe.run(files=[small], verbose=False)
    assert outputs == []
    assert list(tmp_path.glob("*.empty")) and not list(tmp_path.glob("lc_cores-*.hdf5"))
    assert pipe.run(files=[small], verbose=False) == []   # marker makes the re-run skip the slice

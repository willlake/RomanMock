"""Two-step workflow: base flux catalog (step 1) and detection/redshifts from it (step 2)."""
import numpy as np
import pytest

from romanmock import RomanMockConfig, RomanMockPipeline

from conftest import CATALOG_DIR, catalog_available

pytestmark = pytest.mark.skipif(not catalog_available(), reason="diffsky mock not available")


def _small_steps(n=2):
    from romanmock.io import list_catalog_files, step_from_filename

    files = sorted(list_catalog_files(CATALOG_DIR), key=lambda p: p.stat().st_size)
    # skip the tiniest slices (no detections) so that the comparison is not trivially empty
    return [step_from_filename(p) for p in files[8:8 + n]]


def _config(tmp_path, steps, **over):
    d = {
        "catalog": {"path": CATALOG_DIR, "steps": steps},
        "base": {"dir": str(tmp_path / "base")},
        "output": {"dir": str(tmp_path / "direct"), "passthrough_columns": ["ra", "dec", "redshift_true"]},
        "sensitivity": {"type": "constant", "flux_limit": 1e-16, "secondary_flux_limit": 4e-17,
                        "wave_range": [10000.0, 19300.0]},
        "seed": 7,
    }
    d.update(over)
    return RomanMockConfig.from_dict(d)


def _read(files, cols):
    import opencosmo as oc

    return oc.open(*files).select(*cols).get_data("numpy", wrap_single=True)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("two_step")
    steps = _small_steps()
    pipe = RomanMockPipeline(_config(tmp, steps))
    base_files = pipe.build_base(verbose=False)
    return tmp, steps, pipe, base_files


def test_base_catalog_content(built):
    import opencosmo as oc

    tmp, steps, pipe, base_files = built
    assert base_files
    base = oc.open(*base_files)
    expected = ["gal_id", "ra", "dec", "redshift_true", "redshift_rsd", *pipe.flux_columns, "line_bitmask_base"]
    assert set(expected) <= set(base.columns)
    # step 2 writes line_bitmask itself: the base catalog must not own a column of that name
    assert "redshift_obs" not in base.columns and "line_bitmask" not in base.columns
    d = base.select(*pipe.flux_columns, "redshift_rsd", "line_bitmask_base").get_data("numpy", wrap_single=True)
    flux = np.stack([np.asarray(d[c]) for c in pipe.flux_columns], axis=1)
    wave = pipe.line_set.rest_waves[None, :] * (1.0 + d["redshift_rsd"][:, None])
    inband = (wave > 10000.0) & (wave < 19300.0)
    # every galaxy has a line above the primary limit inside the bandpass, and fainter lines are stored too
    assert ((flux > 1e-16) & inband).any(axis=1).all()
    assert (d["line_bitmask_base"] > 0).all()
    assert (flux < 4e-17).any()
    assert str(base.units[pipe.flux_columns[0]]) in ("erg / (s cm2)", "erg / (cm2 s)")
    from romanmock.io import read_manifest

    m = read_manifest(tmp / "base", "romanmock_base")
    assert m["kind"] == "base" and sorted(int(k) for k in m["slices"]) == sorted(steps)
    # a second call reuses the slices
    assert RomanMockPipeline(_config(tmp, steps)).build_base(verbose=False) == base_files


def test_assign_equals_direct_run(built):
    tmp, steps, pipe, _ = built
    direct = pipe.run(verbose=False)
    cfg2 = _config(tmp, steps, source="base")
    cfg2.output.dir = str(tmp / "assigned")
    pipe2 = RomanMockPipeline(cfg2)
    np.testing.assert_allclose(pipe2.line_set.rest_waves, pipe.line_set.rest_waves)
    assigned = pipe2.run(verbose=False)
    cols = ["ra", "dec", "redshift_true", "redshift_rsd", "redshift_assigned", "id_outcome", "line_bitmask", "n_det",
            "line_true", "line_assigned"]
    a, b = _read(direct, cols), _read(assigned, cols)
    assert len(a["ra"]) == len(b["ra"]) > 0
    for c in cols:
        np.testing.assert_array_equal(a[c], b[c], err_msg=c)
    # stage B and the parent accessors work on the assigned catalog
    assert len(pipe2.open_parent()) == len(a["ra"])
    assert pipe2.config_hash() != pipe.config_hash()


def test_assign_brighter_limit_and_parallel(built):
    tmp, steps, pipe, _ = built
    cfg = _config(tmp, steps, source="base",
                  sensitivity={"type": "constant", "flux_limit": 2e-16, "secondary_flux_limit": 1e-16,
                               "wave_range": [10000.0, 19300.0]})
    cfg.output.dir = str(tmp / "bright")
    cfg.output.store_fluxes = True
    cfg.output.passthrough_columns = ["gal_id", "ra", "dec", "redshift_true"]
    files = RomanMockPipeline(cfg).run(verbose=False)
    cols = ["gal_id", "line_bitmask", "n_det", "redshift_rsd", *pipe.flux_columns]
    d = _read(files, cols)
    flux = np.stack([np.asarray(d[c]) for c in pipe.flux_columns], axis=1)
    wave = pipe.line_set.rest_waves[None, :] * (1.0 + d["redshift_rsd"][:, None])
    inband = (wave > 10000.0) & (wave < 19300.0)
    primary = (flux > 2e-16) & inband
    assert primary.any(axis=1).all()
    expect = primary | (primary.any(axis=1)[:, None] & (flux > 1e-16) & inband)
    np.testing.assert_array_equal(d["n_det"], expect.sum(axis=1))
    # the written bitmask is the recomputed one, not the one of the base catalog
    from romanmock.detection import pack_bitmask, popcount

    np.testing.assert_array_equal(d["line_bitmask"], pack_bitmask(expect))
    np.testing.assert_array_equal(popcount(d["line_bitmask"]), d["n_det"])
    # fewer galaxies than at the base limit
    base_n = sum(v["n_galaxies"] for v in RomanMockPipeline(cfg).base_manifest["slices"].values())
    assert 0 < len(d["gal_id"]) < base_n

    cfg.output.dir = str(tmp / "bright_parallel")
    files2 = RomanMockPipeline(cfg).run(verbose=False, n_workers=2)
    d2 = _read(files2, cols)
    for c in cols:
        np.testing.assert_array_equal(d[c], d2[c], err_msg=c)


def test_assign_rejects_selection_outside_base(built):
    tmp, steps, _, _ = built
    deeper = _config(tmp, steps, source="base",
                     sensitivity={"type": "constant", "flux_limit": 5e-17, "wave_range": [10000.0, 19300.0]})
    deeper.output.dir = str(tmp / "deeper")
    with pytest.raises(ValueError, match="deeper than the base catalog"):
        RomanMockPipeline(deeper)
    wider = _config(tmp, steps, source="base",
                    sensitivity={"type": "constant", "flux_limit": 1e-16, "wave_range": [7500.0, 19300.0]})
    wider.output.dir = str(tmp / "wider")
    with pytest.raises(ValueError, match="bandpass"):
        RomanMockPipeline(wider)
    absent = _config(tmp, steps, source="base")
    absent.output.dir = str(tmp / "absent")
    absent.output.passthrough_columns = ["ra", "dec", "logsm_obs"]
    with pytest.raises(KeyError, match="logsm_obs"):
        RomanMockPipeline(absent)
    # a lower secondary limit is allowed: the base catalog stores every line flux
    low2 = _config(tmp, steps, source="base",
                   sensitivity={"type": "constant", "flux_limit": 1e-16, "secondary_flux_limit": 1e-17,
                                "wave_range": [10000.0, 19300.0]})
    low2.output.dir = str(tmp / "low2")
    RomanMockPipeline(low2)


def test_base_columns_configurable(tmp_path):
    import opencosmo as oc

    steps = _small_steps(1)
    cfg = _config(tmp_path, steps)
    cfg.base.columns = ["gal_id", "ra", "dec", "redshift_true", "logsm_obs"]
    cfg.base.store_bitmask = False
    cfg.base.flux_dtype = "float32"
    files = RomanMockPipeline(cfg).build_base(verbose=False)
    base = oc.open(*files)
    assert "logsm_obs" in base.columns and "line_bitmask_base" not in base.columns
    d = base.select("Ha_flux", "ra").get_data("numpy", wrap_single=True)
    assert d["Ha_flux"].dtype == np.float32
    cfg.base.columns = ["ra", "dec", "line_bitmask"]
    cfg.base.overwrite = True
    with pytest.raises(ValueError, match="written by the pipeline"):
        RomanMockPipeline(cfg).build_base(verbose=False)


def test_spatial_index_is_compressed(built, tmp_path):
    import opencosmo as oc

    from romanmock.io import compress_spatial_index, spatial_index_nbytes, write_dataset

    tmp, steps, pipe, base_files = built
    path = base_files[0]
    assert spatial_index_nbytes(path) < 20e6                       # 268 MB uncompressed
    ds = oc.open(path)
    plain = write_dataset(tmp_path / "plain.hdf5", ds, compress_index=False)
    packed = write_dataset(tmp_path / "packed.hdf5", ds)
    assert spatial_index_nbytes(plain) > 200e6
    assert plain.stat().st_size - packed.stat().st_size > 200e6
    from romanmock.io import read_stamp
    assert read_stamp(path)["romanmock_config_hash"] == pipe.base_hash()      # stamps are written after the repack
    assert sorted(f.name for f in tmp_path.iterdir()) == ["packed.hdf5", "plain.hdf5"]      # no temporary files left
    a, b = oc.open(plain), oc.open(packed)
    assert sorted(a.columns) == sorted(b.columns) and len(a) == len(b)
    da, db = a.get_data("numpy"), b.get_data("numpy")
    for c in a.columns:
        np.testing.assert_array_equal(da[c], db[c], err_msg=c)
        assert a.units[c] == b.units[c]
    assert a.descriptions == b.descriptions and a.z_range == b.z_range
    for nside in (64, 1024):
        np.testing.assert_array_equal(np.sort(a.get_pixels(nside)), np.sort(b.get_pixels(nside)))
    with pytest.raises(FileExistsError):
        write_dataset(packed, ds)
    # conversion in place of an existing file, idempotent
    from romanmock.io import compress_index_in_place, spatial_index_is_compressed

    assert not spatial_index_is_compressed(plain) and spatial_index_is_compressed(packed)
    _, before, after, converted = compress_index_in_place(plain)
    assert converted and before - after > 200e6 and spatial_index_is_compressed(plain)
    assert compress_index_in_place(plain)[3] is False
    c = oc.open(plain).get_data("numpy")
    for col in a.columns:
        np.testing.assert_array_equal(c[col], db[col], err_msg=col)
    # the option can be switched off per catalog
    cfg = _config(tmp_path, steps[:1])
    cfg.base.compress_index = False
    files = RomanMockPipeline(cfg).build_base(verbose=False)
    assert spatial_index_nbytes(files[0]) > 200e6

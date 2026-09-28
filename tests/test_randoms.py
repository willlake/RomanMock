"""Footprint of the lightcone and random catalogs."""
import numpy as np
import pytest

from romanmock import BaseCatalog, Catalog, PatchFootprint, RomanMockConfig, RomanMockPipeline, make_random
from romanmock.footprint import read_patch_decomposition
from romanmock.randoms import random_files, sample_sky

from conftest import CATALOG_DIR, catalog_available


def _footprint():
    # two rings; the second one covers part of the circle only
    theta = np.radians([[20.0, 30.0]] * 4 + [[30.0, 40.0]] * 2)
    phi = np.radians([[0, 90], [90, 180], [180, 270], [270, 360], [0, 60], [60, 120]])
    return PatchFootprint(theta[:, 0], theta[:, 1], phi[:, 0], phi[:, 1], ra_offset=180.0)


def test_footprint_area_and_sampling(tmp_path):
    fp = _footprint()
    ring = lambda lo, hi, dphi: np.radians(dphi) * (np.cos(np.radians(lo)) - np.cos(np.radians(hi))) * np.degrees(1) ** 2
    assert fp.area_deg2 == pytest.approx(ring(20, 30, 360) + ring(30, 40, 120))
    ra, dec = fp.sample(400_000, np.random.default_rng(1))
    assert fp.contains(ra, dec).all()
    assert dec.min() >= 50 and dec.max() <= 70 and ra.min() >= 0 and ra.max() < 360
    # uniform on the sky: the share of each ring is its share of the area
    assert np.mean(dec > 60) == pytest.approx(ring(20, 30, 360) / fp.area_deg2, abs=0.003)
    low = dec < 60                                        # phi 0-120 <-> ra 180-300
    assert ra[low].min() >= 180 and ra[low].max() <= 300
    # uniform in sin(dec) inside a ring
    s = np.sin(np.radians(dec[~low]))
    h = np.histogram(s, bins=10, range=(np.sin(np.radians(60)), np.sin(np.radians(70))))[0]
    assert np.abs(h / h.mean() - 1).max() < 0.03
    # points outside
    assert not fp.contains([10.0, 200.0, 200.0], [55.0, 45.0, 75.0]).any()
    assert fp.contains([200.0, 10.0], [55.0, 65.0]).all()
    grid = fp.contains(np.array([200.0, 10.0])[:, None], np.array([55.0, 65.0, 45.0])[None, :])      # broadcast
    assert grid.shape == (2, 3) and grid.tolist() == [[True, True, False], [False, True, False]]
    assert fp.rectangles().shape == (6, 4)
    assert bool(fp.contains(200.0, 55.0))
    fp2 = PatchFootprint.load(fp.save(tmp_path / "fp.json"))
    assert fp2.area_deg2 == fp.area_deg2 and fp2.n_patches == 6


def test_sample_sky_with_cuts_and_file_names(tmp_path):
    fp = _footprint()
    ra, dec = sample_sky(fp, 50_000, np.random.default_rng(2), {"dec": (52.0, 58.0), "ra": (None, 250.0)})
    assert ra.size == 50_000 and dec.min() >= 52 and dec.max() <= 58 and ra.max() <= 250 and ra.min() >= 180
    with pytest.raises(ValueError):
        sample_sky(fp, 10, np.random.default_rng(2), {"dec": (80.0, 85.0)})
    assert [p.name for p in random_files(tmp_path / "ha.hdf5", "all", 2, False)] == ["ha_00.hdf5", "ha_01.hdf5"]
    assert [p.name for p in random_files(tmp_path / "ha.hdf5", "galaxies", 1, True)] == ["ha_galaxies_00.hdf5"]
    assert [p.name for p in random_files(tmp_path, "all", 1, False)] == ["random_00.hdf5"]


def test_random_catalogs_without_base(tmp_path):
    fp = _footprint()
    rng = np.random.default_rng(3)
    ra, dec = fp.sample(30_000, rng)
    outcome = rng.choice([1, 2, 3], size=ra.size, p=[0.5, 0.3, 0.2]).astype(np.int8)
    z = np.where(outcome == 3, rng.uniform(1.0, 1.2, ra.size), rng.uniform(1.0, 1.9, ra.size))
    cat = Catalog({"ra": ra, "dec": dec, "redshift_assigned": z, "id_outcome": outcome}, {"z_column": "redshift_assigned"})
    assert len(cat.subset("galaxies")) + len(cat.subset("interlopers")) == len(cat.subset("all")) == len(cat)
    ran = cat.make_random(which=["all", "galaxies", "interlopers"], n_times=3, footprint=fp, seed=4, verbose=False)
    for kind, n in (("all", len(cat)), ("galaxies", np.sum(outcome != 3)), ("interlopers", np.sum(outcome == 3))):
        assert len(ran[kind]) == 3
        for r in ran[kind]:
            assert len(r) == n and r.columns == ["ra", "dec", "redshift_assigned"] and fp.contains(r["ra"], r["dec"]).all()
            assert np.isin(r["redshift_assigned"], cat.subset(kind)["redshift_assigned"]).all()     # drawn from the data
        assert not np.array_equal(ran[kind][0]["ra"], ran[kind][1]["ra"])
    assert ran["interlopers"][0]["redshift_assigned"].max() <= 1.2
    # each catalog can be made again on its own; files hold the same
    files = cat.make_random(which="galaxies", n_times=2, first=1, footprint=fp, seed=4, write_to=tmp_path / "gal.hdf5", verbose=False)
    assert [f.name for f in files] == ["gal_01.hdf5", "gal_02.hdf5"]
    back = Catalog.read(files[0])
    np.testing.assert_array_equal(back["ra"], ran["galaxies"][1]["ra"])
    np.testing.assert_array_equal(back["redshift_assigned"], ran["galaxies"][1]["redshift_assigned"])
    assert back.settings["random"] == {"which": "galaxies", "index": 1, "seed": 4, "n_data": int(np.sum(outcome != 3))}
    with pytest.raises(KeyError):
        cat[["ra", "dec", "redshift_assigned"]].make_random(which="interlopers", footprint=fp, verbose=False)
    with pytest.raises(ValueError):
        cat.make_random(which="stars", footprint=fp, verbose=False)
    with pytest.raises(ValueError):
        cat.make_random(verbose=False)                   # no base catalog and no footprint


@pytest.mark.skipif(not catalog_available(), reason="diffsky mock not available")
def test_footprint_and_randoms_of_the_mock(tmp_path):
    from pathlib import Path

    from romanmock.io import list_catalog_files, step_from_filename

    table = read_patch_decomposition(Path(CATALOG_DIR) / "lc_cores-decomposition.txt")
    assert table.shape[1] == 5 and np.sum(4 * np.pi * 0 + (table[:, 4] - table[:, 3]) * (np.cos(table[:, 1]) - np.cos(table[:, 2]))) == pytest.approx(4 * np.pi, rel=1e-5)
    files = sorted(list_catalog_files(CATALOG_DIR), key=lambda p: p.stat().st_size)
    steps = [step_from_filename(p) for p in files[10:12]]
    cfg = RomanMockConfig.from_dict({"catalog": {"path": CATALOG_DIR, "steps": steps}, "base": {"dir": str(tmp_path / "base")}})
    RomanMockPipeline(cfg).build_base(verbose=False)
    base = BaseCatalog(tmp_path / "base")
    fp = base.footprint
    assert fp.n_patches == 60 and fp.ra_offset == 180.0 and fp.area_deg2 == pytest.approx(2062.65, abs=0.05)
    assert fp.area_deg2 == pytest.approx(base.area_deg2, rel=2e-2)             # the HEALPix estimate, two sparse slices
    assert (tmp_path / "base" / "romanmock_base_footprint.json").exists()
    cat = base.make_catalog(confusion={"SIII_9532": {"Ha": 0.3}}, n_workers=1, verbose=False)
    assert np.mean(fp.contains(cat["ra"], cat["dec"])) > 0.995                   # satellites spill over the edge
    ran = cat.make_random(n_times=2, verbose=False)
    assert len(ran) == 2 and len(ran[0]) == len(cat) and fp.contains(ran[0]["ra"], ran[0]["dec"]).all()
    assert ran[0].area_deg2 == pytest.approx(fp.area_deg2)
    # data and randoms have the same large-scale distribution on the sky
    h = lambda c: np.histogram2d(c["ra"], np.sin(np.radians(c["dec"])), bins=(6, 3), range=((0, 360), (np.sin(np.radians(57.85)), np.sin(np.radians(75.16)))))[0]
    d, r = h(cat), h(ran[0])
    assert np.array_equal(d == 0, r == 0) or np.all(d[r == 0] < 1e-3 * len(cat))
    assert np.abs(d[r > 0] / r[r > 0] - 1).max() < 0.25
    files = make_random(cat.write(tmp_path / "cat.hdf5"), which=["galaxies", "interlopers"], n_times=1,
                        write_to=tmp_path / "ran" / "cat_random.hdf5", verbose=False)
    assert files["interlopers"][0].name == "cat_random_interlopers_00.hdf5"
    assert len(Catalog.read(files["interlopers"][0])) == len(cat.subset("interlopers")) > 0

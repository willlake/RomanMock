"""Footprint of the lightcone and random catalogs."""
import numpy as np
import pytest

from romanmock import BaseCatalog, Catalog, PatchFootprint, RomanMockConfig, RomanMockPipeline, make_random
from romanmock.footprint import read_patch_decomposition
from romanmock.randoms import data_key, sample_sky

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
    # check against galaxies: a footprint that is too small or too large is noticed
    good = fp.check(ra, dec)
    assert good["inside"] == 1.0 and good["empty"] == 0.0 and good["n_pixels"] > 50
    small = PatchFootprint(fp.theta_lo[:4], fp.theta_hi[:4], fp.phi_lo[:4], fp.phi_hi[:4])
    assert small.check(ra, dec)["inside"] < 0.9
    part = dec > 60
    assert fp.check(ra[part], dec[part])["empty"] > 0.1
    fp2 = PatchFootprint.load(fp.save(tmp_path / "fp.json"))
    assert fp2.area_deg2 == fp.area_deg2 and fp2.n_patches == 6


def test_sample_sky_with_cuts():
    fp = _footprint()
    ra, dec = sample_sky(fp, 50_000, np.random.default_rng(2), {"dec": (52.0, 58.0), "ra": (None, 250.0)})
    assert ra.size == 50_000 and dec.min() >= 52 and dec.max() <= 58 and ra.max() <= 250 and ra.min() >= 180
    with pytest.raises(ValueError):
        sample_sky(fp, 10, np.random.default_rng(2), {"dec": (80.0, 85.0)})


def _contaminated(n=20_000, seed=5):
    """A toy contaminated catalog on the toy footprint: multi-line galaxies, single-line ones with a correct redshift,
    and two kinds of interlopers taken for Ha ([O III] 5007 and Hb), each kind at its own redshifts."""
    fp = _footprint()
    rng = np.random.default_rng(seed)
    ra, dec = fp.sample(n, rng)
    kind = rng.choice(4, size=n, p=[0.6, 0.2, 0.15, 0.05])          # multi, single correct, OIII as Ha, Hb as Ha
    data = {"ra": ra, "dec": dec, "redshift_assigned": np.array([1.0, 1.05, 1.1, 1.15])[kind] + rng.uniform(0, 0.05, n),
            "id_outcome": np.array([1, 2, 3, 3], dtype=np.int8)[kind], "line_true": np.array([0, 0, 2, 1], dtype=np.int8)[kind],
            "line_assigned": np.zeros(n, dtype=np.int8)}
    return Catalog(data, {"z_column": "redshift_assigned"}, {}, ["Ha", "Hb", "OIII_5007"]), kind, fp


def test_random_catalog_of_a_data_catalog(tmp_path):
    """The random catalog depends on the data and the seed only: n_times as many points, uniform inside the footprint,
    redshifts drawn from the data; the same data give the same randoms, more blocks keep the first ones, other data
    (another part of the catalog) give independent ones, and the file written block by block holds the same."""
    cat, kind, fp = _contaminated()
    n = len(cat)
    ran = cat.make_random(3, footprint=fp, seed=4, verbose=False)
    assert len(ran) == 3 * n and ran.columns == ["ra", "dec", "redshift_assigned"] and fp.contains(ran["ra"], ran["dec"]).all()
    assert np.isin(ran["redshift_assigned"], cat["redshift_assigned"]).all()                 # drawn from the data
    assert ran.settings["random"] == {"n_times": 3, "n_data": n, "seed": 4, "data_key": data_key(cat["redshift_assigned"])}
    assert ran.area_deg2 == pytest.approx(fp.area_deg2) and "Random catalog · 3 × 20,000 points" in ran.summary()
    again = make_random(cat, n_times=5, footprint=fp, seed=4, verbose=False)
    for c in ran.columns:                                      # the same data: the same randoms, more blocks keep the first
        np.testing.assert_array_equal(again[c][:3 * n], ran[c], err_msg=c)
    assert not np.array_equal(cat.make_random(1, footprint=fp, seed=5, verbose=False)["ra"], ran["ra"][:n])
    path = cat.make_random(3, footprint=fp, seed=4, write_to=tmp_path / "ran.hdf5", verbose=False)
    back = Catalog.read(path)
    assert path.name == "ran.hdf5" and back.settings["random"] == ran.settings["random"]
    for c in ran.columns:
        np.testing.assert_array_equal(back[c], ran[c], err_msg=c)
    assert Catalog.read(cat.make_random(1, footprint=fp, seed=4, write_to=tmp_path, verbose=False)).columns == ran.columns
    # a part of the catalog is a data catalog of its own: its size, its redshifts, its own random points
    for which, k, lo in (("multi_line", 0, 1.0), ("single_correct", 1, 1.05), ("OIII_5007_as_Ha", 2, 1.1), ("Hb_as_Ha", 3, 1.15)):
        part = cat.subset(which)
        r = part.make_random(2, footprint=fp, seed=4, verbose=False)
        assert len(r) == 2 * np.sum(kind == k) and fp.contains(r["ra"], r["dec"]).all()
        assert r["redshift_assigned"].min() >= lo and r["redshift_assigned"].max() <= lo + 0.05         # its own n(z)
        assert not np.array_equal(r["ra"][:100], ran["ra"][:100])        # independent of the whole catalog's randoms
    inter = cat.subset("interlopers").make_random(1, footprint=fp, seed=4, verbose=False)
    assert inter["redshift_assigned"].min() >= 1.1 and len(inter) == np.sum(kind >= 2)
    low = cat[cat["redshift_assigned"] < 1.05].make_random(1, footprint=fp, seed=4, verbose=False)          # any selection
    assert len(low) == np.sum(cat["redshift_assigned"] < 1.05) and low["redshift_assigned"].max() < 1.05
    with pytest.raises(KeyError):
        cat[["ra", "dec"]].make_random(footprint=fp, verbose=False)
    with pytest.raises(ValueError, match="footprint"):
        cat.make_random(verbose=False)                   # no base catalog and no footprint
    with pytest.raises(ValueError, match="nothing to make randoms"):
        cat[cat["redshift_assigned"] > 9].make_random(footprint=fp, verbose=False)


def test_parts_of_a_catalog():
    """The parts of a contaminated catalog, with one entry per kind of interloper."""
    cat, kind, _ = _contaminated()
    comp = cat.components()
    counts = np.bincount(kind, minlength=4)
    assert list(comp) == ["all", "multi_line", "single_correct", "galaxies", "interlopers", "OIII_5007_as_Ha", "Hb_as_Ha"]
    assert [comp[k] for k in ("multi_line", "single_correct", "OIII_5007_as_Ha", "Hb_as_Ha")] == counts.tolist()
    assert comp["galaxies"] == comp["multi_line"] + comp["single_correct"] and comp["all"] == len(cat)
    assert comp["interlopers"] == comp["OIII_5007_as_Ha"] + comp["Hb_as_Ha"] == len(cat.subset("interlopers"))
    assert len(cat.subset("Hb_as_Ha")) == counts[3] and (cat.subset("Hb_as_Ha")["line_true"] == 1).all()
    with pytest.raises(ValueError, match="unknown lines"):
        cat.subset("SIII_9532_as_Ha")
    with pytest.raises(ValueError, match="itself"):
        cat.subset("Ha_as_Ha")
    with pytest.raises(ValueError, match="which must be"):
        cat.subset("stars")
    with pytest.raises(KeyError, match="line_true"):
        cat[["ra", "dec", "redshift_assigned", "id_outcome"]].subset("Hb_as_Ha")


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
    assert base.area_deg2 == fp.area_deg2
    assert (tmp_path / "base" / "romanmock_base_footprint.json").exists()
    import json

    cache = tmp_path / "base" / "romanmock_base_footprint.json"
    stored = json.loads(cache.read_text())
    assert stored["check"]["inside"] > 0.99 and stored["check"]["empty"] <= 0.01 and stored["base"] == base._identity()
    # a cache that belongs to another base catalog (rebuilt, other mock) is not used
    cache.write_text(json.dumps({**_footprint().to_dict(), "base": {**stored["base"], "n_galaxies": 1}}))
    assert BaseCatalog(tmp_path / "base").footprint.n_patches == 60
    assert json.loads(cache.read_text())["base"] == base._identity()
    cat = base.make_catalog(confusion={"SIII_9532": {"Ha": 0.3}}, n_workers=1, verbose=False)
    assert np.mean(fp.contains(cat["ra"], cat["dec"])) > 0.995                   # satellites spill over the edge
    ran = cat.make_random(n_times=2, verbose=False)
    assert len(ran) == 2 * len(cat) and fp.contains(ran["ra"], ran["dec"]).all()
    assert ran.area_deg2 == pytest.approx(fp.area_deg2)
    # data and randoms have the same large-scale distribution on the sky
    h = lambda c: np.histogram2d(c["ra"], np.sin(np.radians(c["dec"])), bins=(6, 3), range=((0, 360), (np.sin(np.radians(57.85)), np.sin(np.radians(75.16)))))[0]
    d, r = h(cat), h(ran) / 2
    assert np.array_equal(d == 0, r == 0) or np.all(d[r == 0] < 1e-3 * len(cat))
    assert np.abs(d[r > 0] / r[r > 0] - 1).max() < 0.25
    path = make_random(cat.write(tmp_path / "cat.hdf5"), n_times=1, write_to=tmp_path / "ran" / "cat_random.hdf5", verbose=False)
    assert path.name == "cat_random.hdf5" and len(Catalog.read(path)) == len(cat)
    inter = cat.subset("interlopers")
    assert len(inter.make_random(n_times=1, verbose=False)) == len(inter) > 0

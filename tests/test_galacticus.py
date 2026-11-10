import json
import os

import h5py
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from romanmock.fluxdist import FluxDistribution
from romanmock.fluxes import feature_fluxes
from romanmock.galacticus import (
    COMPONENTS,
    GALACTICUS_LINES,
    GalacticusMock,
    air_to_vacuum,
    chunk_index,
    file_info,
    read_file,
    resolve_column,
)
from romanmock.lines import GALACTICUS_LINE_WAVELENGTHS, GALACTICUS_TRANSITIONS, LineSet, get_line_set

GALACTICUS_DIR = os.environ.get(
    "ROMANMOCK_TEST_GALACTICUS",
    "/global/cfs/cdirs/m4943/Galacticus/mockCatalogs/GP-MCMC-Calibration1_allUNIT_16sqDegMock_XMM-LSS",
)
Z_EDGES = np.array([0.0, 0.5, 1.0, 4.0])
LOGF_EDGES = np.round(np.arange(-21.0, -11.0 + 1e-9, 0.02), 6)


def galacticus_available():
    return os.path.isdir(os.path.join(GALACTICUS_DIR, "galaxyCatalogs"))


def _write_file(path, rng, n, dust_factor=0.5):
    """A small file with the layout of the Galacticus mocks (two components dust-free, dust = half)."""
    z = rng.uniform(0.1, 3.5, n)
    with h5py.File(path, "w") as f:
        f.create_group("Parameters/cosmologyParameters").attrs.update(
            {"HubbleConstant": 67.74, "OmegaMatter": 0.3089, "OmegaBaryon": 0.0462, "OmegaDarkEnergy": 0.6911, "temperatureCMB": 2.72548})
        f.create_group("Parameters/geometryLightcone").attrs.update({"angularSize": 4.0})
        node = f.create_group("Lightcone/Output1/nodeData")
        dust = f.create_group("Lightcone/Output1/dustAttenuatedNodeData")
        dust.attrs["dust_model"] = "test"
        dust.attrs["emission_line_dust"] = json.dumps({"model": "test"})
        node["lightconeRedshiftCosmological"] = z
        node["lightconeRedshiftObserved"] = z + 1e-3 * rng.standard_normal(n)
        node["rightAscension"] = rng.uniform(33, 38, n)
        node["declination"] = rng.uniform(-7, -2.5, n)
        node["nodeIndex"] = np.arange(n, dtype=np.int64)
        node["nodeIsIsolated"] = rng.integers(0, 2, n)
        node["diskMassStellar"] = 10 ** rng.uniform(7, 11, n)
        for line in GALACTICUS_LINES:
            for comp in COMPONENTS:
                lum = 10 ** rng.normal(40.5, 0.7, n) if comp != "AGN" else np.zeros(n)
                ds = node.create_dataset(f"luminosityEmissionLine{comp}:{line}", data=lum)
                ds.attrs["wavelength"] = 6562.818 if line == "balmerAlpha6565" else 5006.852
                dust.create_dataset(f"luminosityEmissionLine{comp}:{line}", data=dust_factor * lum)
    return z


@pytest.fixture
def small_mock(tmp_path):
    rng = np.random.default_rng(7)
    cat = tmp_path / "mock" / "galaxyCatalogs"
    cat.mkdir(parents=True)
    zs = [_write_file(cat / f"romanUNIT-d{1 + 25 * i}_{25 * (i + 1)}_16sqDeg.hdf5", rng, n) for i, n in enumerate((3000, 2000, 2500))]
    return tmp_path / "mock", zs


def test_wavelengths_and_preset():
    assert air_to_vacuum(6562.818) == pytest.approx(6564.63, abs=0.01)
    assert GALACTICUS_LINE_WAVELENGTHS["balmerAlpha6565"] == pytest.approx(air_to_vacuum(6562.818487568687), abs=1e-3)
    assert GALACTICUS_LINE_WAVELENGTHS["oxygenIII5008"] == pytest.approx(air_to_vacuum(5006.851692569257), abs=1e-3)
    ls = get_line_set("galacticus_transitions")
    assert ls is GALACTICUS_TRANSITIONS and len(ls) == 16 and ls.names[:7] == ["Ha", "Hb", "OIII_5007", "OIII_4959", "OII_3727", "SIII_9532", "SIII_9069"]
    assert set(ls.required_columns) == set(GALACTICUS_LINES)
    assert ls["OII_3727"].is_blend and ls.component_wave("oxygenII3730") == GALACTICUS_LINE_WAVELENGTHS["oxygenII3730"]
    # round trip through the config (what the workers receive)
    assert LineSet.from_config(ls.to_config()).rest_waves.tolist() == ls.rest_waves.tolist()


def test_resolve_column():
    assert resolve_column("redshift_true") == ["nodeData/lightconeRedshiftCosmological"]
    assert resolve_column("balmerAlpha6565") == [f"dustAttenuatedNodeData/luminosityEmissionLine{c}:balmerAlpha6565" for c in COMPONENTS]
    assert resolve_column("balmerAlpha6565", dust=False)[0] == "nodeData/luminosityEmissionLineDisk:balmerAlpha6565"
    assert resolve_column("luminosityEmissionLineAGN:sulfurIII9533") == ["dustAttenuatedNodeData/luminosityEmissionLineAGN:sulfurIII9533"]
    assert resolve_column("diskMassStellar") == ["nodeData/diskMassStellar"]
    assert chunk_index("romanUNIT-d1026_1050_16sqDeg.hdf5") == 1026


def test_read_and_fluxes(small_mock):
    root, zs = small_mock
    mock = GalacticusMock(root)
    assert len(mock.files) == 3 and [chunk_index(p) for p in mock.files] == [1, 26, 51]
    assert mock.area_deg2 == 16.0 and mock.cosmology.H0.value == 67.74 and mock.cosmology.Om0 == 0.3089
    assert mock.n_gal == 7500 and set(mock.lines) == set(GALACTICUS_LINES)
    data = mock.read(["redshift_true", "redshift_rsd", "ra", "balmerAlpha6565", "gal_id"])
    np.testing.assert_array_equal(data["redshift_true"], np.concatenate(zs))
    assert data["gal_id"].dtype == np.int64 and data["balmerAlpha6565"].dtype == np.float64
    # the line is the sum of the components of the chosen group
    one = read_file(mock.files[0], ["balmerAlpha6565", *[f"luminosityEmissionLine{c}:balmerAlpha6565" for c in COMPONENTS]])
    np.testing.assert_allclose(one["balmerAlpha6565"], sum(one[f"luminosityEmissionLine{c}:balmerAlpha6565"] for c in COMPONENTS))
    nodust = read_file(mock.files[0], ["balmerAlpha6565"], dust=False)
    np.testing.assert_allclose(one["balmerAlpha6565"], 0.5 * nodust["balmerAlpha6565"])
    # redshift cut on the observed redshift, both ends
    cut = mock.read(["redshift_rsd", "redshift_true"], z_range=(1.0, 2.0), z_column="redshift_rsd")
    assert cut["redshift_rsd"].size > 0 and (cut["redshift_rsd"] >= 1.0).all() and (cut["redshift_rsd"] < 2.0).all()
    open_end = mock.read(["redshift_true"], z_range=(2.0, None))
    assert open_end["redshift_true"].size == np.sum(np.concatenate(zs) >= 2.0)
    # fluxes agree with the direct conversion
    fl = mock.read_fluxes(columns=["redshift_true", "ra"], z_range=(0.5, 3.0))
    raw = mock.read(["redshift_true", *GALACTICUS_TRANSITIONS.required_columns], z_range=(0.5, 3.0))
    direct = feature_fluxes(raw, GALACTICUS_TRANSITIONS, raw["redshift_true"], mock.cosmology)
    np.testing.assert_allclose(fl["Ha_flux"], direct[:, 0])
    np.testing.assert_allclose(fl["OII_3727_flux"], direct[:, GALACTICUS_TRANSITIONS.bit("OII_3727")])
    assert fl["ra"].size == raw["redshift_true"].size
    # streaming gives the same rows file by file
    streamed = np.concatenate([d["redshift_true"] for _, d in mock.iter_files(["redshift_true"])])
    np.testing.assert_array_equal(streamed, data["redshift_true"])
    with pytest.raises(KeyError):
        mock.read(["noSuchColumn"])


def test_flux_distribution_matches_direct_and_saves(small_mock, tmp_path):
    root, zs = small_mock
    mock = GalacticusMock(root)
    dist = mock.flux_distribution(Z_EDGES, LOGF_EDGES, verbose=False)
    assert dist.steps == [1, 26, 51] and dist.n_gal.sum() == 7500 and dist.line_names == GALACTICUS_TRANSITIONS.names
    fl = mock.read_fluxes()
    z = fl["redshift_true"]
    for name in ("Ha", "OII_3727"):
        f = fl[f"{name}_flux"]
        for j in range(Z_EDGES.size - 1):
            sel = (z >= Z_EDGES[j]) & (z < Z_EDGES[j + 1])
            assert dist.n_gal[j] == sel.sum()
            assert abs(np.log10(dist.quantile(name, 0.5)[j]) - np.log10(np.median(f[sel]))) < 0.03
            assert dist.fraction_above(name, 1e-17)[j] == pytest.approx(np.mean(f[sel] > 1e-17), abs=0.01)
    again = FluxDistribution.load(dist.save(tmp_path / "fd.npz"))
    np.testing.assert_array_equal(again.counts, dist.counts)
    assert again.steps == dist.steps
    # the dust-free variant is brighter by the factor of the test file
    nodust = GalacticusMock(root, dust=False).flux_distribution(Z_EDGES, LOGF_EDGES, verbose=False)
    assert np.log10(nodust.quantile("Ha", 0.5)[1] / dist.quantile("Ha", 0.5)[1]) == pytest.approx(np.log10(2.0), abs=0.03)


def test_line_fractions_match_direct(small_mock, tmp_path):
    from romanmock.confusion import LineFractionTable
    from romanmock.detection import detect_lines
    from romanmock.redshift import single_lines
    from romanmock.sensitivity import ConstantFluxLimit, ThresholdDetection

    root, _ = small_mock
    mock = GalacticusMock(root)
    kw = dict(wave_bin=1000.0, galaxies_per_bin=20, min_count=1)
    table = mock.line_fractions(2e-17, 1e-17, verbose=False, write_to=tmp_path / "lf.npz", **kw)
    fl = mock.read_fluxes()
    flux = np.stack([fl[f"{n}_flux"] for n in GALACTICUS_TRANSITIONS.names], axis=1)
    det = detect_lines(flux, fl["redshift_rsd"], GALACTICUS_TRANSITIONS, ThresholdDetection(ConstantFluxLimit(2e-17, secondary_flux_limit=1e-17)))
    wave, f, line = single_lines(det)
    direct = LineFractionTable.from_single_lines(GALACTICUS_TRANSITIONS, wave, f, line, table.wave_edges, 20, min_count=1)
    assert table.n_single == wave.size > 100 and table.settings["n_base"] == 7500 and table.settings["flux_limit"] == 2e-17
    np.testing.assert_array_equal(table.counts, direct.counts)
    np.testing.assert_allclose(table.flux_edges, direct.flux_edges)
    np.testing.assert_array_equal(LineFractionTable.load(tmp_path / "lf.npz").counts, table.counts)
    # the single lines themselves, with columns of their galaxies
    single = mock.single_lines(2e-17, 1e-17, columns=["redshift_rsd", "diskMassStellar"], n_workers=2, verbose=False)
    rows = np.flatnonzero(det.n_det == 1)
    np.testing.assert_array_equal(single["line"], line)
    np.testing.assert_allclose(single["wave"], wave)
    np.testing.assert_allclose(single["flux"], f)
    np.testing.assert_array_equal(single["redshift_rsd"], fl["redshift_rsd"][rows])
    np.testing.assert_array_equal(single["diskMassStellar"], mock.read(["diskMassStellar"])["diskMassStellar"][rows])
    # the same with two workers, and the S/N route needs a fixed size
    np.testing.assert_array_equal(mock.line_fractions(2e-17, 1e-17, n_workers=2, verbose=False, **kw).counts, table.counts)
    with pytest.raises(ValueError, match="fixed size"):
        mock.line_fractions(snr=5, size="r50_disk_2d", verbose=False)
    # cuts on any column remove galaxies before the detection
    cut = mock.line_fractions(2e-17, 1e-17, cuts={"diskMassStellar": (1e9, None)}, verbose=False, **kw)
    big = mock.read(["diskMassStellar"])["diskMassStellar"] >= 1e9
    det_cut = detect_lines(flux[big], fl["redshift_rsd"][big], GALACTICUS_TRANSITIONS,
                           ThresholdDetection(ConstantFluxLimit(2e-17, secondary_flux_limit=1e-17)))
    assert cut.n_single == np.sum(det_cut.n_det == 1) and cut.settings["n_cut"] == np.sum(~big) > 0


@pytest.mark.skipif(not galacticus_available(), reason="Galacticus mock not available")
def test_real_mock_serial_equals_parallel():
    mock = GalacticusMock(GALACTICUS_DIR)
    assert len(mock.files) >= 2 and mock.area_deg2 == 16.0
    assert mock.rest_wavelengths()["balmerAlpha6565"] == pytest.approx(GALACTICUS_LINE_WAVELENGTHS["balmerAlpha6565"], abs=1e-3)
    files = mock.files[:2]
    a = mock.flux_distribution(Z_EDGES, LOGF_EDGES, files=files, n_workers=1, verbose=False)
    b = mock.flux_distribution(Z_EDGES, LOGF_EDGES, files=files, n_workers=2, verbose=False)
    np.testing.assert_array_equal(a.counts, b.counts)
    assert a.steps == b.steps == sorted(chunk_index(p) for p in files) and a.n_gal.sum() > 0
    one = mock.read(["redshift_true", "balmerAlpha6565", "is_central"], files=files[:1])
    assert one["redshift_true"].size == file_info(files[0]).n_gal
    assert a.n_gal.sum() == np.sum(mock.read(["redshift_true"], files=files)["redshift_true"] < Z_EDGES[-1])
    assert (one["balmerAlpha6565"] >= 0).all() and set(np.unique(one["is_central"])) <= {0, 1}

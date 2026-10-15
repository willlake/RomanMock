import numpy as np
import pytest

from romanmock import (
    FAIL,
    ConfusionMatrix,
    ConstantFluxLimit,
    IdOutcome,
    ThresholdDetection,
    WavelengthBinnedConfusion,
    assign_redshifts,
    detect_lines,
    feature_fluxes,
)
from romanmock.confusion import build_confusion
from romanmock.diagnostics import empirical_confusion
from romanmock.lines import ROMAN_TRANSITIONS


def test_identity_and_row_normalisation():
    cm = ConfusionMatrix.identity(ROMAN_TRANSITIONS, p_fail=0.1)
    assert cm.matrix.shape == (9, 10)
    np.testing.assert_allclose(cm.matrix.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        ConfusionMatrix(ROMAN_TRANSITIONS.names, np.ones((9, 10)))


def test_from_dict_fills_diagonal():
    cm = ConfusionMatrix.from_dict(ROMAN_TRANSITIONS, {"Ha": {"OIII_5007": 0.2, "fail": 0.1}}, default_p_fail=0.05)
    i, j = ROMAN_TRANSITIONS.bit("Ha"), ROMAN_TRANSITIONS.bit("OIII_5007")
    assert cm.matrix[i, i] == pytest.approx(0.7)
    assert cm.matrix[i, j] == pytest.approx(0.2)
    assert cm.p_fail[i] == pytest.approx(0.1)
    assert cm.matrix[1, 1] == pytest.approx(0.95) and cm.p_fail[1] == pytest.approx(0.05)
    rt = ConfusionMatrix.from_dict(ROMAN_TRANSITIONS, cm.to_dict()["rows"])
    np.testing.assert_allclose(rt.matrix, cm.matrix)


def test_sampling_reproduces_matrix():
    rng = np.random.default_rng(0)
    cm = ConfusionMatrix.from_dict(ROMAN_TRANSITIONS, {"Ha": {"OIII_5007": 0.3, "OII_3727": 0.1, "fail": 0.1}})
    true = np.zeros(200000, dtype=int)
    j = cm.assign(true, None, rng)
    n = len(ROMAN_TRANSITIONS)
    counts = np.bincount(np.where(j == FAIL, n, j), minlength=n + 1) / true.size
    np.testing.assert_allclose(counts, cm.matrix[0], atol=0.005)


def test_wavelength_binned_confusion_selects_matrix():
    rng = np.random.default_rng(0)
    a = ConfusionMatrix.identity(ROMAN_TRANSITIONS, p_fail=0.0)
    b = ConfusionMatrix.identity(ROMAN_TRANSITIONS, p_fail=1.0)
    wb = WavelengthBinnedConfusion([10000.0, 15000.0, 20000.0], [a, b])
    true = np.zeros(10, dtype=int)
    wave = np.array([12000.0] * 5 + [18000.0] * 5)
    j = wb.assign(true, {"wave_obs": wave}, rng)
    assert (j[:5] == 0).all() and (j[5:] == FAIL).all()


def test_build_confusion_from_config():
    cm = build_confusion({"type": "identity", "p_fail": 0.2}, ROMAN_TRANSITIONS)
    assert np.allclose(cm.p_fail, 0.2)
    cm = build_confusion({"type": "matrix", "rows": {"Hb": {"Ha": 0.5}}}, ROMAN_TRANSITIONS)
    assert cm.matrix[1, 0] == 0.5 and cm.matrix[1, 1] == 0.5


def test_assign_redshifts_formula(synthetic, cosmo):
    rng = np.random.default_rng(1)
    z_rsd = synthetic["redshift_obs"]
    flux = feature_fluxes(synthetic, ROMAN_TRANSITIONS, synthetic["redshift_true"], cosmo)
    det = detect_lines(flux, z_rsd, ROMAN_TRANSITIONS, ThresholdDetection(ConstantFluxLimit(1e-16)), rng)
    cm = ConfusionMatrix.from_dict(ROMAN_TRANSITIONS, {"OIII_5007": {"Ha": 0.5, "fail": 0.1}})
    za = assign_redshifts(det, z_rsd, ROMAN_TRANSITIONS, cm, rng)

    multi = det.n_det >= 2
    np.testing.assert_array_equal(za.z_assigned[multi], z_rsd[multi])
    assert (za.outcome[multi] == IdOutcome.MULTI_LINE).all()
    assert (za.outcome[det.n_det == 0] == IdOutcome.NOT_DETECTED).all()
    assert np.isnan(za.z_assigned[det.n_det == 0]).all()

    mis = za.outcome == IdOutcome.SINGLE_MISID
    assert mis.any()
    rest = ROMAN_TRANSITIONS.rest_waves
    expected = rest[za.line_true[mis]] * (1 + z_rsd[mis]) / rest[za.line_assigned[mis]] - 1
    np.testing.assert_allclose(za.z_assigned[mis], expected)
    # [O III] 5007 -> Ha maps (1+z) by 5008.3/6564.7
    assert np.all(za.line_true[mis] == ROMAN_TRANSITIONS.bit("OIII_5007"))
    assert np.all(za.line_assigned[mis] == ROMAN_TRANSITIONS.bit("Ha"))

    failed = za.outcome == IdOutcome.SINGLE_FAILED
    assert failed.any() and np.isnan(za.z_assigned[failed]).all() and (za.line_assigned[failed] == FAIL).all()

    emp = empirical_confusion({"n_det": det.n_det, "line_true": za.line_true, "line_assigned": za.line_assigned}, ROMAN_TRANSITIONS)
    i = ROMAN_TRANSITIONS.bit("OIII_5007")
    if (det.n_det[za.line_true == i] == 1).sum() > 500:
        assert abs(emp[i, 0] - 0.5) < 0.1


def test_multi_line_sigma_z(synthetic, cosmo):
    rng = np.random.default_rng(2)
    z_rsd = synthetic["redshift_obs"]
    flux = feature_fluxes(synthetic, ROMAN_TRANSITIONS, synthetic["redshift_true"], cosmo)
    det = detect_lines(flux, z_rsd, ROMAN_TRANSITIONS, ThresholdDetection(ConstantFluxLimit(1e-16)), rng)
    za = assign_redshifts(det, z_rsd, ROMAN_TRANSITIONS, None, rng, sigma_z=1e-3)
    multi = det.n_det >= 2
    scatter = (za.z_assigned[multi] - z_rsd[multi]) / (1 + z_rsd[multi])
    assert 0.8e-3 < scatter.std() < 1.2e-3


def test_default_confusion_adapts_to_line_set():
    from romanmock.confusion import DEFAULT_CONFUSION_ROWS, default_confusion, default_confusion_rows
    from romanmock.lines import PLAN_SPECIES

    cm = default_confusion(ROMAN_TRANSITIONS)
    i, j = ROMAN_TRANSITIONS.bit("Ha"), ROMAN_TRANSITIONS.bit("OIII_5007")
    assert cm.matrix[i, j] == 0.005 and cm.matrix[i, i] == pytest.approx(0.995)
    assert np.all(cm.p_fail == 0.0)
    rows = default_confusion_rows(PLAN_SPECIES)
    assert rows["OIII"] == {"Ha": 0.05, "fail": 0.0} and rows["SIII"] == {"Ha": 0.05, "fail": 0.0}
    assert rows["Ha"] == {"OIII": 0.005, "fail": 0.0}
    cm2 = default_confusion(PLAN_SPECIES)
    assert cm2.matrix.shape == (5, 6)
    assert build_confusion({"type": "default"}, ROMAN_TRANSITIONS).matrix.tolist() == cm.matrix.tolist()
    assert set(DEFAULT_CONFUSION_ROWS) == {"Ha", "OIII_5007", "SIII_9532", "Hb"}


# ---------------------------------------------------------------------------------
# the empirical identification: LineFractionTable
# ---------------------------------------------------------------------------------
def _single_line_arrays(rng, n=60000):
    """Single lines spread over the bandpass: Ha below 1e-15, [S III] 9532 above, a few [O III] 5007."""
    wave = rng.uniform(10000, 19300, n)
    flux = 10 ** rng.uniform(-16, -14, n)
    line = np.where(flux > 1e-15, 5, 0)
    line[:30] = 2
    return wave, flux, line


def test_line_fraction_table_from_single_lines():
    from romanmock import LineFractionTable

    rng = np.random.default_rng(3)
    wave, flux, line = _single_line_arrays(rng)
    edges = np.linspace(10000, 19300, 11)
    t = LineFractionTable.from_single_lines(ROMAN_TRANSITIONS, wave, flux, line, edges, n_flux_bins=10, min_count=100)
    assert t.counts.shape == (10, 10, 9) and t.n_single == wave.size
    assert t.lines_used == ["Ha", "SIII_9532"]                              # [O III] 5007: 30 galaxies, left out
    per_cell = t.counts.sum(axis=2)
    assert (per_cell.max(axis=1) - per_cell.min(axis=1)).max() <= 1          # equal counts inside a wavelength bin
    np.testing.assert_allclose(t.fraction.sum(axis=2), 1.0)
    assert t.fraction[..., 2].max() == 0.0
    assert np.all(t.fraction[:, 6:, 5] == 1.0) and np.all(t.fraction[:, :4, 0] == 1.0)
    iw, jf = t.cells([9000.0, 15000.0, 25000.0], [1e-14, 1.2e-16, 1e-20])   # 10 % quantile of the flux: 1.58e-16
    assert iw.tolist() == [0, 5, 9] and jf.tolist() == [9, 0, 0]              # outside the edges: the nearest bin
    ctx = {"wave_obs": np.full(100, 15000.0), "flux": np.full(100, 1e-14)}
    assert (t.assign(np.zeros(100, int), ctx, rng) == 5).all()
    ctx["flux"][:] = 2e-16
    assert (t.assign(np.zeros(100, int), ctx, rng) == 0).all()
    assert "SIII_9532" in t.summary() and "OIII_5007 (30)" in t.summary()
    with pytest.raises(ValueError):
        LineFractionTable.from_single_lines(ROMAN_TRANSITIONS, wave, flux, line, edges, flux_quantiles=[0, 0.5, 0.9])
    with pytest.raises(ValueError):
        LineFractionTable.from_single_lines(ROMAN_TRANSITIONS, wave, flux, line, edges, min_count=10 ** 6)
    custom = LineFractionTable.from_single_lines(ROMAN_TRANSITIONS, wave, flux, line, edges, flux_quantiles=[0, 0.5, 0.9, 0.99, 1])
    assert custom.n_flux_bins == 4 and abs(custom.counts[0, -1].sum() / custom.counts[0].sum() - 0.01) < 0.003
    # the default: as many equal-count bins as hold galaxies_per_bin galaxies, per wavelength bin (ragged, inf padding)
    wave_r = 10000 + 9300 * rng.beta(2, 1, wave.size)                        # more galaxies to the red
    ragged = LineFractionTable.from_single_lines(ROMAN_TRANSITIONS, wave_r, flux, line, edges, galaxies_per_bin=200)
    n_per_wave = np.histogram(wave_r, bins=edges)[0]
    np.testing.assert_array_equal(ragged.n_flux_bins_per_wave, np.maximum(n_per_wave // 200, 1))
    assert ragged.n_flux_bins == ragged.n_flux_bins_per_wave.max() and ragged.counts[~ragged.valid_cells].sum() == 0
    cells = ragged.n_per_cell[ragged.valid_cells]
    assert cells.min() >= 200 and cells.max() < 400
    iw, jf = ragged.cells(wave_r, flux)
    assert np.all(jf < ragged.n_flux_bins_per_wave[iw])                       # the padding never counts
    assert np.isinf(ragged.flux_edges[0, 1:-1]).sum() == ragged.n_flux_bins - ragged.n_flux_bins_per_wave[0]
    assert np.isfinite(ragged.flux_edges[:, -1]).all() and ragged.settings["flux_bins"] == {"galaxies_per_bin": 200}
    assert "flux bins" in ragged.summary() and LineFractionTable.from_dict(ragged.to_dict()).content_hash() == ragged.content_hash()


def test_line_fraction_table_draw_max_and_expected_confusion():
    from romanmock import LineFractionTable

    rng = np.random.default_rng(4)
    n = 60000
    wave = rng.uniform(10000, 19300, n)
    flux = 10 ** rng.uniform(-16, -14, n)
    line = rng.choice([0, 5], size=n, p=[0.7, 0.3])                          # 70 % Ha, 30 % [S III] 9532 in every cell
    edges = np.linspace(10000, 19300, 4)
    t = LineFractionTable.from_single_lines(ROMAN_TRANSITIONS, wave, flux, line, edges, n_flux_bins=5, p_fail=0.1)
    m = 200000
    ctx = {"wave_obs": rng.uniform(10000, 19300, m), "flux": 10 ** rng.uniform(-16, -14, m)}
    j = t.assign(np.full(m, 5), ctx, rng)                                    # the true line is not used
    assert abs((j == 0).mean() - 0.63) < 0.01 and abs((j == 5).mean() - 0.27) < 0.01 and abs((j == FAIL).mean() - 0.1) < 0.01
    ec = t.expected_confusion()
    assert isinstance(ec, ConfusionMatrix)
    np.testing.assert_allclose(ec.matrix[[0, 5]][:, [0, 5, 9]], [[0.63, 0.27, 0.1]] * 2, atol=0.01)
    assert ec.matrix[1, 1] == pytest.approx(0.9) and ec.matrix[1, 9] == pytest.approx(0.1)   # no galaxies: identity row
    t.mode = "max"
    j = t.assign(np.zeros(m, int), ctx, rng)
    assert abs((j == 0).mean() - 0.9) < 0.01 and (j == 5).sum() == 0
    np.testing.assert_allclose(t.expected_confusion().matrix[[0, 5]][:, [0, 5, 9]], [[0.9, 0.0, 0.1]] * 2)
    with pytest.raises(ValueError):
        LineFractionTable.from_single_lines(ROMAN_TRANSITIONS, wave, flux, line, edges, mode="other")


def test_line_fraction_table_files_and_config(tmp_path):
    import json

    from romanmock import LineFractionTable
    from romanmock.lines import PLAN_SPECIES

    rng = np.random.default_rng(5)
    wave, flux, line = _single_line_arrays(rng)
    t = LineFractionTable.from_single_lines(ROMAN_TRANSITIONS, wave, flux, line, np.linspace(10000, 19300, 5), n_flux_bins=4,
                                            settings={"flux_limit": 1e-16})
    path = t.save(tmp_path / "fractions")
    assert path.suffix == ".npz" and t.source == str(path)
    back = LineFractionTable.load(path)
    assert back.content_hash() == t.content_hash() and back.settings == t.settings and back.lines_used == t.lines_used
    assert LineFractionTable.from_dict(t.to_dict()).content_hash() == t.content_hash()
    cfg = build_confusion({"type": "empirical", "file": str(path), "mode": "max", "p_fail": 0.05}, ROMAN_TRANSITIONS)
    assert cfg.mode == "max" and cfg.p_fail == 0.05
    assert cfg.to_config()["type"] == "empirical" and cfg.to_config()["file"] == str(path)
    assert build_confusion({"type": "empirical", "table": t.to_dict()}, ROMAN_TRANSITIONS).content_hash() == t.content_hash()
    assert build_confusion("identity", ROMAN_TRANSITIONS).matrix[0, 0] == 1.0
    with pytest.raises(ValueError):
        build_confusion({"type": "empirical"}, ROMAN_TRANSITIONS)                   # no table
    with pytest.raises(ValueError):
        build_confusion({"type": "empirical", "file": str(path)}, PLAN_SPECIES)     # other lines
    assert len(json.dumps(t.to_config())) < 5000                                    # compact: fits in file attributes


def test_assign_redshifts_with_line_fractions(synthetic, cosmo):
    from romanmock import LineFractionTable
    from romanmock.redshift import single_lines

    rng = np.random.default_rng(6)
    z_rsd = synthetic["redshift_obs"]
    flux = feature_fluxes(synthetic, ROMAN_TRANSITIONS, synthetic["redshift_true"], cosmo)
    det = detect_lines(flux, z_rsd, ROMAN_TRANSITIONS, ThresholdDetection(ConstantFluxLimit(1e-16)), rng)
    wave, f, line = single_lines(det)
    assert wave.size == (det.n_det == 1).sum() > 500
    np.testing.assert_allclose(wave, ROMAN_TRANSITIONS.rest_waves[line] * (1 + z_rsd[det.n_det == 1]))
    t = LineFractionTable.from_single_lines(ROMAN_TRANSITIONS, wave, f, line, np.linspace(10000, 19300, 4), n_flux_bins=3, min_count=1)
    za = assign_redshifts(det, z_rsd, ROMAN_TRANSITIONS, t, rng)
    single = det.n_det == 1
    assert set(np.unique(za.outcome[single])) <= {int(IdOutcome.SINGLE_CORRECT), int(IdOutcome.SINGLE_MISID)}
    assert (za.outcome[single] == IdOutcome.SINGLE_MISID).any()
    mis = za.outcome == IdOutcome.SINGLE_MISID
    rest = ROMAN_TRANSITIONS.rest_waves
    np.testing.assert_allclose(za.z_assigned[mis], rest[za.line_true[mis]] * (1 + z_rsd[mis]) / rest[za.line_assigned[mis]] - 1)
    emp = empirical_confusion({"n_det": det.n_det, "line_true": za.line_true, "line_assigned": za.line_assigned}, ROMAN_TRANSITIONS)
    many = np.bincount(line, minlength=9) > 300
    np.testing.assert_allclose(emp[many], t.expected_confusion().matrix[many], atol=0.08)

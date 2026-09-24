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

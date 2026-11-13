"""Tests of romanmock.proposal (array level)."""
import numpy as np
import pytest

from romanmock.detection import detect_lines
from romanmock.lines import get_line_set
from romanmock.confusion import WindowedConfusion, build_confusion
from romanmock.proposal import (SCAN, Proposal, line_hint, primary_snr, proposal_counts, proposal_grid, scan_edges, target_window,
                                target_windows, window_intervals)
from romanmock.sensitivity import ConstantFluxLimit, ThresholdDetection

LS = get_line_set("roman_transitions_paa")
NAMES = LS.names
BAND = (10000.0, 19300.0)
HA, O3 = NAMES.index("Ha"), NAMES.index("OIII_5007")


def test_grid_has_the_range_ends_as_edges():
    edges = proposal_grid((1.0, 1.2), 0.01, 4.06)
    assert edges[0] >= 0 and edges[0] < 0.01 + 1e-9 and edges[-1] >= 4.06
    assert np.isclose(edges, 1.0).any() and np.isclose(edges, 1.2).any()
    np.testing.assert_allclose(np.diff(edges), 0.01)
    odd = proposal_grid((1.0, 1.205), 0.01, 3.0)          # the bins are adjusted so that both ends are edges
    assert np.isclose(odd, 1.0).any() and np.isclose(odd, 1.205).any()


def _galaxies(more=False):
    """Ha single (z 1.1), Ha + [O III] (z 1.1), [O III] single in the Ha window (z 1.75), [O III] single outside it
    (z 2.5), nothing detected (z 1.1); with ``more`` also an Ha single where [O III] 5007 is at 1.0-1.2 (z 0.6) and an
    [O III] single inside the range (z 1.15). None on a bin edge."""
    z = np.array([1.103, 1.103, 1.753, 2.503, 1.103] + ([0.603, 1.153, 0.453] if more else []))
    flux = np.zeros((z.size, len(NAMES)))
    flux[0, HA] = 2e-16
    flux[1, HA], flux[1, O3] = 2e-16, 1e-16
    flux[2, O3] = 2e-16
    flux[3, O3] = 2e-16
    flux[4, HA] = 1e-17
    if more:                     # and an [S III] 9532 single line where Ha is at 1.0-1.2 (z 0.45)
        flux[5, HA] = 2e-16
        flux[6, O3] = 2e-16
        flux[7, NAMES.index("SIII_9532")] = 2e-16
    model = ThresholdDetection(ConstantFluxLimit(1.5e-16, BAND, secondary_flux_limit=6e-17))
    return z, detect_lines(flux, z, LS, model)


def test_counts_of_the_target_and_of_the_potential_interlopers():
    z, det = _galaxies()
    window = target_window(LS.rest_waves, HA, (1.0, 1.2), BAND)
    assert window == pytest.approx((LS.rest_waves[HA] * 2.0, LS.rest_waves[HA] * 2.2))
    edges = proposal_grid((1.0, 1.2), 0.01, 3.0)
    c = proposal_counts(det, z, LS, "Ha", window, edges)
    k = lambda zz: np.searchsorted(edges, zz, side="right") - 1
    assert c["n_det"].sum() == 4 and c["n_det"][k(1.103)] == 2
    assert c["n_single"][k(1.103), HA] == 1 and c["n_single"][k(1.753), O3] == 1 and c["n_single"][k(2.503), O3] == 1
    assert c["n_line_det"][k(1.103), HA] == 2 and c["n_line_det"][k(1.103), O3] == 1
    # the [O III] single line at z 1.75 falls where Ha is at 1.0-1.2; the one at z 2.5 does not (last axis: the target)
    assert c["n_window_single"].shape == c["n_window_det"].shape == c["n_assigned"].shape == (edges.size - 1, len(NAMES), 1)
    assert c["n_window_single"][:, O3, 0].sum() == 1 and c["n_window_single"][k(1.753), O3, 0] == 1
    assert c["n_window_det"][k(1.753), O3, 0] == 1 and c["n_window_det"][k(2.503), O3, 0] == 0
    z_as = LS.rest_waves[O3] * 2.753 / LS.rest_waves[HA] - 1
    assert c["n_assigned"][k(z_as), O3, 0] == 1 and c["n_assigned"][:, O3, 0].sum() == 1
    assert c["n_assigned"][k(1.103), HA, 0] == 1 and c["n_assigned"].sum() == 2    # the target's own single line lands on itself
    assert c["n_window_single"][k(1.103), HA, 0] == 1 and c["n_window_det"][k(1.103), HA, 0] == 2   # the target: its own range
    # a list of targets counts each window on its own: no single line falls where [O III] 5007 is at 1.0-1.2, but the
    # two detected galaxies at z 1.1 have their [O III] there (detected or not)
    both = proposal_counts(det, z, LS, ["Ha", "OIII_5007"], [window, target_window(LS.rest_waves, O3, (1.0, 1.2), BAND)], edges)
    for key in ("n_window_det", "n_window_single", "n_assigned"):
        np.testing.assert_array_equal(both[key][..., :1], c[key], err_msg=key)
    assert both["n_window_single"][..., 1].sum() == 0 and both["n_window_det"][k(1.103), O3, 1] == 2
    iv = window_intervals(NAMES, LS.rest_waves, window)
    assert iv["Ha"] == pytest.approx((1.0, 1.2)) and iv["OIII_5007"][0] < 1.753 < iv["OIII_5007"][1]


def _proposal(min_count=1, min_share=1e-3):
    z, det = _galaxies()
    window = target_window(LS.rest_waves, HA, (1.0, 1.2), BAND)
    edges = proposal_grid((1.0, 1.2), 0.01, 3.0)
    counts = proposal_counts(det, z, LS, "Ha", window, edges)
    settings = {"line": "Ha", "z_range": [1.0, 1.2], "targets": ["Ha"], "windows": [list(window)], "min_count": min_count,
                "min_share": min_share,
                "sensitivity": {"type": "constant", "flux_limit": 1.5e-16, "secondary_flux_limit": 6e-17, "wave_range": list(BAND)},
                "rescaling": None, "doublets": None, "base": "/nowhere"}
    return Proposal(counts, edges, settings, NAMES, LS.rest_waves, area_deg2=2.0)


def test_proposal_numbers_template_and_round_trip(tmp_path):
    p = _proposal()
    assert p.n_sample == 2
    inside = p.single_lines_inside()
    assert inside == {"Ha": {"n": 1, "share": 0.5, "share_single": 1.0, "major": True}}
    inter = p.interlopers()
    assert list(inter) == ["OIII_5007"] and inter["OIII_5007"]["n"] == 1
    assert inter["OIII_5007"]["per_deg2"] == 0.5 and inter["OIII_5007"]["share"] == 1.0
    assert inter["OIII_5007"]["at_most"] == pytest.approx(1 / 3)      # a share of the catalog: 2 + the 1 interloper
    z, f = p.fraction_inside()
    assert np.nanmax(f) == 0.5 and z.min() > 1.0 and z.max() < 1.2
    z, f = p.fraction_interloper("OIII_5007")
    assert f.tolist() == [1.0] and z[0] == pytest.approx(1.755)
    assert p.confusion_template(0.05, 0.01) == {"Ha": {"OIII_5007": 0.01}, "OIII_5007": {"Ha": 0.05}}
    assert p.default_name() == "Ha_z1p0_1p2"
    text = p.summary()
    assert "Potential interlopers" in text and "OIII_5007" in text and text.startswith("━━ Proposal · Hα at 1.0 ≤ z ≤ 1.2")
    assert max(map(len, text.splitlines())) <= 100
    q = Proposal.load(p.save(tmp_path / "p.npz"))
    assert q.settings == p.settings and q.line_names == NAMES
    for key in p.counts:
        np.testing.assert_array_equal(q.counts[key], p.counts[key])
    # a line below the cut is minor: left out of the template, but listed and drawn as thin lines marked minor
    q = _proposal(min_count=2)
    assert q.confusion_template() == {} and q.major_interlopers == [] and q.minor_pairs == [("OIII_5007", "Ha")]
    fig = q.plot()
    thin = [line for ax in fig.axes[1:] for line in ax.get_lines() if line.get_label().endswith("(minor)")]
    assert len(thin) == 2 and all(line.get_linewidth() < 1 for line in thin)
    assert "thin lines in proposal.plot()): OIII_5007 (1)" in " ".join(q.summary().split())      # the note is wrapped
    import matplotlib.pyplot as plt

    plt.close(fig)
    with pytest.raises(ValueError, match="identification"):
        p.make_catalog(None, tmp_path)


def _every_line():
    z, det = _galaxies(more=True)
    windows = target_windows(NAMES, LS.rest_waves, (1.0, 1.2), BAND)
    edges = proposal_grid((1.0, 1.2), 0.01, 3.0)
    counts = proposal_counts(det, z, LS, list(windows), list(windows.values()), edges)
    settings = {"line": "all", "z_range": [1.0, 1.2], "targets": list(windows), "windows": [list(w) for w in windows.values()],
                "min_count": 1, "min_share": 1e-3, "rescaling": None, "doublets": None, "base": "/nowhere",
                "sensitivity": {"type": "constant", "flux_limit": 1.5e-16, "secondary_flux_limit": 6e-17, "wave_range": list(BAND)}}
    return Proposal(counts, edges, settings, NAMES, LS.rest_waves, area_deg2=2.0)


def test_proposal_of_every_line(tmp_path):
    """line="all": the sample is every detected galaxy inside the range; the potential interlopers are pairs, a single
    line taken for a line that single-line galaxies inside the range show (Ha and [O III] 5007 here)."""
    p = _every_line()
    assert p.targets == ["Ha", "Hb", "OIII_5007", "OIII_4959", "SIII_9532", "SIII_9069", "SIII_6312"]   # in the bandpass
    assert p.all_lines and p.n_sample == 3 and p.assignable == ["Ha", "OIII_5007"]
    inter = p.interlopers()
    assert set(inter) == {"Ha", "OIII_5007"} and list(inter["Ha"]) == ["OIII_5007", "SIII_9532"] and list(inter["OIII_5007"]) == ["Ha"]
    # at most: a share of the catalog if every potential interloper were taken, the 3 galaxies inside plus 3 interlopers
    assert inter["Ha"]["OIII_5007"]["n"] == 1 and inter["OIII_5007"]["Ha"]["at_most"] == pytest.approx(1 / 6)
    assert p.major_pairs == [("OIII_5007", "Ha"), ("SIII_9532", "Ha"), ("Ha", "OIII_5007")]
    assert p.major_interlopers == ["OIII_5007", "SIII_9532", "Ha"]
    # by wavelength the Ha single line at z 1.1 could pass for [S III] 6312 inside the range, but that line is never single
    rare = p.interlopers("SIII_6312")
    assert rare["Ha"]["n"] == 1 and not rare["Ha"]["major"] and not rare["OIII_5007"]["major"]
    # Ha and [O III] taken for each other: both kinds of entry at once, p_interloper wins; [S III] 9532 is taken for Ha,
    # but Ha is not taken for [S III] 9532, which is not assignable (that entry would add [S III] interlopers)
    assert p.confusion_template(0.05, 0.01) == {"Ha": {"OIII_5007": 0.05}, "OIII_5007": {"Ha": 0.05}, "SIII_9532": {"Ha": 0.05}}
    z, f = p.contamination_at_most()      # two land at 1.10-1.11, where 2 galaxies are detected: 2 of 2 + 2; at 1.11-1.12
    k = [int(np.argmin(abs(z - v))) for v in (1.105, 1.115)]            # the [S III] 9532 one is alone
    assert f[k].tolist() == [0.5, 1.0] and np.nansum(f) == 1.5
    z, f = p.contamination_at_most("Ha", "OIII_5007")
    assert np.nanmax(f) == 0.25
    z, f = p.fraction_interloper("Ha", "OIII_5007")
    assert f.tolist() == [1.0] and z[0] == pytest.approx(0.605)
    with pytest.raises(ValueError, match="target"):
        p.fraction_interloper("Ha")
    with pytest.raises(ValueError, match="windows"):
        p.window
    assert p.default_name() == "all_z1p0_1p2" and "OIII_5007 as Ha" in repr(p)
    text = p.summary()
    assert "Proposal · every galaxy" in text and "taken for" in text and "not taken for" in text and "SIII_6312" in text
    q = Proposal.load(p.save(tmp_path / "all.npz"))
    assert q.settings == p.settings and q.interlopers() == p.interlopers()


def test_lines_taken_for_by_their_share_of_the_single_lines():
    """For line="all", a line is taken for when it is the single line of min_share of the single-line galaxies inside the
    range, not of every detected galaxy. Here [O III] 5007 is the single line of 1 of the 3 detected galaxies at 1.0-1.2
    and of 1 of the 2 single-line ones. (In the mock at 1.0-1.2 it is 0.04 % of the detected galaxies but 1.4 % of the
    single-line ones, and the single Hα lines of 0.53-0.68 that could be taken for it make 5 % of the catalog.)"""
    p = _every_line()
    again = lambda **kw: Proposal(p.counts, p.z_edges, {**p.settings, **kw}, NAMES, LS.rest_waves, area_deg2=2.0)
    inside = p.single_lines_inside()
    assert inside["OIII_5007"]["share"] == pytest.approx(1 / 3) and inside["OIII_5007"]["share_single"] == 0.5
    q = again(min_share=0.4)              # below 0.4 of the detected galaxies, but half of the single lines: taken for
    assert q.assignable == ["Ha", "OIII_5007"] == q.worst_case_lines() and "Ha" in q.interlopers()["OIII_5007"]
    none = again(min_share=0.6)
    assert none.assignable == [] == none.worst_case_lines() and none.interlopers() == {}
    assert none.worst_case_lines(min_count=1) == ["Ha", "OIII_5007"]       # the harsher worst case: any line single inside
    text = none.summary()
    assert "none above the minor-line cut" in text and "worst case" not in text and max(map(len, text.splitlines())) <= 100
    text = p.summary()
    assert "of single¹" in text and "the worst case, every single line an interloper" in text


def test_hint_says_what_line_means():
    text = line_hint("Ha", NAMES, LS.rest_waves)
    assert all(f'"{k}"' in text for k in ("Ha", "OIII_5007", "all")) and "6564.7" in text and "every galaxy" in text
    assert all(n in text for n in NAMES) and max(map(len, text.splitlines())) <= 100 and '▶ "Ha"' in text
    rows = line_hint("SIII_9069", NAMES, LS.rest_waves).splitlines()
    assert rows[0] == '[romanmock] line = "SIII_9069": which galaxies the catalog holds' and rows[1].startswith('  ▶ "SIII_9069"')
    assert '▶ "all"' in line_hint("all", NAMES, LS.rest_waves)


MODEL = ThresholdDetection(ConstantFluxLimit(1.5e-16, BAND, secondary_flux_limit=6e-17))
HB = NAMES.index("Hb")


def _scanned(interloper_fraction=0.3, single_line=False):
    """Galaxies whose lines change with the secondary limit (primary 1.5e-16, proposed secondary 6e-17, q = 0.4): Hα
    alone; Hα with [O III] at 0.667 of the primary limit (single above q = 0.667); [O III] with Hα at 0.533 (Hα detected,
    and the galaxy in the Hα sample, below q = 0.533); two [O III] lines where Hα would be at 1.0-1.2, one alone, one
    with Hβ at 0.2 (a potential interloper above q = 0.2)."""
    z = np.array([1.103, 1.103, 1.103, 1.753, 1.753] + ([1.103, 1.103, 1.753] if single_line else []))
    flux = np.zeros((z.size, len(NAMES)))
    flux[0, HA] = 2e-16
    flux[1, HA], flux[1, O3] = 2e-16, 1e-16
    flux[2, O3], flux[2, HA] = 2e-16, 8e-17
    flux[3, O3] = 2e-16
    flux[4, O3], flux[4, HB] = 2e-16, 3e-17
    if single_line:          # Hα with [O III] at 0.35 of the primary limit; Hα and [O III] both above it, inside the range
        flux[5, HA], flux[5, O3] = 2e-16, 5.25e-17       # and where Hα would be at 1.0-1.2 (the brighter one [O III])
        flux[6, HA], flux[6, O3] = 2e-16, 3e-16
        flux[7, HA], flux[7, O3] = 2e-16, 3e-16
    model = ThresholdDetection(MODEL.sensitivity, single_line="Ha") if single_line else MODEL
    det = detect_lines(flux, z, LS, model)
    window = target_window(LS.rest_waves, HA, (1.0, 1.2), BAND)
    edges = proposal_grid((1.0, 1.2), 0.01, 3.0)
    counts = proposal_counts(det, z, LS, "Ha", window, edges, primary_snr=primary_snr(model), q_edges=scan_edges(),
                             z_range=(1.0, 1.2), q_show=(0.3, 0.4, 0.5))
    sens = ConstantFluxLimit(1.5e-16, BAND, secondary_flux_limit=None) if single_line else MODEL.sensitivity
    settings = {"line": "Ha", "z_range": [1.0, 1.2], "targets": ["Ha"], "windows": [list(window)], "min_count": 1,
                "min_share": 1e-3, "sensitivity": sens.to_config(), "rescaling": None, "doublets": None,
                "base": "/nowhere", "scan": dict(SCAN), "interloper_fraction": None if single_line else interloper_fraction,
                "detection": model.to_config(), "secondary_ratios": [0.3, 0.4, 0.5]}
    return Proposal(counts, edges, settings, NAMES, LS.rest_waves, area_deg2=2.0), counts


def test_secondary_scan(tmp_path):
    """A detected galaxy has a single line at q when its second line, in units of the primary limit, is at most q; the
    Hα sample loses the galaxy whose Hα is a secondary detection above q = 0.533. On the edges of the scan the counts are
    exact, and at the proposed secondary limit they are those of the tables against redshift."""
    p, c = _scanned()
    assert c["scan_window"].shape == (SCAN["n"] + 2, len(NAMES), 1, 2) and c["scan_target"].shape == (SCAN["n"] + 2, 1)
    assert p.has_scan and p.thresholds == "flux" and p.primary == 1.5e-16 and p.secondary == 6e-17
    n_sample, pairs = p._numbers(np.array([0.1, 0.4, 0.6, 1.0]))
    np.testing.assert_allclose(n_sample, [3, 3, 2, 2])
    n, n_out = pairs[("OIII_5007", "Ha")]
    np.testing.assert_allclose(n, [1, 2, 2, 2])
    np.testing.assert_allclose(n_out, n)                                 # one target line: none of them is in its sample
    assert n[1] == c["n_window_single"][:, O3, 0].sum() and n_sample[1] == c["n_line_det"][p.inside, HA].sum()
    assert set(pairs) == {("OIII_5007", "Ha")}
    scan = p.interloper_scan([2.25e-17, 6e-17, 9e-17])                  # q 0.15, 0.4, 0.6
    np.testing.assert_allclose(scan["share"][("OIII_5007", "Ha")], [1 / 4, 2 / 5, 2 / 4])
    np.testing.assert_allclose(scan["total"], [1 / 4, 2 / 5, 2 / 4])
    assert scan["secondary"] == pytest.approx([2.25e-17, 6e-17, 9e-17], abs=0)
    assert p.secondary_for(0.45) == pytest.approx(8e-17, rel=0.015, abs=0) and p.secondary_for(0.6) is None
    assert p.secondary_for(0.2) == pytest.approx(scan_edges()[0] * 1.5e-16, abs=0)   # possible at every secondary limit
    text = p.summary()
    assert "── Interlopers against the secondary limit" in text and "secondary F [1e−17]" in text and "6⁴" in text
    assert "none" not in text and "possible from a secondary flux limit of 3e-17 up" in text       # 30 %: from q 0.2
    assert max(map(len, text.splitlines())) <= 100
    assert "possible from a secondary flux limit of 8e-17 up; at this proposal's 6e-17 at most 40.00 %" in \
        _scanned(0.45)[0].summary().replace("\n  ", " ")
    q = Proposal.load(p.save(tmp_path / "p.npz"))
    np.testing.assert_allclose(q.interloper_scan()["total"], p.interloper_scan()["total"])


def test_design_by_interloper_fractions():
    """A table of interloper fractions: the catalog holds N / (1 - F), and each pair is taken with p = f N_cat / n."""
    p, _ = _scanned()
    d = p.design({"Ha": {"OIII_5007": 0.25}})                           # q 0.4: N 3, n 2
    assert d["n_catalog"] == pytest.approx(4.0) and d["total"] == 0.25 and d["secondary"] == 6e-17
    (r,) = d["pairs"]
    assert (r["line"], r["as"]) == ("OIII_5007", "Ha") and r["p"] == pytest.approx(0.5) and r["n_expected"] == pytest.approx(1.0)
    assert r["window"] == pytest.approx(list(p.window))
    assert p.design({"Ha": {"OIII_5007": 0.25}}, secondary_flux_limit=2e-17)["pairs"][0]["p"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="at least 3e-17"):          # n 1 below q 0.2: too few single lines
        p.design({"Ha": {"OIII_5007": 0.3}}, secondary_flux_limit=2e-17)
    with pytest.raises(ValueError, match="secondary_flux_limit"):
        p.design({"Ha": {"OIII_5007": 0.1}}, secondary_snr=2)
    with pytest.raises(ValueError, match="between"):
        p.design({"Ha": {"OIII_5007": 0.1}}, secondary_flux_limit=2e-16)
    with pytest.raises(KeyError, match="this proposal is for Ha"):
        p.design({"OIII_5007": {"Ha": 0.1}})
    with pytest.raises(ValueError, match="no single SIII_9532 line"):
        p.design({"Ha": {"SIII_9532": 0.01}})
    with pytest.raises(TypeError):
        p.design({"OIII_5007": 0.1})
    assert p.fraction_template() == {"Ha": {"OIII_5007": 0.3}}
    with pytest.raises(ValueError, match="at least 8e-17"):
        p.fraction_template(0.45)
    text = p._design_text(d)
    assert "Interlopers by design" in text and "0.5000" in text and max(map(len, text.splitlines())) <= 100


def test_windowed_confusion():
    """A single line is taken for another only inside the window of its pair, with the pair's probability; one uniform
    deviate per galaxy, as a confusion matrix draws."""
    window = target_window(LS.rest_waves, HA, (1.0, 1.2), BAND)
    model = build_confusion({"type": "windowed", "pairs": [{"line": "OIII_5007", "as": "Ha", "window": list(window), "p": 0.5}]}, LS)
    assert isinstance(model, WindowedConfusion) and model.to_config()["pairs"][0]["p"] == 0.5
    rng = np.random.default_rng(1)
    n = 20000
    true = np.full(n, O3)
    wave = np.where(np.arange(n) < n // 2, 13800.0, 10500.0)            # half inside the window of Hα, half not
    j = model.assign(true, {"wave_obs": wave}, rng)
    inside = wave == 13800.0
    assert np.all(j[~inside] == O3) and abs((j[inside] == HA).mean() - 0.5) < 0.02
    a, b = np.random.default_rng(1), np.random.default_rng(1)
    model.assign(true, {"wave_obs": wave}, a)
    b.random(n)
    assert a.random() == b.random()                                    # the same stream as ConfusionMatrix.assign
    assert model.pair_probabilities()[O3, HA] == 0.5 and model.pair_probabilities().sum() == 0.5
    with pytest.raises(ValueError, match="overlapping"):
        WindowedConfusion(NAMES, [("OIII_5007", "Ha", (13000, 14000), 0.6), ("OIII_5007", "Hb", (13500, 14500), 0.6)])
    both = WindowedConfusion(NAMES, [("OIII_5007", "Ha", (13000, 14000), 0.5), ("OIII_5007", "Hb", (13500, 14500), 0.5)])
    j = both.assign(np.full(n, O3), {"wave_obs": np.full(n, 13700.0)}, np.random.default_rng(2))
    assert set(np.unique(j)) == {HA, HB} and abs((j == HA).mean() - 0.5) < 0.02


def _every_line_over(z_range):
    """The galaxies of :func:`_every_line` proposed for another range."""
    z, det = _galaxies(more=True)
    windows = target_windows(NAMES, LS.rest_waves, z_range, BAND)
    edges = proposal_grid(z_range, 0.01, 3.0)
    counts = proposal_counts(det, z, LS, list(windows), list(windows.values()), edges)
    settings = {"line": "all", "z_range": list(z_range), "targets": list(windows), "windows": [list(w) for w in windows.values()],
                "min_count": 1, "min_share": 1e-3, "rescaling": None, "doublets": None, "base": "/nowhere",
                "sensitivity": {"type": "constant", "flux_limit": 1.5e-16, "secondary_flux_limit": 6e-17, "wave_range": list(BAND)}}
    return Proposal(counts, edges, settings, NAMES, LS.rest_waves, area_deg2=2.0), z, det


def test_worst_case():
    """The worst case takes every single line for another line, one that puts it inside the range wherever there is one.
    At 1.0-1.2: the single [O III] line of z 1.75 and the [S III] 9532 line of z 0.45 are taken for Hα, the single Hα line
    of z 0.6 for [O III], and the single Hα (z 1.1) and [O III] (z 1.15) lines of the range leave it. Over 1.0-1.94 the
    windows of Hα and [O III] overlap: the
    single Hα line of z 1.1 taken for [O III], and the [O III] line of z 1.75 taken for Hα, stay inside with a wrong
    redshift, and the [S III] line inside both windows goes to the first line, Hα (the counts then hold it twice)."""
    from romanmock.confusion import WorstCaseConfusion
    from romanmock.redshift import IdOutcome, assign_redshifts

    p, z, det = _every_line_over((1.0, 1.2))
    assert p.worst_case_lines() == ["Ha", "OIII_5007"] == p.assignable
    w = p.worst_case()
    assert {(r["line"], r["as"]): r["n"] for r in w["pairs"]} == {("OIII_5007", "Ha"): 1, ("SIII_9532", "Ha"): 1, ("Ha", "OIII_5007"): 1}
    assert (w["n_multi"], w["n_interlopers"], w["n_catalog"]) == (1, 3, 4) and w["share"] == 0.75
    assert w["lost"] == {"Ha": 1, "OIII_5007": 1} and w["moved"] == {} and w["overlaps"] == []
    model = build_confusion({"type": "worst_case", "z_range": [1.0, 1.2], "taken_for": w["taken_for"]}, LS)
    assert isinstance(model, WorstCaseConfusion) and model.to_config()["taken_for"] == ["Ha", "OIII_5007"]
    za = assign_redshifts(det, z, LS, model)
    kept = (det.n_det > 0) & (za.z_assigned >= 1.0) & (za.z_assigned <= 1.2)
    assert kept.sum() == w["n_catalog"] and (kept & (za.outcome == IdOutcome.SINGLE_MISID)).sum() == w["n_interlopers"]
    assert not np.any(za.outcome == IdOutcome.SINGLE_CORRECT)        # no single line keeps its line
    assert za.line_assigned[5] == O3 and za.z_assigned[5] == pytest.approx(LS.rest_waves[HA] * 1.603 / LS.rest_waves[O3] - 1)
    assert za.line_assigned[0] == O3 and za.z_assigned[0] > 1.2       # the Hα line of the range leaves it, taken for [O III]
    text = p.worst_case_summary()
    assert "Worst case · every galaxy" in text and "Left out" in text and "SIII_6312" in text and max(map(len, text.splitlines())) <= 100
    # one line only: a single line of that line has nothing else to be taken for, and gets no redshift
    lone = WorstCaseConfusion(LS, (1.0, 1.2), ["Ha"])
    assert lone.assign(np.array([HA, O3]), {"wave_obs": np.array([13800.0, 13800.0])}).tolist() == [-1, HA]
    assert lone.pair_probabilities()[O3, HA] == 1 and lone.pair_probabilities()[HA, HA] == 0
    with pytest.raises(KeyError, match="unknown"):
        WorstCaseConfusion(LS, (1.0, 1.2), ["Hz"])
    # over 1.0-1.94: single lines of the range stay inside with a wrong redshift
    wide, z, det = _every_line_over((1.0, 1.94))
    w = wide.worst_case(["Ha", "OIII_5007"])
    assert w["overlaps"] == [("Ha", "OIII_5007")] and w["moved"] == {"Ha": 1, "OIII_5007": 1} and w["lost"] == {"OIII_5007": 1}
    za = assign_redshifts(det, z, LS, WorstCaseConfusion(LS, (1.0, 1.94), ["Ha", "OIII_5007"]))
    kept = (det.n_det > 0) & (za.z_assigned >= 1.0) & (za.z_assigned <= 1.94)
    assert kept[[0, 2]].all() and (za.outcome[[0, 2]] == IdOutcome.SINGLE_MISID).all()    # inside, with a wrong redshift
    assert za.line_assigned[7] == HA and kept.sum() == 6 < w["n_catalog"] == 7      # [S III]: counted in both windows
    rows = {(r["line"], r["as"]): r for r in wide.misidentifications()}
    assert rows[("Ha", "OIII_5007")]["n_inside"] == 1 and rows[("Ha", "OIII_5007")]["worst_case"]


def test_single_line_mode():
    """Without a secondary limit a proposal is for single-line clean catalogs: every detected galaxy keeps one line, the
    target line wherever it is above the primary limit (every Hα galaxy above it is a single-line Hα galaxy, even with a
    brighter [O III]), else its brightest line, and there are no interlopers. The true single-line share is counted
    against redshift at 0.3, 0.4 and 0.5 times the primary limit; with a secondary limit the catalog keeps the same
    galaxies (primary_line) with their other lines above it."""
    import matplotlib

    matplotlib.use("Agg")
    p, c = _scanned(single_line=True)
    both = np.zeros((1, len(NAMES)))
    both[0, HA], both[0, O3] = 2e-16, 3e-16
    brightest = detect_lines(both, [1.1], LS, ThresholdDetection(MODEL.sensitivity, single_line=True))
    first = detect_lines(both, [1.1], LS, ThresholdDetection(MODEL.sensitivity, single_line="Ha"))
    assert brightest.n_det.tolist() == first.n_det.tolist() == [1] and brightest.detected[0, O3] and first.detected[0, HA]
    assert first.primary[0, O3] and ThresholdDetection(MODEL.sensitivity, single_line="Ha").to_config()["single_line"] == "Ha"
    assert p.single_line and p.secondary == np.inf and p.settings["interloper_fraction"] is None
    assert p._detection_at(0.4) == {"type": "threshold", "snr_at_limit": 6.5, "primary_line": "Ha"}
    assert p._detection_at(np.inf)["single_line"] == "Ha"
    assert c["n_det"].sum() == 8 and set(np.flatnonzero(c["n_single"].sum(axis=0))) == {HA, O3}
    assert p.n_sample == 4                  # galaxies 0, 1, 5, 6; galaxy 2 (Hα below the primary limit) is an [O III] one
    # with a secondary limit: only the galaxies with Hα above the primary limit, each with its lines above the secondary
    model = ThresholdDetection(MODEL.sensitivity, **{k: v for k, v in p._detection_at(0.4).items() if k == "primary_line"})
    flux = np.zeros((4, len(NAMES)))
    flux[0, HA], flux[0, O3] = 2e-16, 1e-16                             # Hα above the primary, [O III] above 6e-17
    flux[1, HA], flux[1, O3] = 2e-16, 5e-17                             # [O III] below the secondary
    flux[2, O3], flux[2, HA] = 2e-16, 8e-17                             # Hα only above the secondary
    flux[3, O3] = 2e-16
    det = detect_lines(flux, np.full(4, 1.1), LS, model)
    assert det.n_det.tolist() == [2, 1, 0, 0] and model.to_config()["primary_line"] == "Ha"
    z, f = p.single_line_fraction(0.3)
    k = int(np.argmin(abs(z - 1.105)))
    assert f[k] == pytest.approx(1 / 4) and p.single_line_fraction(0.4)[1][k] == pytest.approx(2 / 4)
    assert p.single_line_fraction(0.5)[1][k] == pytest.approx(2 / 4)
    with pytest.raises(KeyError, match="not 0.45"):
        p.single_line_fraction(0.45)
    text = p.summary()
    assert "── Hα single-line galaxies against redshift" in text and "4.5e-17" in text and "7.5e-17" in text
    assert "single-line clean catalog" in text and "make_catalog(secondary_flux_limit=…, out_dir=…)" in text
    assert "Potential interlopers" not in text and "Interlopers against" not in text
    assert max(map(len, text.splitlines())) <= 100 and "no interlopers" in repr(p)
    for call in (p.interloper_scan, lambda: p.secondary_for(0.1), lambda: p.design({"Ha": {"OIII_5007": 0.25}}),
                 lambda: p.fraction_template(0.1), p.plot_secondary, p.confusion_template):
        with pytest.raises(ValueError, match="single-line clean catalogs"):
            call()
    with pytest.raises(ValueError, match="single-line clean catalogs"):
        p.make_catalog(fractions={"Ha": {"OIII_5007": 0.25}}, out_dir="/nowhere")
    with pytest.raises(ValueError, match="single-line clean catalogs"):
        p.make_catalog({"OIII_5007": {"Ha": 0.1}}, out_dir="/nowhere")
    assert len(p.plot().axes) == 1

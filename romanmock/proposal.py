"""Step 1 of the two-step construction of a catalog with line interlopers: the proposal.

A proposal answers, before any identification is chosen, how many galaxies of a catalog would rest on a single
line: for a target line (Hα by default), a redshift range and proposed thresholds (flux limits or S/N), it counts
in one pass over the base catalog

* the **single-line galaxies inside the redshift range**: the share of the detected galaxies at each true
  redshift whose only detected line is the target line, or another line;
* the **potential interlopers**: the single-line galaxies at other redshifts whose line falls where the target
  line would be for a redshift inside the range, so that, taken for the target line, they would enter the
  catalog. Their single-line fraction against their own (true) redshift, and how many of them would land at each
  assigned redshift.

Lines with few single-line galaxies (fewer than ``min_count``, or below ``min_share`` of the detected galaxies,
respectively of the target sample) are left out as minor. The proposal fixes everything but the identification:
:meth:`Proposal.make_catalog` takes the confusion table (or ``"empirical"``) and an output directory and writes
the catalog, with the same thresholds, line rescaling and doublet rule.

``line="all"`` proposes a catalog of every galaxy with a redshift in the range, whichever line gave it. Its sample
is every detected galaxy inside the range, and its potential interlopers are pairs: a single line that, taken for
another line, gets a redshift inside the range. That other line must be one that single-line galaxies inside the
range actually show (:attr:`Proposal.assignable`); the counts keep every line in the bandpass.

The proposal also scans the secondary limit, the threshold of the other lines of a galaxy, which decides how many
galaxies keep a single line: for every secondary limit from the primary one down to a hundredth of it, how many
potential interlopers there are, and so the most they can make of the catalog (:meth:`Proposal.interloper_scan`,
:meth:`Proposal.plot_secondary`, a table in the summary). A catalog can then be designed by its interloper fractions
instead of a confusion table: ``proposal.make_catalog(fractions={"Ha": {"OIII_5007": 0.005, "SIII_9532": 0.005}},
secondary_flux_limit=1e-16, out_dir=...)`` takes, inside the window of each pair, the probability that makes the
interlopers of the pair that share of the catalog (:meth:`Proposal.design`; the identification is a
:class:`~romanmock.confusion.WindowedConfusion`).

Without a secondary limit and without an expected interloper fraction, a proposal is for a **single-line clean
catalog**: the galaxies with the target line above the primary limit (every detected galaxy for ``line="all"``), each
with its true redshift, and no interlopers. Its summary gives the share of them that are true single-line galaxies,
without another line above 0.3, 0.4 and 0.5 times the primary limit. :meth:`Proposal.make_catalog` then makes the catalog with one line per galaxy, every galaxy a
single-line one, or, given a secondary limit, the same galaxies with their other lines above it, so that only those
without another line are single-line galaxies.

``BaseCatalog.make_proposal`` builds it; :func:`proposal_counts` is the array-level part.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .detection import DetectionResult
from .lines import get_line_set
from .redshift import brightest_detected

PROPOSAL_STREAM = 50000   #: random stream of the proposal scan, (seed, PROPOSAL_STREAM, step); used by noisy detection only
ALL_LINES = "all"         #: ``line`` of a proposal for every galaxy, whichever line gave its redshift
#: columns of a catalog made from a proposal when none are given
CATALOG_COLUMNS = ["gal_id", "ra", "dec", "redshift_true", "redshift_rsd", "redshift_assigned", "id_outcome", "line_true",
                   "line_assigned", "n_det"]
#: the setting of the secondary limit in the sensitivity config, per kind of thresholds (Proposal.thresholds)
SECONDARY_KEYS = {"flux": "secondary_flux_limit", "snr": "secondary_snr", "ratio": "secondary_ratio"}
COUNT_KEYS = ("n_det", "n_line_det", "n_single", "n_window_det", "n_window_single", "n_assigned")
#: the scan of the secondary limit: q = secondary / primary limit on the edges 10^(log_min + step k), k = 0 .. n
SCAN = {"log_min": -2.0, "step": 0.001, "n": 2000}
SECONDARY_RATIOS = (0.3, 0.4, 0.5)  #: secondary limits (times the primary) of the single-line fractions against redshift
SCAN_SHOWN = 0.1          #: the lowest secondary limit the reports and figures show, as a fraction of the primary limit


def _number(x: float) -> str:
    """A count that may be fractional (an expectation): ``75,563``, or ``1.33`` below 100."""
    return f"{x:.3g}" if abs(x) < 100 else f"{x:,.0f}"


def scan_edges(log_min: float = SCAN["log_min"], step: float = SCAN["step"], n: int = SCAN["n"], extra=()) -> np.ndarray:
    """Edges of the secondary-limit scan, as fractions ``q`` of the primary limit (0.01 to 1 by default), with the
    ``extra`` ones (the proposal's own secondary limit, where the counts are then exact)."""
    edges = 10.0 ** np.round(float(log_min) + float(step) * np.arange(int(n) + 1), 12)
    return np.unique(np.concatenate([edges, np.asarray(list(extra), dtype=np.float64)]))


def secondary_ratio(sensitivity: Mapping) -> float:
    """The secondary limit of a sensitivity config as a fraction of its primary limit (1 for a single tier)."""
    kind = sensitivity.get("type", "constant")
    if kind == "constant":
        second = sensitivity.get("secondary_flux_limit")
        return 1.0 if second is None else float(second) / float(sensitivity["flux_limit"])
    if kind == "roman_grism":
        second = sensitivity.get("secondary_snr")
        return 1.0 if second is None else float(second) / float(sensitivity["snr"])
    return float(sensitivity.get("secondary_ratio") or 1.0)


def primary_snr(model) -> float:
    """The S/N a detection model gives a line at the primary threshold, so that ``det.snr / primary_snr(model)`` is each
    line's signal in units of that threshold: above 1 a primary detection and, in a galaxy with one, detected above
    ``q`` when the secondary limit is ``q`` times the primary one. Every sensitivity model sets its secondary limit as
    such a fraction at each wavelength (flux limits ``F2 / F1``, S/N thresholds ``snr2 / snr1``, a tabulated limit its
    ``secondary_ratio``), so one number per galaxy decides its lines at every secondary limit."""
    value = getattr(model, "snr_threshold", None)
    return float(value if value is not None else model.snr_at_limit)


# =====================================================================================
# array level
# =====================================================================================
def proposal_grid(z_range: Sequence[float], dz: float, z_max: float) -> np.ndarray:
    """Redshift bin edges of a proposal: ``z_range`` cut into equal bins of about ``dz`` (its ends are edges),
    continued with the same bins from 0 to ``z_max``."""
    lo, hi = float(z_range[0]), float(z_range[1])
    n_in = max(1, int(round((hi - lo) / dz)))
    step = (hi - lo) / n_in
    below = int(np.floor(lo / step + 1e-9))
    above = int(np.ceil((max(z_max, hi) - lo) / step - 1e-9))
    return lo + step * np.arange(-below, above + 1)


def target_window(rest_waves, target_index: int, z_range, wave_range) -> tuple[float, float]:
    """Observed wavelengths [Å] where the target line is for a redshift inside ``z_range``, inside the bandpass."""
    rest = float(np.asarray(rest_waves)[target_index])
    lo = max(float(wave_range[0]), rest * (1.0 + float(z_range[0])))
    hi = min(float(wave_range[1]), rest * (1.0 + float(z_range[1])))
    if not hi > lo:
        raise ValueError(f"the target line is outside the bandpass {list(wave_range)} at {list(z_range)}")
    return lo, hi


def window_intervals(names, rest_waves, window) -> dict[str, tuple[float, float]]:
    """``{line: (z_min, z_max)}``: the true redshifts at which each line falls inside ``window``."""
    rest = np.asarray(rest_waves, dtype=np.float64)
    return {n: (float(window[0] / r - 1.0), float(window[1] / r - 1.0)) for n, r in zip(names, rest)}


def target_windows(names, rest_waves, z_range, wave_range) -> dict[str, tuple[float, float]]:
    """``{line: window}`` for every line inside the bandpass at some redshift of ``z_range``: the targets of a proposal
    with ``line="all"``."""
    out = {}
    for i, name in enumerate(names):
        try:
            out[name] = target_window(rest_waves, i, z_range, wave_range)
        except ValueError:
            continue
    return out


def line_hint(line: str, names, rest_waves) -> str:
    """What the ``line`` argument of :meth:`BaseCatalog.make_proposal` means, the choice made marked ▶; printed by it."""
    import textwrap

    from .report import PREFIX, WIDTH, label

    rest = dict(zip(names, np.asarray(rest_waves, dtype=np.float64)))
    shown = list(dict.fromkeys([*([line] if line in rest else []), *(n for n in ("Ha", "OIII_5007") if n in rest)]))
    rows = []
    for j, n in enumerate(shown):
        z = f"z = λ_obs / {rest[n]:.1f} Å − 1"
        if j == 0:
            text = (f"the galaxies in which {label(n)} is identified, with {z} inside z_range: {label(n)} detected, alone "
                    f"or with other lines, or a single line taken for {label(n)} (an interloper)")
        else:
            text = f"the same for {label(n)}, {z}" + ("; a galaxy with both lines is in both catalogs" if j == 1 else "")
        rows.append((f'"{n}"', text))
    rows.append((f'"{ALL_LINES}"', "every galaxy with a redshift inside z_range, whichever line gave it"))
    others = [n for n in names if n not in shown]
    if others:
        rows.append(("or", ", ".join(f'"{n}"' for n in others)))
    w = max(len(k) for k, _ in rows) + 2
    out = [f'{PREFIX} line = "{line}": which galaxies the catalog holds']
    for key, text in rows:
        mark = "▶ " if key == f'"{line}"' else "  "
        out += textwrap.wrap(text, WIDTH, initial_indent=f"  {mark}{key:<{w}s}", subsequent_indent=" " * (4 + w),
                             break_on_hyphens=False, break_long_words=False)
    return "\n".join(out)


def proposal_counts(det: DetectionResult, z_rsd, line_set, targets, windows, z_edges, merged=None,
                    both=None, primary_snr=None, q_edges=None, z_range=None, q_show=None) -> dict[str, np.ndarray]:
    """The counts behind a proposal, per bin of ``z_edges`` (bin k: ``z_edges[k] <= z < z_edges[k+1]``).

    ``targets`` is the target line, or a list of them (``line="all"``: every line in the bandpass), and ``windows``
    its window, or one per target: the observed wavelengths of the target for the redshift range.

    ``n_det[z]``: detected galaxies at true redshift ``z_rsd``; ``n_line_det[z, i]``: those in which line ``i`` is
    detected; ``n_single[z, i]``: single-line galaxies whose line is ``i``; ``n_window_det[z, i, k]``: detected
    galaxies whose line ``i`` falls inside the window of target ``k``; ``n_window_single[z, i, k]``: single-line
    galaxies whose line ``i`` falls inside it; ``n_assigned[z, i, k]``: the same galaxies binned in the redshift they
    would get if taken for target ``k``.
    With a doublet rule, ``n_doublet_both[z]`` / ``n_doublet_merged[z]`` count the detected galaxies with both lines
    in the bandpass and those whose doublet was merged (``both``, ``merged``: per-galaxy masks).

    With ``primary_snr`` (:func:`primary_snr` of the detection model), ``q_edges`` (:func:`scan_edges`) and ``z_range``,
    also the scan of the secondary limit, in bins of ``q`` = secondary / primary limit (bin 0: values <= the first edge,
    bin k: between edges k-1 and k, the last: above the last edge). A detected galaxy has a single line at ``q`` when
    its second line, in units of the primary threshold, is at most ``q``. ``scan_window[q, i, k, inside]``: detected
    galaxies whose brightest line ``i`` falls inside the window of target ``k``, binned in their second line, split by
    a true redshift outside (0) or inside (1) the range; ``scan_target[q, k]``: detected galaxies inside the range
    binned in the signal of target ``k`` (detected at ``q`` when above it). With ``q_show`` (secondary limits as
    fractions of the primary one), per bin of true redshift: ``n_primary[z, i]``, detected galaxies with line ``i``
    above the primary limit; ``nq_single[j, z, i]``, those whose only line above ``q_show[j]`` is ``i`` (single-line at
    that secondary limit); ``nq_window_single[j, z, i, k]``, those of them whose line falls inside the window of
    target ``k``.
    """
    ls = get_line_set(line_set)
    names, rest = ls.names, np.asarray(ls.rest_waves, dtype=np.float64)
    if isinstance(targets, str):
        targets, windows = [targets], [windows]
    edges = np.asarray(z_edges, dtype=np.float64)
    nz, nl = edges.size - 1, len(names)
    z = np.asarray(z_rsd, dtype=np.float64)
    zbin = np.searchsorted(edges, z, side="right") - 1
    ok = (zbin >= 0) & (zbin < nz) & np.isfinite(z)
    detected = ok & (det.n_det > 0)
    single = ok & (det.n_det == 1)
    line = brightest_detected(det)
    wave = rest[None, :] * (1.0 + z[:, None])

    def hist(mask):
        return np.bincount(zbin[mask], minlength=nz).astype(np.int64)

    def hist_lines(mask, bins=zbin):
        return np.bincount(bins[mask] * nl + line[mask], minlength=nz * nl).reshape(nz, nl).astype(np.int64)

    out = {"n_det": hist(detected),
           "n_line_det": np.stack([hist(detected & det.detected[:, i]) for i in range(nl)], axis=1),
           "n_single": hist_lines(single)}
    cell = zbin[:, None] * nl + np.arange(nl)[None, :]            # (redshift bin, line) of every galaxy and line
    rows = np.flatnonzero(single)
    wave_single = np.full(z.size, np.nan)                        # the observed wavelength of the line of a single
    wave_single[rows] = wave[rows, line[rows]]
    window_det, window_single, assigned = [], [], []
    for target, (w0, w1) in zip(targets, windows):
        m = detected[:, None] & (wave >= w0) & (wave <= w1)
        window_det.append(np.bincount(cell[m], minlength=nz * nl).reshape(nz, nl))
        in_win = single & (wave_single >= w0) & (wave_single <= w1)
        window_single.append(hist_lines(in_win))
        z_as = np.where(in_win, wave_single / rest[names.index(target)] - 1.0, np.inf)
        bin_as = np.searchsorted(edges, z_as, side="right") - 1
        assigned.append(hist_lines(in_win & (bin_as >= 0) & (bin_as < nz), bin_as))
    out["n_window_det"] = np.stack(window_det, axis=2).astype(np.int64)
    out["n_window_single"] = np.stack(window_single, axis=2)
    out["n_assigned"] = np.stack(assigned, axis=2)
    if merged is not None:
        out["n_doublet_both"] = hist(detected & np.asarray(both, dtype=bool))
        out["n_doublet_merged"] = hist(detected & np.asarray(merged, dtype=bool))
    if primary_snr is not None:
        centres = 0.5 * (edges[1:] + edges[:-1])
        inside = np.zeros(z.size, dtype=bool)
        inside[ok] = ((centres > float(z_range[0])) & (centres < float(z_range[1])))[zbin[ok]]
        out.update(_scan_counts(det, float(primary_snr), wave, detected, inside, targets, windows, names,
                                np.asarray(q_edges, dtype=np.float64), zbin, nz, q_show))
    return out


def _scan_counts(det: DetectionResult, primary_snr: float, wave, detected, inside, targets, windows, names,
                 q_edges, zbin=None, nz=None, q_show=None) -> dict[str, np.ndarray]:
    """The scan of the secondary limit of :func:`proposal_counts`."""
    nl, nq = len(names), q_edges.size + 1
    rows = np.flatnonzero(detected)
    r = np.asarray(det.snr, dtype=np.float64)[rows] / primary_snr        # signal in units of the primary threshold
    top = np.argmax(r, axis=1)                                          # the brightest line: the single one, if any
    second = np.partition(r, nl - 2, axis=1)[:, nl - 2] if nl > 1 else np.zeros(rows.size)
    qbin = np.searchsorted(q_edges, second, side="left")                # single at q iff second <= q
    ins = inside[rows].astype(np.int64)
    w_top = wave[rows, top]
    window = []
    for w0, w1 in windows:
        m = (w_top >= w0) & (w_top <= w1)
        cell = (qbin[m] * nl + top[m]) * 2 + ins[m]
        window.append(np.bincount(cell, minlength=nq * nl * 2).reshape(nq, nl, 2))
    r_in = r[ins.astype(bool)]
    target = [np.bincount(np.searchsorted(q_edges, r_in[:, names.index(t)], side="left"), minlength=nq) for t in targets]
    out = {"scan_window": np.stack(window, axis=2).astype(np.int64), "scan_target": np.stack(target, axis=1).astype(np.int64)}
    if q_show is not None:
        primary = np.asarray(det.primary, dtype=bool)[rows]
        zb = zbin[rows]
        out["n_primary"] = np.stack([np.bincount(zb[primary[:, i]], minlength=nz) for i in range(nl)], axis=1).astype(np.int64)
        single_q, window_q = [], []
        for qv in q_show:
            single = second <= float(qv)
            single_q.append(np.bincount(zb[single] * nl + top[single], minlength=nz * nl).reshape(nz, nl))
            per = []
            for w0, w1 in windows:
                m = single & (w_top >= w0) & (w_top <= w1)
                per.append(np.bincount(zb[m] * nl + top[m], minlength=nz * nl).reshape(nz, nl))
            window_q.append(np.stack(per, axis=2))
        out["nq_single"] = np.stack(single_q).astype(np.int64)
        out["nq_window_single"] = np.stack(window_q).astype(np.int64)
    return out


def add_counts(total: dict | None, part: Mapping[str, np.ndarray]) -> dict:
    """Sum of two count dicts of :func:`proposal_counts`."""
    if total is None:
        return {k: np.array(v, copy=True) for k, v in part.items()}
    for k, v in part.items():
        total[k] = total[k] + v
    return total


# =====================================================================================
# the result
# =====================================================================================
class Proposal:
    """What a catalog of one line (or of every line, ``line="all"``) in a redshift range would hold at proposed
    thresholds, before the identification of its single-line galaxies is chosen. Made by
    :meth:`BaseCatalog.make_proposal`.

    ``proposal.summary()`` prints the single-line fractions, the potential interlopers and the most they can make of
    the catalog against the secondary limit; ``proposal.plot()`` draws them against redshift and
    ``proposal.plot_secondary()`` against the secondary limit. ``proposal.make_catalog(fractions=..., out_dir=...)``
    makes a catalog with the interloper fractions asked for (:meth:`design`, :meth:`fraction_template`);
    ``proposal.make_catalog(confusion, out_dir)`` one with a confusion table (:meth:`confusion_template`) or the
    empirical identification.
    """

    def __init__(self, counts: Mapping[str, np.ndarray], z_edges, settings: Mapping, line_names: Sequence[str],
                 rest_waves, area_deg2: float | None = None, base=None):
        self.counts = {k: np.asarray(v) for k, v in counts.items()}
        self.z_edges = np.asarray(z_edges, dtype=np.float64)
        self.z = 0.5 * (self.z_edges[1:] + self.z_edges[:-1])
        self.settings = dict(settings)
        self.line_names = list(line_names)
        self.rest_waves = np.asarray(rest_waves, dtype=np.float64)
        self.area_deg2 = None if area_deg2 is None else float(area_deg2)
        self._base = base

    # ---- what was asked ------------------------------------------------------------
    @property
    def line(self) -> str:
        """The target line, or ``"all"``."""
        return self.settings["line"]

    @property
    def all_lines(self) -> bool:
        """A proposal for every galaxy, whichever line gave its redshift (``line="all"``)."""
        return self.line == ALL_LINES

    @property
    def z_range(self) -> tuple[float, float]:
        return tuple(self.settings["z_range"])

    @property
    def targets(self) -> list[str]:
        """The lines whose windows were counted: the target line, or every line in the bandpass for ``line="all"``."""
        return list(self.settings["targets"])

    @property
    def windows(self) -> dict[str, tuple[float, float]]:
        """``{target: (min, max)}``: the observed wavelengths [Å] of each target for a redshift inside the range."""
        return {t: tuple(w) for t, w in zip(self.settings["targets"], self.settings["windows"])}

    @property
    def window(self) -> tuple[float, float]:
        """The window of the target line."""
        if self.all_lines:
            raise ValueError("a proposal with line='all' has one window per line: proposal.windows")
        return self.windows[self.line]

    @property
    def target_index(self) -> int:
        """Index of the target line in ``line_names``."""
        if self.all_lines:
            raise ValueError("a proposal with line='all' has no single target line")
        return self.line_names.index(self.line)

    def _k(self, target: str) -> int:
        """Index of ``target`` on the last axis of the window counts."""
        if target not in self.settings["targets"]:
            raise KeyError(f"{target!r} is not counted in this proposal; its lines are {self.targets}")
        return self.settings["targets"].index(target)

    def _target(self, target: str | None) -> str:
        if target is None:
            if self.all_lines:
                raise ValueError("line='all': give the line the single lines are taken for, target=<line>")
            return self.line
        return target

    @property
    def inside(self) -> np.ndarray:
        """Redshift bins inside the requested range."""
        lo, hi = self.z_range
        return (self.z > lo) & (self.z < hi)

    def _sample(self) -> np.ndarray:
        """The sample per bin of true redshift: the galaxies with the target line detected, or every detected galaxy."""
        return self.counts["n_det"] if self.all_lines else self.counts["n_line_det"][:, self.target_index]

    @property
    def n_sample(self) -> int:
        """Galaxies at a true redshift inside the range with the target line detected (every detected galaxy for
        ``line="all"``): the sample before interlopers."""
        return int(self._sample()[self.inside].sum())

    @property
    def assignable(self) -> list[str]:
        """The lines a single line can be taken for in this catalog: the target line or, for ``line="all"``, the lines
        in the bandpass that are single lines of a major share of the galaxies inside the range. A single line falling
        where another line would be is taken for it only if single-line galaxies show that line; lines that are almost
        never single there ([S III] λ6312, a faint doublet member, ...) are not taken for."""
        if not self.all_lines:
            return [self.line]
        major = set(self.major_inside)
        return [t for t in self.targets if t in major]

    @property
    def base(self):
        if self._base is None:
            from .catalog import BaseCatalog

            self._base = BaseCatalog(self.settings["base"], self.settings.get("base_tag", "romanmock_base"))
        return self._base

    # ---- the numbers ---------------------------------------------------------------
    def _major(self, count: int, share: float) -> bool:
        return count >= int(self.settings["min_count"]) and share >= float(self.settings["min_share"])

    def single_lines_inside(self) -> dict[str, dict]:
        """Per line: the single-line galaxies inside the range whose line it is (``n``), their share of the detected
        galaxies inside the range (``share``), and whether they count (``major``)."""
        inside = self.inside
        n_det = int(self.counts["n_det"][inside].sum())
        out = {}
        for i, name in enumerate(self.line_names):
            n = int(self.counts["n_single"][inside, i].sum())
            if n:
                share = n / max(n_det, 1)
                out[name] = {"n": n, "share": share, "major": self._major(n, share)}
        return out

    def interlopers(self, target: str | None = None) -> dict:
        """Per line other than ``target``: the potential interlopers, single-line galaxies whose line falls where
        ``target`` would be inside the range. ``n``: how many; ``per_deg2``; ``z_true``: their redshifts; ``share``:
        their share of the detected galaxies at those redshifts; ``at_most``: their share of the catalog if every
        potential interloper were taken (the catalog then holds the sample plus all of them), the most any
        identification gives; ``major``.

        ``target`` defaults to the target line. For ``line="all"`` it can be any line of :attr:`targets`, and without
        it the result is ``{target: {line: ...}}`` over the :attr:`assignable` lines (only those pairs can be major)."""
        if target is None and self.all_lines:
            return {t: rows for t in self.assignable if (rows := self.interlopers(t))}
        target = self._target(target)
        k, t = self._k(target), self.line_names.index(target)
        n_sample = max(self.n_sample, 1)
        n_catalog = max(self._catalog_at_most(), 1)
        intervals = window_intervals(self.line_names, self.rest_waves, self.windows[target])
        assignable = target in self.assignable
        out = {}
        for i, name in enumerate(self.line_names):
            if i == t:
                continue
            n = int(self.counts["n_window_single"][:, i, k].sum())
            if not n:
                continue
            n_det = int(self.counts["n_window_det"][:, i, k].sum())
            at_most = int(self.counts["n_assigned"][:, i, k].sum()) / n_catalog
            out[name] = {"n": n, "per_deg2": None if not self.area_deg2 else n / self.area_deg2, "z_true": intervals[name],
                         "share": n / max(n_det, 1), "at_most": at_most,
                         "major": assignable and self._major(n, n / n_sample)}
        return out

    def _from_outside(self, target: str, i: int) -> int:
        """Potential interlopers of line ``i`` taken for ``target`` that are not in the sample already: all of them for a
        target line, those whose true redshift is outside the range for ``line="all"``."""
        n = self.counts["n_window_single"][:, i, self._k(target)]
        return int(n[~self.inside].sum() if self.all_lines else n.sum())

    def _catalog_at_most(self) -> int:
        """The catalog if every potential interloper (minor ones too) were taken for an :attr:`assignable` line."""
        total = self.n_sample
        for t in self.assignable:
            own = self.line_names.index(t)
            total += sum(self._from_outside(t, i) for i in range(len(self.line_names)) if i != own)
        return total

    @property
    def major_inside(self) -> list[str]:
        return [k for k, v in self.single_lines_inside().items() if v["major"]]

    @property
    def major_pairs(self) -> list[tuple[str, str]]:
        """The potential interlopers that count, as ``(line, the line it would be taken for)``."""
        return [(x, t) for t in self.assignable for x, v in self.interlopers(t).items() if v["major"]]

    @property
    def major_interlopers(self) -> list[str]:
        """The lines whose single lines are potential interlopers that count (in any pair, for ``line="all"``)."""
        return list(dict.fromkeys(x for x, _ in self.major_pairs))

    def fraction_inside(self, line: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Single-line fraction of the detected galaxies against true redshift inside the range: all single lines
        (``line=None``) or those whose line is ``line``."""
        inside = self.inside
        n_single = self.counts["n_single"][inside]
        n = n_single.sum(axis=1) if line is None else n_single[:, self.line_names.index(line)]
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.z[inside], np.where(self.counts["n_det"][inside] > 0, n / self.counts["n_det"][inside], np.nan)

    def fraction_interloper(self, line: str, target: str | None = None, ratio: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Single-line fraction of the potential interlopers of ``line`` against their true redshift: the share of the
        detected galaxies at that redshift whose only line is ``line``, inside the window of ``target`` (default: the
        target line; required for ``line="all"``). ``ratio``: at a secondary limit of that many times the primary one
        (one of ``settings["secondary_ratios"]``) instead of the proposal's own."""
        i, k = self.line_names.index(line), self._k(self._target(target))
        det = self.counts["n_window_det"][:, i, k]
        keep = det > 0
        if ratio is None:
            single = self.counts["n_window_single"][:, i, k]
        else:
            ratios = list(self.settings.get("secondary_ratios") or [])
            match = [j for j, r in enumerate(ratios) if np.isclose(r, ratio, rtol=1e-9, atol=0.0)]
            if not match or "nq_window_single" not in self.counts:
                raise KeyError(f"the proposal counted the secondary limits of {ratios} times the primary one, not {ratio}")
            single = self.counts["nq_window_single"][match[0]][:, i, k]
        return self.z[keep], single[keep] / det[keep]

    def contamination_at_most(self, line: str | None = None, target: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Against the assigned redshift inside the range: the potential interlopers of ``line`` (all lines for
        ``None``) taken for ``target`` as a share of the catalog at that redshift if every potential interloper were
        taken (the sample plus all of them), the most any identification gives. ``target`` defaults to the target
        line; for ``line="all"``, to every :attr:`assignable` line, where a single line that could be taken for two of
        them is counted for each (an upper bound)."""
        inside = self.inside
        targets = self.assignable if (target is None and self.all_lines) else [self._target(target)]
        n = np.zeros(int(inside.sum()))
        every = np.zeros(int(inside.sum()))
        assignable = self.assignable
        for tgt in dict.fromkeys([*assignable, *targets]):
            a = self.counts["n_assigned"][inside, :, self._k(tgt)]
            own = self.line_names.index(tgt)                 # the target's own single lines land on themselves
            if tgt in assignable:
                every = every + np.delete(a, own, axis=1).sum(axis=1)
            if tgt not in targets:
                continue
            if line is None:
                n = n + np.delete(a, own, axis=1).sum(axis=1)
            elif line != tgt:
                n = n + a[:, self.line_names.index(line)]
        catalog = self._sample()[inside] + every
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.z[inside], np.where(catalog > 0, n / catalog, np.nan)

    # ---- the secondary limit and the interloper fractions ----------------------------
    @property
    def has_scan(self) -> bool:
        """Was the secondary limit scanned (proposals made since 2026-11-10)?"""
        return "scan_window" in self.counts

    def _require_scan(self) -> None:
        if not self.has_scan:
            raise ValueError("this proposal has no scan of the secondary limit (it was made before 2026-11-10): make it again")

    @property
    def q_edges(self) -> np.ndarray:
        """Edges of the secondary-limit scan, as fractions of the primary limit (the proposal's own among them)."""
        self._require_scan()
        return scan_edges(**self.settings.get("scan", SCAN))

    @property
    def single_line(self) -> bool:
        """Made without a secondary limit: a proposal for single-line clean catalogs, without interlopers. Its sample is
        the galaxies with the target line above the primary limit (``line="all"``: every detected galaxy), and
        :meth:`make_catalog` gives each its true redshift: with one line per galaxy (the target line, even where
        another is brighter; ``line="all"``: the brightest), or, given a secondary limit, with its other lines above
        that limit."""
        return bool((self.settings.get("detection") or {}).get("single_line"))

    def _require_interlopers(self) -> None:
        if self.single_line:
            raise ValueError("this proposal has no secondary limit: it is for single-line clean catalogs, without "
                             "interlopers (proposal.make_catalog(out_dir=...)); for a catalog with interlopers, make the "
                             "proposal with interloper_fraction=... (a catalog by design) or with a secondary limit")

    @property
    def thresholds(self) -> str:
        """How the thresholds are given: ``"flux"`` (flux limits), ``"snr"`` (S/N in the Roman grism) or ``"ratio"``
        (a tabulated limit; its secondary limit a fraction of it)."""
        return {"constant": "flux", "roman_grism": "snr"}.get(self.settings["sensitivity"].get("type", "constant"), "ratio")

    @property
    def primary(self) -> float:
        """The primary threshold in the units of the thresholds: a flux limit, an S/N, or 1 for a tabulated limit."""
        s = self.settings["sensitivity"]
        return {"flux": lambda: float(s["flux_limit"]), "snr": lambda: float(s["snr"]), "ratio": lambda: 1.0}[self.thresholds]()

    @property
    def secondary(self) -> float:
        """The secondary limit of this proposal in the units of its thresholds: ``inf`` without one (one line per
        galaxy, :attr:`single_line`), the primary limit for a single tier."""
        if self.single_line:
            return float("inf")
        value = self.settings["sensitivity"].get(SECONDARY_KEYS[self.thresholds])
        return self.primary if value is None else float(value)

    def _secondary_name(self) -> str:
        return {"flux": "secondary flux limit", "snr": "secondary S/N", "ratio": "secondary limit"}[self.thresholds]

    def _fmt(self, value: float) -> str:
        """A secondary limit as text: ``6e-17``, ``1.5e-16`` (flux), ``2.5`` (S/N), ``none`` (no secondary limit)."""
        if value is None or not np.isfinite(value):
            return "none"
        return f"{value:.2g}" if self.thresholds == "flux" else f"{value:.3g}"

    def _at(self, value: float) -> str:
        """``at a secondary flux limit of 6e-17``, or ``without a secondary limit``."""
        return f"at a {self._secondary_name()} of {self._fmt(value)}" if np.isfinite(value) else "without a secondary limit"

    def _secondary_q(self, secondary_flux_limit=None, secondary_snr=None) -> tuple[float, float]:
        """``(q, value)`` of a secondary limit asked for (``secondary_flux_limit`` with flux limits, ``secondary_snr`` with
        S/N thresholds; neither: the proposal's own, ``inf`` for none): a fraction of the primary limit, and in the
        units of the thresholds."""
        given = {k: v for k, v in (("secondary_flux_limit", secondary_flux_limit), ("secondary_snr", secondary_snr))
                 if v is not None}
        if not given:
            return self.secondary / self.primary, self.secondary
        if len(given) > 1:
            raise ValueError("give secondary_flux_limit or secondary_snr, not both")
        (key, value), = given.items()
        if key != SECONDARY_KEYS[self.thresholds]:
            raise ValueError(f"the thresholds of this proposal are {self.thresholds} thresholds: give its secondary limit as "
                             f"{SECONDARY_KEYS[self.thresholds]}")
        value = float(value)
        low = float(self.q_edges[0]) if self.has_scan else 0.0
        q = value / self.primary
        if not (value > 0 and low * (1 - 1e-9) <= q <= 1 + 1e-9):
            raise ValueError(f"{key} must lie between {self._fmt(low * self.primary)} and the primary limit "
                             f"{self._fmt(self.primary)}; got {value:g}")
        return min(q, 1.0), value

    def _sensitivity_at(self, value: float) -> dict:
        """The sensitivity config of the proposal with the secondary limit ``value`` (units of the thresholds; ``inf``:
        none, a single tier)."""
        return {**self.settings["sensitivity"], SECONDARY_KEYS[self.thresholds]: float(value) if np.isfinite(value) else None}

    def _detection_at(self, q: float) -> dict:
        """The detection config at the secondary limit ``q``: one line per galaxy without one (the target line first),
        the two tiers with one; for a single-line proposal (:attr:`single_line`) with a secondary limit, only the
        galaxies with the target line above the primary limit, so that the sample stays the same."""
        det = dict(self.settings.get("detection") or {"type": "threshold"})
        det.pop("single_line", None)
        det.pop("primary_line", None)
        if not np.isfinite(q):
            det["single_line"] = True if self.all_lines else self.line
        elif self.single_line and not self.all_lines:
            det["primary_line"] = self.line
        return det

    def _thresholds_text(self, value: float) -> str:
        """The thresholds at the secondary limit ``value`` (units of the thresholds; ``inf``: none), as text."""
        from .report import symbols
        from .sensitivity import describe_sensitivity

        text = symbols(describe_sensitivity(self._sensitivity_at(value)))
        return text if np.isfinite(value) else f"{text}; one line per galaxy, no secondary limit"

    def _interp(self, hist, q) -> np.ndarray:
        """``hist`` (the scan bins on axis 0) summed over the values at most each of ``q``: exact on the edges of the scan,
        linear in log q between them. Shape ``(len(q),) + hist.shape[1:]``."""
        e = self.q_edges
        hist = np.asarray(hist, dtype=np.float64)
        q = np.atleast_1d(np.asarray(q, dtype=np.float64))
        c = np.cumsum(hist, axis=0)[: e.size]                                          # c[k]: values <= e[k]
        le = np.log10(e)
        x = np.log10(np.clip(q, e[0], e[-1]))
        k = np.clip(np.searchsorted(le, x, side="left"), 1, e.size - 1)
        w = ((x - le[k - 1]) / (le[k] - le[k - 1])).reshape((-1,) + (1,) * (c.ndim - 1))
        return c[k - 1] * (1.0 - w) + c[k] * w

    def _numbers(self, q, targets: Sequence[str] | None = None) -> tuple[np.ndarray, dict]:
        """At the secondary limits ``q`` (fractions of the primary limit, an array): the sample before interlopers and, per pair ``(line, taken for)`` toward ``targets`` (default: the
        :attr:`assignable` lines), ``(n, n_out)``: its potential interlopers and those of them that are not in the
        sample already (all for a target line; for ``line="all"`` those with a true redshift outside the range). Pairs
        without any single line are left out."""
        self._require_interlopers()
        self._require_scan()
        q = np.atleast_1d(np.asarray(q, dtype=np.float64))
        sw, st = self.counts["scan_window"], self.counts["scan_target"]
        if self.all_lines:
            n_sample = np.full(q.size, float(self.counts["n_det"][self.inside].sum()))
        else:
            k = self._k(self.line)
            n_sample = float(st[:, k].sum()) - self._interp(st[:, k], q)
        pairs = {}
        for t in (self.assignable if targets is None else targets):
            k, own = self._k(t), self.line_names.index(t)
            both = self._interp(sw[:, :, k, :].sum(axis=-1), q)
            out = self._interp(sw[:, :, k, 0], q) if self.all_lines else both
            every = sw[:, :, k, :].sum(axis=(0, -1))
            for i, x in enumerate(self.line_names):
                if i != own and every[i] > 0:
                    pairs[(x, t)] = (both[:, i], out[:, i])
        return n_sample, pairs

    def _scan_at(self, q) -> dict:
        q = np.atleast_1d(np.asarray(q, dtype=np.float64))
        n_sample, pairs = self._numbers(q)
        catalog = n_sample + sum((out for _, out in pairs.values()), np.zeros(q.size))
        with np.errstate(invalid="ignore", divide="ignore"):
            share = {p: np.where(catalog > 0, n / catalog, 0.0) for p, (n, _) in pairs.items()}
        return {"secondary": q * self.primary, "q": q, "n_sample": n_sample, "n": {p: n for p, (n, _) in pairs.items()},
                "share": share, "total": sum(share.values(), np.zeros(q.size))}

    def interloper_scan(self, secondary=None) -> dict:
        """The most the potential interlopers can make of the catalog, against the secondary limit: if every one of them
        were taken, the catalog would hold the sample plus all of them, and each pair ``(line, taken for)`` its share.
        A higher secondary limit detects fewer second lines, leaves more galaxies with a single line and so allows more
        interlopers, the most at the primary limit (a single tier).

        ``secondary``: secondary limits in the units of the thresholds (fluxes, or S/N); default every edge of the scan
        from where interlopers become possible (at most from 0.1 times the primary limit) up to the primary limit.
        Returns ``{"secondary", "q" (fractions of the primary limit), "n_sample", "n": {pair: potential interlopers},
        "share": {pair: share of the catalog}, "total"}``, arrays over the secondary limits."""
        self._require_interlopers()
        if secondary is None:
            e = self.q_edges
            q = e[e >= self._shown_low() * (1 - 1e-9)]
        else:
            q = np.atleast_1d(np.asarray(secondary, dtype=np.float64)) / self.primary
        return self._scan_at(q)

    def _expected(self, total=None) -> float:
        total = self.settings.get("interloper_fraction") if total is None else total
        if total is None:
            raise ValueError("give the expected total interloper fraction, or make the proposal with interloper_fraction=...")
        total = float(total)
        if not 0.0 < total < 1.0:
            raise ValueError(f"an interloper fraction is a share of the catalog, between 0 and 1; got {total:g}")
        return total

    def _lowest(self, feasible: np.ndarray) -> int | None:
        """Index of the lowest edge from which ``feasible`` holds up to the primary limit."""
        tail = np.logical_and.accumulate(np.asarray(feasible, dtype=bool)[::-1])[::-1]
        return int(np.argmax(tail)) if tail.any() else None

    def secondary_for(self, total: float | None = None) -> float | None:
        """The lowest secondary limit (units of the thresholds) above which the potential interlopers can make ``total``
        of the catalog (default: the proposal's ``interloper_fraction``); ``None`` when not even at the primary limit."""
        self._require_interlopers()
        total = self._expected(total)
        e = self.q_edges
        tot = self._scan_at(e)["total"]
        k = self._lowest(tot >= total)
        if k is None:
            return None
        if k == 0:
            return float(e[0] * self.primary)
        x0, x1, y0, y1 = np.log10(e[k - 1]), np.log10(e[k]), tot[k - 1], tot[k]
        x = x1 if y1 <= y0 else x0 + (total - y0) / (y1 - y0) * (x1 - x0)
        return float(10.0 ** min(max(x, x0), x1) * self.primary)

    def _parse_fractions(self, fractions) -> list[tuple[str, str, float]]:
        """``[(line, taken for, fraction)]`` of a table ``{taken for: {line: fraction}}``, zeros dropped."""
        if not isinstance(fractions, Mapping) or not all(isinstance(v, Mapping) for v in fractions.values()):
            raise TypeError("fractions is {line taken for: {interloper line: share of the catalog}}, e.g. "
                            "{'Ha': {'OIII_5007': 0.005, 'SIII_9532': 0.005}}")
        rows = []
        for t, row in fractions.items():
            if self.all_lines and t not in self.targets:
                raise KeyError(f"{t!r} is not inside the bandpass for this range; the lines of this proposal are {self.targets}")
            if not self.all_lines and t != self.line:
                raise KeyError(f"this proposal is for {self.line}: fractions={{{self.line!r}: {{line: fraction}}}}")
            for x, f in row.items():
                if x not in self.line_names:
                    raise KeyError(f"unknown line {x!r}; the lines are {self.line_names}")
                if x == t:
                    raise ValueError(f"{x} taken for itself is no interloper")
                f = float(f)
                if not (np.isfinite(f) and 0.0 <= f < 1.0):
                    raise ValueError(f"the fraction of {x} -> {t} must be a share of the catalog in [0, 1); got {f:g}")
                if f > 0:
                    rows.append((x, t, f))
        if not rows:
            raise ValueError("the table asks for no interlopers; for a clean catalog use make_catalog(confusion={}, ...)")
        return rows

    def design(self, fractions, secondary_flux_limit=None, secondary_snr=None) -> dict:
        """What a table of interloper fractions asks of the identification at a secondary limit (default: the
        proposal's): for each pair ``(line, taken for)``, the probability ``p`` that a
        single line of ``line`` inside the window of the other (where it gets a redshift inside the range) is taken for
        it, so that the interlopers of the pair make ``fraction`` of the catalog.

        ``fractions`` is ``{taken for: {line: fraction}}``: ``{"Ha": {"OIII_5007": 0.005, "SIII_9532": 0.005}}`` asks
        that the [O III] λ5007 and the [S III] λ9532 galaxies taken for Hα each make 0.5 % of the catalog (of the Hα
        sample for ``line="Ha"``; of every galaxy of the catalog for ``line="all"``). Lines not listed are identified
        correctly. With ``N`` the sample, ``n`` the potential interlopers of a pair and ``F`` the sum of the fractions,
        the catalog holds ``N / (1 - F)`` galaxies (for ``line="all"``, ``N / (1 - sum(fraction n_out / n))``, since an
        interloper whose true redshift is inside the range is in the sample already) and ``p = fraction x catalog / n``.
        Raises when a pair needs ``p > 1``, more interlopers than there are single lines, naming the secondary limit
        above which the table can be met.

        Returns ``{"fractions", "secondary", "q", "n_sample", "n_catalog", "total", "pairs": [{"line",
        "as", "fraction", "p", "n_potential", "n_expected", "window"}], "unlisted": [...]}``, the potential interlopers
        left out of the table in ``unlisted``. ``make_catalog(fractions=...)`` uses it."""
        self._require_interlopers()
        q, value = self._secondary_q(secondary_flux_limit, secondary_snr)
        return self._design(fractions, q, value)

    def _design(self, fractions, q: float, value: float) -> dict:
        from .confusion import WindowedConfusion

        rows = self._parse_fractions(fractions)
        targets = list(dict.fromkeys([*self.assignable, *(t for _, t, _ in rows)]))
        n_sample, pairs = self._numbers(np.array([q]), targets)
        n_sample = float(n_sample[0])
        n = {p: float(v[0][0]) for p, v in pairs.items()}
        out = {p: float(v[1][0]) for p, v in pairs.items()}
        where = self._at(value)
        empty = [(x, t) for x, t, _ in rows if n.get((x, t), 0.0) <= 0]
        if empty:
            raise ValueError("no single " + ", ".join(f"{x} line falls where {t} would be inside the range" for x, t in empty)
                             + f" {where}: no such interlopers can be made")
        total = sum(f for *_, f in rows)
        shrink = 1.0 - sum(f * out[(x, t)] / n[(x, t)] for x, t, f in rows)
        if total >= 1.0 or shrink <= 0:
            raise ValueError(f"the fractions add up to {total:g}: they are shares of the catalog, which also holds the sample")
        n_catalog = n_sample / shrink
        design_pairs = [{"line": x, "as": t, "fraction": f, "p": f * n_catalog / n[(x, t)], "n_potential": n[(x, t)],
                         "n_expected": f * n_catalog, "window": [float(w) for w in self.windows[t]]} for x, t, f in rows]
        short = [r for r in design_pairs if r["p"] > 1.0 + 1e-9]
        if short:
            k = self._lowest(self._feasible(rows))
            need = "; ".join(f"{r['line']} -> {r['as']} {100 * r['fraction']:.3g} % needs {_number(r['n_expected'])} of the "
                             f"{_number(r['n_potential'])} there are" for r in short)
            hint = (f"; the table can be met with a {self._secondary_name()} of at least "
                    f"{self._fmt(self.q_edges[k] * self.primary)}" if k is not None else
                    f"; not even with the {self._secondary_name()} at the primary limit")
            raise ValueError(f"too few single lines {where}: {need}{hint}, or ask for smaller fractions")
        for r in design_pairs:
            r["p"] = min(r["p"], 1.0)
        WindowedConfusion(self.line_names, design_pairs)               # refuses overlapping windows asking for p > 1
        listed = {(x, t) for x, t, _ in rows}
        unlisted = [{"line": x, "as": t, "n_potential": v} for (x, t), v in sorted(n.items(), key=lambda kv: -kv[1])
                    if (x, t) not in listed and t in self.assignable and v > 0 and self._major(v, v / max(n_sample, 1.0))]
        asked = {}
        for x, t, f in rows:
            asked.setdefault(t, {})[x] = f
        return {"fractions": asked, "thresholds": self.thresholds, "secondary": float(value), "q": float(q), "n_sample": n_sample, "n_catalog": n_catalog, "total": total,
                "pairs": design_pairs, "unlisted": unlisted}

    def _feasible(self, rows, q=None) -> np.ndarray:
        """At the secondary limits ``q`` (default: the edges of the scan): does every pair of ``rows``
        ``(line, taken for, fraction)`` have enough single lines?"""
        q = self.q_edges if q is None else np.atleast_1d(np.asarray(q, dtype=np.float64))
        targets = list(dict.fromkeys([*self.assignable, *(t for _, t, _ in rows)]))
        n_sample, pairs = self._numbers(q, targets)
        zero = (np.zeros(q.size), np.zeros(q.size))
        with np.errstate(invalid="ignore", divide="ignore"):
            shrink = 1.0 - sum(f * np.where(pairs.get((x, t), zero)[0] > 0, pairs.get((x, t), zero)[1] / pairs.get((x, t), zero)[0], 1.0)
                               for x, t, f in rows)
            n_catalog = np.where(shrink > 0, n_sample / shrink, np.inf)
        ok = shrink > 0
        for x, t, f in rows:
            n = pairs.get((x, t), zero)[0]
            ok &= (n > 0) & (f * n_catalog <= n * (1 + 1e-9))
        return ok

    def fraction_template(self, total: float | None = None, secondary_flux_limit=None, secondary_snr=None,
                          minor: bool = False) -> dict:
        """A table of interloper fractions for :meth:`make_catalog`, ``{taken for: {line: share of the catalog}}``: the
        expected ``total`` (default: the ``interloper_fraction`` of the proposal) shared among the potential
        interlopers at the secondary limit (default: the proposal's) in proportion to their numbers, so that every one
        of them is taken with the same probability. The major pairs only, unless ``minor``. Raises when ``total`` is
        more than they can make there, naming the lowest secondary limit that allows it."""
        self._require_interlopers()
        total = self._expected(total)
        q, value = self._secondary_q(secondary_flux_limit, secondary_snr)
        n_sample, pairs = self._numbers(np.array([q]))
        n_sample = float(n_sample[0])
        use = {p: (float(n[0]), float(out[0])) for p, (n, out) in pairs.items()
               if n[0] > 0 and (minor or self._major(n[0], n[0] / max(n_sample, 1.0)))}
        where = self._at(value)
        if not use:
            raise ValueError(f"no potential interlopers{'' if minor else ' above the minor-line cut'} {where}")
        most = sum(n for n, _ in use.values()) / (n_sample + sum(out for _, out in use.values()))
        if total > most:
            lowest = self.secondary_for(total)
            allows = (f"; not even the {self._secondary_name()} at the primary limit allows it" if lowest is None else
                      f"; a {self._secondary_name()} of at least {self._fmt(lowest)} allows it")
            raise ValueError(f"the potential interlopers can make at most {100 * most:.3g} % of the catalog {where}, less than "
                             f"{100 * total:.3g} %{allows}")
        whole = sum(n for n, _ in use.values())
        rows = {}
        for (x, t), (n, _) in sorted(use.items(), key=lambda kv: -kv[1][0]):
            rows.setdefault(t, {})[x] = float(f"{total * n / whole:.4g}")
        return rows

    def _shown_low(self) -> float:
        """The lowest secondary limit the table and the figure show, as a fraction of the primary limit: 0.1, or the last
        edge without any possible interloper when the potential interlopers all keep a second line below a higher one."""
        e = self.q_edges
        e = e[e >= SCAN_SHOWN * (1 - 1e-9)]
        nonzero = np.flatnonzero(self._scan_at(e)["total"] > 0)
        return SCAN_SHOWN if not nonzero.size or nonzero[0] == 0 else float(e[nonzero[0] - 1])

    def _shown_values(self) -> list[float]:
        """Secondary limits for the table of the summary: round values from the lowest shown one up to the primary
        limit, with the proposal's own and those of ``secondary_ratios``, at most nine."""
        hi = self.primary
        lo = self._shown_low() * hi
        if self.thresholds == "flux":
            mantissas = (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9) if np.log10(hi / lo) < 0.65 else (1, 1.5, 2, 3, 4, 6, 8)
            decades = range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1)
            values = sorted({float(f"{m * 10.0 ** d:.3g}") for d in decades for m in mantissas})
            lo = max([v for v in values if v <= lo * (1 + 1e-9)], default=lo)          # a round value at or below
        else:
            step = next((st for st in (0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0) if (hi - lo) / st <= 7), 10.0)
            start = np.floor(lo / step + 1e-9) * step
            if start > 0:
                lo = float(start)
            else:                                      # a round value below: 0.527 -> 0.5
                e = np.floor(np.log10(lo))
                lo = float(f"{max(m for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8) if m <= lo / 10 ** e + 1e-9) * 10 ** e:.3g}")
            values = [lo] + [float(f"{v:.6g}") for v in np.arange(np.ceil(lo / step + 1e-9) * step, hi + 1e-9, step)]
        inside = lambda v: lo * (1 - 1e-9) <= v <= hi * (1 + 1e-9)
        required = {lo, float(f"{hi:.6g}"), self.secondary}
        required |= {float(f"{r * hi:.6g}") for r in self.settings.get("secondary_ratios") or [] if inside(r * hi)}
        same = lambda a, b: np.isclose(a, b, rtol=1e-6, atol=0.0)
        values = sorted(required | {v for v in values if inside(v) and not any(same(v, r) for r in required)})
        while len(values) > 9:                         # drop the round value whose neighbours are closest together
            free = [i for i in range(1, len(values) - 1) if values[i] not in required]
            if not free:
                break
            values.pop(free[int(np.argmin([np.log(values[i + 1] / values[i - 1]) for i in free]))])
        return values

    def _major_scan_pairs(self, top: dict) -> tuple[list, list]:
        """The pairs of a scan (``_scan_at`` at the primary limit, where single lines are most) that are major there, and
        the others."""
        n_top, sample_top = {p: float(v[0]) for p, v in top["n"].items()}, float(top["n_sample"][0])
        order = sorted(n_top, key=lambda p: -n_top[p])
        shown = [p for p in order if self._major(n_top[p], n_top[p] / max(sample_top, 1.0))]
        return shown, [p for p in order if p not in shown]

    def _scan_section(self) -> list[str]:
        """The summary's table of the most the potential interlopers can make against the secondary limit."""
        from .report import keep, label, notes, percent, section, table

        values = self._shown_values()
        scan = self.interloper_scan(values)
        shown, rest = self._major_scan_pairs(self._scan_at(np.array([1.0])))
        what = "that line" if self.all_lines else label(self.line)
        out = [section("Interlopers against the secondary limit")]
        out += notes(f"if every potential interloper were taken for {what}: the share of the catalog each would make, in %, "
                     "the most any identification gives. Other lines are detected above the secondary limit, so a higher "
                     "one leaves more galaxies with a single line.")
        if self.thresholds == "flux":
            unit = 10.0 ** np.floor(np.log10(min(values)))
            head = f"secondary F [{unit:.0e}]".replace("e-", "e−")
            heads = [f"{v / unit:.3g}" for v in values]
        else:
            head = "secondary S/N" if self.thresholds == "snr" else "secondary / primary"
            heads = [self._fmt(v) for v in values]
        heads = [h + ("⁴" if np.isclose(v, self.secondary, rtol=1e-6, atol=0.0) else "") for h, v in zip(heads, values)]

        def cells(shares):
            return ["0" if v == 0 else "<0.01" if v < 5e-5 else f"{100 * v:.2f}" for v in shares]

        name = (lambda p: f"{p[0]} → {p[1]}") if self.all_lines else (lambda p: p[0])
        body = [[name(p)] + cells(scan["share"][p]) for p in shown]
        if rest:
            body.append(["other lines"] + cells(sum(scan["share"][p] for p in rest)))
        body.append(["total"] + cells(scan["total"]))
        out += table([(head, "<")] + [(h, ">") for h in heads], body)
        primary = {"flux": f"F > {self.primary:.2g}", "snr": f"S/N ≥ {self.primary:g}"}.get(self.thresholds, "the tabulated limit")
        out += notes(f"this proposal's {self._secondary_name()}, {self._fmt(self.secondary)} (the tables above); primary "
                     f"{primary}", mark="⁴ ")
        if rest:
            out += notes("other lines: " + ", ".join(keep(name(p)) for p in rest))
        low = self._shown_low()
        if low > SCAN_SHOWN:
            e = self.q_edges
            onset = e[np.searchsorted(e, low, side="right")] * self.primary
            out += notes(f"below a {self._secondary_name()} of {self._fmt(onset)} every potential interloper keeps a second "
                         "line: none can be made")
        expected = self.settings.get("interloper_fraction")
        if expected:
            lowest = self.secondary_for(expected)
            here = float(self._scan_at(self.secondary / self.primary)["total"][0])
            mine = f"at this proposal's {self._fmt(self.secondary)}"
            if lowest is None:
                text = f"more than the potential interlopers can make, even with the {self._secondary_name()} at the primary limit"
            elif lowest <= self.q_edges[0] * self.primary * (1 + 1e-9):
                text = f"possible at every {self._secondary_name()} (scanned down to {self._fmt(lowest)})"
            else:
                text = f"possible from a {self._secondary_name()} of {self._fmt(lowest)} up"
            out += notes(f"expected {percent(expected)} of interlopers in all: {text}; {mine} at most {percent(here)}")
        key = SECONDARY_KEYS[self.thresholds]
        target = self.assignable[0] if self.assignable else (self.targets[0] if self.targets else "Ha")
        out += [f"  next:  proposal.make_catalog(fractions={{'{target}': {{...}}}}, {key}=…, out_dir=…)",
                f"         proposal.fraction_template(total, {key}=…) shares a total in proportion"]
        return out

    def single_line_fraction(self, ratio: float, line: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Against true redshift inside the range: the share of the galaxies with ``line`` above the primary limit
        (default: the target line; for ``line="all"`` every detected galaxy, whatever its line) that are true single-line
        galaxies at a secondary limit of ``ratio`` times the primary one, i.e. without another line above it.
        ``ratio`` is one of ``settings["secondary_ratios"]`` (0.3, 0.4 and 0.5 by default)."""
        if "nq_single" not in self.counts:
            raise ValueError("this proposal has no single-line fractions at fixed secondary limits; make it again")
        ratios = list(self.settings.get("secondary_ratios") or [])
        match = [j for j, r in enumerate(ratios) if np.isclose(r, ratio, rtol=1e-9, atol=0.0)]
        if not match:
            raise KeyError(f"the proposal counted the secondary limits of {ratios} times the primary one, not {ratio}")
        inside = self.inside
        single = self.counts["nq_single"][match[0]][inside]
        if line is None and self.all_lines:
            num, den = single.sum(axis=1), self.counts["n_det"][inside]
        else:
            i = self.line_names.index(line or self.line)
            num, den = single[:, i], self.counts["n_primary"][inside, i]
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.z[inside], np.where(den > 0, num / den, np.nan)

    def _single_line_section(self) -> list[str]:
        """The summary's table of the true single-line fraction against redshift at the secondary limits of
        ``secondary_ratios`` (for a proposal without a secondary limit)."""
        from .report import count, label, notes, section, table

        ratios = list(self.settings.get("secondary_ratios") or [])
        if not ratios or "nq_single" not in self.counts:
            return []
        inside = self.inside
        lo_edges, hi_edges = self.z_edges[:-1][inside], self.z_edges[1:][inside]
        target = None if self.all_lines else self.line
        if target is None:
            den = self.counts["n_det"][inside]
            nums = [self.counts["nq_single"][j][inside].sum(axis=1) for j in range(len(ratios))]
        else:
            i = self.line_names.index(target)
            den = self.counts["n_primary"][inside, i]
            nums = [self.counts["nq_single"][j][inside, i] for j in range(len(ratios))]
        n_bins = int(inside.sum())
        step = max(1, int(np.ceil(n_bins / 20)))                   # at most 20 rows

        def pct(a, b):
            return f"{100 * a / b:.2f}" if b > 0 else "–"

        rows = []
        for a in range(0, n_bins, step):
            b = min(a + step, n_bins)
            d = int(den[a:b].sum())
            rows.append([f"{lo_edges[a]:.2f}–{hi_edges[b - 1]:.2f}", count(d)] + [pct(int(n[a:b].sum()), d) for n in nums])
        rows.append(["all", count(int(den.sum()))] + [pct(int(n.sum()), int(den.sum())) for n in nums])
        values = [r * self.primary for r in ratios]
        heads = ([f"{v:.2g}" for v in values] if self.thresholds == "flux" else
                 [f"S/N {v:g}" for v in values] if self.thresholds == "snr" else [f"{r:g}×" for r in ratios])
        who = "detected galaxies" if target is None else f"{label(target)} galaxies"
        out = [section(f"{'Single-line' if target is None else label(target) + ' single-line'} galaxies against redshift")]
        listed = ", ".join(f"{r:g}" for r in ratios)
        out += notes(f"without a secondary limit (this proposal) every one of the {who} is a single-line galaxy. With one, "
                     "only those without another line above it are: their share in %, for secondary limits of "
                     f"{listed} times the primary one")
        out += table([("z", "<"), (f"{'galaxies' if target is None else label(target) + ' galaxies'}¹", ">")]
                     + [(h, ">") for h in heads], rows)
        what = "detected" if target is None else f"with {label(target)} above the primary limit"
        out += notes(f"galaxies {what} at that true redshift", mark="¹ ")
        key = SECONDARY_KEYS[self.thresholds]
        out += ["", "  next:  proposal.make_catalog(out_dir=…): one line per galaxy, every galaxy single-line",
                f"         proposal.make_catalog({key}=…, out_dir=…): with the other lines above it"]
        out += notes("both clean, every redshift the true one, and with the same galaxies", indent=9)
        return out

    def _design_text(self, d: dict) -> str:
        """The design of a catalog (:meth:`design`) as text."""
        from .report import count, fields, keep, label, notes, percent, table, title, zfmt

        lo, hi = self.z_range
        value = d["secondary"]
        out = [title(f"Interlopers by design · {'every galaxy' if self.all_lines else label(self.line)} at "
                     f"{zfmt(lo)} ≤ z ≤ {zfmt(hi)}")]
        sample = "detected galaxies" if self.all_lines else f"galaxies with {label(self.line)} detected"
        out += fields([("thresholds", self._thresholds_text(value)),
                       ("sample", f"{count(round(d['n_sample']))} {sample} inside the range"),
                       ("catalog", f"{_number(d['n_catalog'])} galaxies expected, {percent(d['total'])} of them "
                                   "interlopers")])
        cols = [("line", "<")] + ([("taken for", "<")] if self.all_lines else []) + [
            ("fraction¹", ">"), ("potential²", ">"), ("P³", ">"), ("expected", ">")]
        body = [[r["line"]] + ([r["as"]] if self.all_lines else []) + [
            percent(r["fraction"], 3), _number(r["n_potential"]), f"{r['p']:.4f}", _number(r["n_expected"])]
            for r in d["pairs"]]
        out += [""] + table(cols, body)
        what = "that line" if self.all_lines else label(self.line)
        out += notes("asked: the share of the catalog", mark="¹ ")
        out += notes(f"single lines where {what} would be inside the range, {self._at(value)}", mark="² ")
        out += notes(f"the probability that such a single line is taken for {what}; elsewhere, and for the lines not listed, "
                     "the identification is correct", mark="³ ")
        if d["unlisted"]:
            out += notes("potential interlopers not listed, so identified correctly: " + ", ".join(
                keep(f"{u['line']}{' → ' + u['as'] if self.all_lines else ''} ({_number(u['n_potential'])})")
                for u in d["unlisted"]))
        return "\n".join(out)

    @staticmethod
    def _made(cat, design: dict) -> list[dict]:
        """The interlopers of each pair of a design in the catalog made from it, from its line report."""
        report = cat.info.get("line_report") or {}
        n = {(r["line_true"], r["line_assigned"]): int(r.get("n", 0)) for r in report.get("interlopers", [])}
        size = len(cat)
        return [{"line": r["line"], "as": r["as"], "asked": r["fraction"], "n": n.get((r["line"], r["as"]), 0),
                 "made": n.get((r["line"], r["as"]), 0) / size if size else 0.0, "n_expected": r["n_expected"]}
                for r in design["pairs"]]

    def _made_text(self, design: dict, made: list[dict], size: int) -> str:
        from .report import count, notes, percent, section, table

        out = [section("Interlopers by design: asked and made")]
        cols = [("line", "<")] + ([("taken for", "<")] if self.all_lines else []) + [
            ("asked", ">"), ("made", ">"), ("interlopers", ">"), ("expected", ">")]
        out += table(cols, [[m["line"]] + ([m["as"]] if self.all_lines else []) + [
            percent(m["asked"], 3), percent(m["made"], 3), count(m["n"]), _number(m["n_expected"])] for m in made])
        out += notes(f"{count(size)} galaxies in the catalog ({_number(design['n_catalog'])} expected); each pair's "
                     "interlopers are drawn one by one, so their number scatters by about its square root")
        return "\n".join(out)

    def plot_secondary(self, ax=None):
        """The most the potential interlopers can make of the catalog against the secondary limit (every one taken; the
        share of the catalog): one curve per major pair, coloured by its true line (line style: the line it is taken
        for, ``line="all"``), the other lines together, and the total in black. The expected total interloper fraction
        of the proposal is the dashed line, the lowest secondary limit that allows it the dot, and the proposal's own
        secondary limit the dotted line. Returns the figure."""
        import matplotlib.pyplot as plt

        from .plotting import TEXT_PRIMARY, TEXT_SECONDARY, feature_color, feature_label
        from .report import zfmt

        self._require_interlopers()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
        fig = ax.figure
        scan = self.interloper_scan()
        x = scan["secondary"]
        shown, rest = self._major_scan_pairs(self._scan_at(np.array([1.0])))
        style = {t: ("-", "--", ":", "-.")[j % 4] for j, t in enumerate(dict.fromkeys(t for _, t in shown))}
        for xl, t in shown:
            name = f"{feature_label(xl)} as {feature_label(t)}" if self.all_lines else feature_label(xl)
            ax.plot(x, 100 * scan["share"][(xl, t)], color=feature_color(xl), ls=style[t], lw=1.8, label=name)
        if rest:
            ax.plot(x, 100 * sum(scan["share"][p] for p in rest), color=TEXT_SECONDARY, ls=":", lw=1.4, label="other lines")
        ax.plot(x, 100 * scan["total"], color=TEXT_PRIMARY, lw=2.4, label="total")
        expected = self.settings.get("interloper_fraction")
        if expected:
            ax.axhline(100 * expected, color=TEXT_SECONDARY, ls="--", lw=1.2, zorder=1)
            lowest = self.secondary_for(expected)
            text = f"expected {100 * expected:g} %"
            if lowest is not None and np.isfinite(lowest) and lowest > x[0]:
                ax.plot([lowest], [100 * expected], "o", color=TEXT_PRIMARY, ms=7, zorder=5)
                text += f": {self._secondary_name()} ≥ {self._fmt(lowest)}"
            ax.annotate(text, (x[0], 100 * expected), xytext=(4, 4), textcoords="offset points", fontsize=9,
                        color=TEXT_SECONDARY, va="bottom")
        if x[0] <= self.secondary <= x[-1]:
            ax.axvline(self.secondary, color=TEXT_SECONDARY, ls=":", lw=1.2, zorder=0)
            ax.annotate(f"this proposal, {self._fmt(self.secondary)}", (self.secondary, 0.5),
                        xycoords=("data", "axes fraction"), xytext=(-3, 0), textcoords="offset points", fontsize=8,
                        color=TEXT_SECONDARY, rotation=90, ha="right", va="center")
        if self.thresholds == "flux":
            from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

            keep = {r * self.primary for r in self.settings.get("secondary_ratios") or []}
            ticks = []
            for v in sorted(self._shown_values(), key=lambda v: (not any(np.isclose(v, k, rtol=1e-6, atol=0.0) for k in keep), v)):
                if all(abs(np.log10(v / t)) > 0.07 for t in ticks):       # labels at least 0.07 dex apart
                    ticks.append(v)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator(sorted(ticks)))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2g}"))
            ax.set_xlabel("secondary flux limit [erg s⁻¹ cm⁻²]")
        else:
            ax.set_xlabel("secondary S/N" if self.thresholds == "snr" else "secondary limit / primary limit")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(bottom=0)
        ax.set_ylabel("at most [% of the catalog]")
        lo, hi = self.z_range
        what = "every galaxy" if self.all_lines else feature_label(self.line)
        primary = {"flux": f"F > {self.primary:.2g}", "snr": f"S/N ≥ {self.primary:g}"}.get(self.thresholds, "")
        ax.set_title(f"{what} at {zfmt(lo)} < z < {zfmt(hi)}, {primary}: interlopers if every one is taken", loc="left")
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.93))
        return fig

    # ---- for the second step -------------------------------------------------------
    def confusion_template(self, p_interloper: float = 0.0, p_target: float = 0.0, minor: bool = False) -> dict:
        """The rows of a confusion table for this catalog, to fill in and pass to :meth:`make_catalog`:
        ``{interloper line: {target: p_interloper}}`` for the major potential interlopers (the probability that such
        a single line is taken for the target line, which puts it in the catalog) and ``{target: {line: p_target}}``
        (the probability that a single target line is taken for that line, which loses it). ``minor=True`` adds the
        minor lines. Lines not listed are identified correctly; ``"fail"`` adds a probability of no redshift.

        For ``line="all"`` the targets are the :attr:`assignable` lines. One entry can then be both kinds (a line
        taken for another moves galaxies of other redshifts into the range and galaxies of the range out of it, e.g.
        Hα and [O III] λ5007 taken for each other); such an entry gets ``p_interloper``. A single line is taken only
        for an assignable line, so the loss entries toward the other lines are left out: in a catalog of every galaxy
        they would also bring in interlopers of a line the proposal rules out (a confusion table does not depend on
        the wavelength; e.g. Hα taken for [O II], meant to lose Hα at 1.6 < z < 1.94, also puts the Hα of
        0.52 < z < 0.70 at 1.68 < z < 2.0 as [O II])."""
        self._require_interlopers()
        pairs = [(x, t) for t in self.assignable for x, v in self.interlopers(t).items() if v["major"] or minor]
        assignable = set(self.assignable)
        rows = {}
        for x, t in pairs:                     # a single target line taken for x: it leaves the catalog
            if self.all_lines and x not in assignable:
                continue
            rows.setdefault(t, {})[x] = float(p_target)
        for x, t in pairs:                     # x taken for the target: an interloper
            rows.setdefault(x, {})[t] = float(p_interloper)
        return rows

    def default_name(self) -> str:
        lo, hi = self.z_range
        fmt = lambda v: f"{v:g}".replace(".", "p") if "." in f"{v:g}" else f"{v:g}p0"
        return f"{self.line}_z{fmt(lo)}_{fmt(hi)}"

    def make_catalog(self, confusion=None, out_dir=None, name: str | None = None, columns: Sequence[str] | str | None = None,
                     n_randoms: int = 0, seed: int | None = None,
                     n_workers: int | None = None, verbose: bool = True, *, fractions=None, secondary_flux_limit=None,
                     secondary_snr=None):
        """Step 2: the catalog, with the thresholds, line rescaling and doublet rule of the proposal and the given
        identification of the single-line galaxies, written to ``out_dir``.

        Parameters
        ----------
        confusion : dict, str or LineFractionTable
            How the single-line galaxies are identified: a confusion table ``{true line: {assigned line: p}}``
            (:meth:`confusion_template` lists the entries that matter; lines not listed are identified correctly;
            ``{}`` = all correctly, a clean catalog), ``"empirical"`` (the line mix of the mock at the same thresholds,
            rescaling and doublet rule; one more pass), a :class:`~romanmock.confusion.LineFractionTable` or any
            confusion config of :meth:`BaseCatalog.make_catalog`. Give this or ``fractions``; a proposal without a
            secondary limit (:attr:`single_line`) takes neither: its catalogs are clean.
        fractions : dict
            A catalog designed by its interloper fractions instead: ``{taken for: {line: share of the catalog}}``,
            e.g. ``{"Ha": {"OIII_5007": 0.005, "SIII_9532": 0.005}}``, the [O III] λ5007 and [S III] λ9532 galaxies
            taken for Hα each 0.5 % of the catalog (:meth:`design` says how; :meth:`fraction_template` shares a total
            in proportion). Only the single lines that would get a redshift inside the range are taken, with the
            probability of their pair; every other galaxy keeps its line. The design and the fractions made are printed
            and kept in the summary file.
        secondary_flux_limit, secondary_snr : float
            The secondary limit of the catalog, as a flux limit (a proposal with flux limits) or an S/N (S/N
            thresholds); default: the proposal's. A galaxy is then a single-line galaxy when no other line is above
            it. The interloper scan of the summary shows what each secondary limit allows. For a proposal made without
            a secondary limit (:attr:`single_line`), leaving it out makes the single-line clean catalog: one line per
            galaxy, every galaxy with the target line above the primary limit a single-line galaxy (``id_outcome``
            2); given one, the same galaxies with their other lines above it, the single-line ones (2) those without
            another, the others multi-line (1).
        out_dir : path
            Gets ``<name>.hdf5`` (the catalog, :meth:`Catalog.read`), ``<name>_summary.json`` (the settings, the
            proposal's numbers, the outcome counts and the line report), ``<name>_proposal.npz`` (this proposal,
            :meth:`Proposal.load`) and, with ``n_randoms``, ``<name>_random.hdf5``.
        name : str
            Default ``<line>_z<lo>_<hi>``, e.g. ``Ha_z1p0_1p2`` (``all_z1p0_1p2`` for ``line="all"``).
        columns : list of str or "all"
            Default :data:`CATALOG_COLUMNS`.
        n_randoms : int
            Also the random catalog of the whole catalog, ``n_randoms`` times as many points
            (:meth:`Catalog.make_random`); a part of the catalog gets its own with
            ``cat.subset("OIII_5007_as_Ha").make_random(n_times, write_to=...)``.
        seed : int
            Default: the seed of the proposal.

        The catalog keeps the galaxies in which the target line is identified (``membership="detected"``) with an
        assigned redshift inside the range: the galaxies with a correct redshift and the interlopers taken for the
        target line (``id_outcome`` 3). With ``line="all"`` it keeps every galaxy with an assigned redshift inside
        the range, whichever line gave it (``lines=None``). A single-line clean catalog keeps the galaxies with the
        target line above the primary limit (``line="all"``: every detected galaxy) at a true redshift inside the
        range.
        """
        from .catalog import _json_default

        if self.single_line:
            if fractions is not None or not (confusion is None or (isinstance(confusion, Mapping) and not confusion)):
                raise ValueError("this proposal has no secondary limit: it makes single-line clean catalogs, without "
                                 "interlopers, so leave confusion and fractions out; for a catalog with interlopers, make "
                                 "the proposal with interloper_fraction=... (a catalog by design) or with a secondary limit")
            confusion = {}
        elif (confusion is None) == (fractions is None):
            raise ValueError("give the identification of the single-line galaxies, one of: fractions={line taken for: "
                             "{interloper line: share of the catalog}} (a catalog by design; see design() and "
                             "fraction_template()), or confusion= a confusion table (see confusion_template()), 'empirical', "
                             "a LineFractionTable, or {} for a clean catalog")
        if out_dir is None:
            raise ValueError("give out_dir, the directory the catalog is written to")
        s = self.settings
        q, value = self._secondary_q(secondary_flux_limit, secondary_snr)
        changed = secondary_flux_limit is not None or secondary_snr is not None
        sensitivity = self._sensitivity_at(value) if changed else s["sensitivity"]
        detection = self._detection_at(q) if changed else s["detection"]
        design = None
        if fractions is not None:
            design = self._design(fractions, q, value)
            confusion = {"type": "windowed", "pairs": [{k: r[k] for k in ("line", "as", "window", "p")} for r in design["pairs"]],
                         "meta": {"fractions": design["fractions"], SECONDARY_KEYS[self.thresholds]: design["secondary"]}}
            if verbose:
                print(self._design_text(design) + "\n", flush=True)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = name or self.default_name()
        path = out_dir / f"{name}.hdf5"
        cat = self.base.make_catalog(sensitivity=sensitivity, detection=detection, rescaling=s["rescaling"],
                                     doublets=s["doublets"], confusion=confusion,
                                     lines=None if self.all_lines else [s["line"]], z_range=tuple(s["z_range"]),
                                     z_column="redshift_assigned", columns=CATALOG_COLUMNS if columns is None else columns,
                                     seed=s["seed"] if seed is None else seed, n_workers=n_workers, write_to=path, verbose=verbose)
        self.save(out_dir / f"{name}_proposal.npz")
        proposal = {"settings": self.settings, "n_sample": self.n_sample}
        proposal.update({} if self.single_line else {"single_lines_inside": self.single_lines_inside(),
                                                      "interlopers": self.interlopers()})
        summary = {"catalog": str(path), "n_galaxies": len(cat), "outcomes": cat.outcomes() if "id_outcome" in cat else None,
                   "settings": cat.settings, "line_report": cat.info.get("line_report"), "area_deg2": cat.area_deg2,
                   "proposal": proposal, "written": _dt.datetime.now().isoformat(timespec="seconds")}
        if design is not None:
            made = self._made(cat, design)
            summary["design"] = {**design, "made": made}
            if verbose:
                print("\n" + self._made_text(design, made, len(cat)), flush=True)
        (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=1, default=_json_default))
        if n_randoms:
            cat.make_random(n_times=int(n_randoms), seed=s["seed"] if seed is None else seed,
                            write_to=out_dir / f"{name}_random.hdf5", verbose=verbose)
        if verbose:
            written = [path.name, f"{name}_summary.json", f"{name}_proposal.npz"] + ([f"{name}_random.hdf5"] if n_randoms else [])
            print(f"\n[romanmock] wrote to {out_dir}/\n            {', '.join(written)}", flush=True)
        return cat

    # ---- text and figure -----------------------------------------------------------
    def summary(self) -> str:
        """The proposal as text: its settings and sample, then tables of the single-line galaxies inside the range and
        of the potential interlopers, with notes on what the columns mean. A single-line proposal (:attr:`single_line`)
        has no interlopers: the true single-line share against redshift instead."""
        from .report import count, fields, keep, label, notes, percent, section, span, symbols, table, title, zfmt
        from .sensitivity import describe_sensitivity

        s = self.settings
        lo, hi = self.z_range
        zl, zh = zfmt(lo), zfmt(hi)
        out = [title(f"Proposal · {'every galaxy' if self.all_lines else label(self.line)} at {zl} ≤ z ≤ {zh}")]
        merged = None
        if "n_doublet_both" in self.counts:
            both = int(self.counts["n_doublet_both"].sum())
            if both:
                merged = (f"; merged in {100 * int(self.counts['n_doublet_merged'].sum()) / both:.0f} % of the detected galaxies "
                          "with both lines in the bandpass")
        per = f" ({self.n_sample / self.area_deg2:,.0f} per deg²)" if self.area_deg2 else ""
        sample = "detected galaxies" if self.all_lines else f"galaxies with {label(self.line)} detected"
        expected = s.get("interloper_fraction")
        thresholds = symbols(describe_sensitivity(s["sensitivity"])) + ("; one line per galaxy: no secondary limit"
                                                                        if self.single_line else "")
        out += fields([("thresholds", thresholds),
                       ("line fluxes", f"rescaled ({s['rescaling']['type']})" if s.get("rescaling") else "as stored"),
                       ("doublet", _describe_doublets(s.get("doublets")) + (merged or "")),
                       ("sample", f"{count(self.n_sample)} {sample} at {zl} ≤ z < {zh}{per}"),
                       ("interlopers", "none: a single-line clean catalog, every redshift the true one" if self.single_line
                        else None if not expected else f"{percent(expected)} of the catalog expected in all")])
        if self.single_line:
            return "\n".join(out + [""] + self._single_line_section())

        inside = self.single_lines_inside()
        n_det = int(self.counts["n_det"][self.inside].sum())
        n_single = sum(v["n"] for v in inside.values())
        rows = [["all", count(n_single), percent(n_single / max(n_det, 1))]]
        rows += [[k, count(v["n"]), percent(v["share"])] for k, v in sorted(inside.items(), key=lambda kv: -kv[1]["n"])
                 if v["major"]]
        out += ["", section("Single-line galaxies inside the range")]
        out += table([("line", "<"), ("galaxies", ">"), ("share¹", ">")], rows)
        out += notes(f"of the {count(n_det)} galaxies detected at {zl} ≤ z < {zh} (true redshift)", mark="¹ ")
        minor = [keep(f"{k} ({v['n']:,})") for k, v in inside.items() if not v["major"]]
        if minor:
            out += notes("minor, left out: " + ", ".join(minor))

        assignable = self.assignable
        out += ["", section("Potential interlopers")]
        if self.all_lines:
            out += notes("single lines that, taken for a line that is single inside the range "
                         f"({', '.join(assignable) or 'none'}), would get a redshift inside it")
        else:
            w0, w1 = self.window
            out += notes(f"single lines at {w0:.0f}–{w1:.0f} Å, where {label(self.line)} would be at {zl} ≤ z ≤ {zh}")
        rows = [(x, t, v) for t in assignable for x, v in self.interlopers(t).items()]
        major = sorted([r for r in rows if r[2]["major"]], key=lambda r: -r[2]["n"])
        if major:
            cols = [("line", "<")] + ([("taken for", "<")] if self.all_lines else []) + [
                ("true z", "<"), ("galaxies", ">"), ("per deg²", ">"), ("single-line²", ">"), ("at most³", ">")]
            body = [[x] + ([t] if self.all_lines else []) + [
                span(*v["z_true"]), count(v["n"]), "–" if v["per_deg2"] is None else f"{v['per_deg2']:.1f}",
                percent(v["share"]), percent(v["at_most"])] for x, t, v in major]
            out += table(cols, body)
            out += notes("of the detected galaxies at that true redshift, the share whose only line is this one", mark="² ")
            out += notes("their share of the catalog if every potential interloper were taken (the sample plus all of "
                         "them): the most any identification gives", mark="³ ")
        else:
            out += notes("none above the minor-line cut")
        minor = [keep(f"{x}{' as ' + t if self.all_lines else ''} ({v['n']:,})") for x, t, v in rows if not v["major"]]
        if minor:
            out += notes(f"minor, left out (fewer than {s['min_count']} galaxies or {100 * s['min_share']:g} % of the sample): "
                         + ", ".join(minor))
        if self.all_lines:
            rare = [t for t in self.targets if t not in assignable]
            if rare:
                out += notes("not taken for (in the bandpass, but rarely a single line inside the range): " + ", ".join(rare))
        if self.has_scan:
            out += [""] + self._scan_section()
        return "\n".join(out)

    def __repr__(self) -> str:
        from .report import zfmt

        lo, hi = self.z_range
        head = f"<Proposal: {'every galaxy' if self.all_lines else self.line} at {zfmt(lo)}–{zfmt(hi)}, {self.n_sample:,} galaxies"
        if self.single_line:
            return head + "; single-line clean catalogs, no interlopers>"
        pairs = [f"{x} as {t}" for x, t in self.major_pairs] if self.all_lines else self.major_interlopers
        return head + f"; potential interlopers: {', '.join(pairs) or 'none'}>"

    def plot(self, fig=None):
        """Three panels: (a) the single-line fraction inside the range against true redshift, all lines and per line;
        (b) the single-line fraction of each potential interloper against its own redshift; (c) the potential
        interlopers against the redshift they would get, as a share of the catalog there if every one were taken. Only
        the lines that are not minor are drawn. For ``line="all"`` an interloper is a pair, its colour the true line and
        its line style the line it would be taken for. Panel (b) gets a log axis when its curves differ by more than 20
        times in height. :meth:`plot_secondary` draws the most the interlopers can make against the secondary limit.
        A single-line proposal (:attr:`single_line`) has one panel: the true single-line share of its sample against
        redshift at each of ``secondary_ratios``. Returns the figure."""
        import matplotlib.pyplot as plt

        from .plotting import TEXT_PRIMARY, feature_color, feature_label

        lo, hi = self.z_range
        z, f = self.fraction_inside()
        mark = {"marker": "o", "ms": 4} if z.size < 5 else {}            # a short range: a line alone would not show
        if self.single_line:
            if fig is None:
                fig = plt.figure(figsize=(8.0, 4.8), constrained_layout=True)
            ax = fig.subplots()
            who = "every galaxy" if self.all_lines else feature_label(self.line)
            for j, r in enumerate(self.settings.get("secondary_ratios") or []):
                z, f = self.single_line_fraction(r)
                ax.plot(z, 100 * f, color=TEXT_PRIMARY if self.all_lines else feature_color(self.line),
                        ls=("-", "--", ":", "-.")[j % 4], lw=1.8, label=f"secondary {self._fmt(r * self.primary)}", **mark)
            ax.set_xlim(lo, hi)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("true redshift (z_rsd)")
            ax.set_ylabel(f"true single-line [% of {'the detected' if self.all_lines else who + ' above the primary'}]")
            ax.set_title(f"{who} at {lo} < z < {hi}: no other line above the secondary limit", loc="left")
            ax.legend(fontsize=9)
            return fig
        if fig is None:
            fig = plt.figure(figsize=(17, 5.2), constrained_layout=True)
        ax1, ax2, ax3 = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1.25, 1]})
        ax1.plot(z, 100 * f, color=TEXT_PRIMARY, lw=2.0, label="all single lines", **mark)
        for name in self.major_inside:
            z, f = self.fraction_inside(name)
            ax1.plot(z, 100 * f, color=feature_color(name), lw=1.6, label=feature_label(name), **mark)
        ax1.set_ylabel("single-line galaxies [% of the detected]")
        ax1.set_title(f"(a) inside {lo} < z < {hi}: which line is single", loc="left")
        ax1.set_xlim(lo, hi)
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel("true redshift (z_rsd)")
        ax1.legend(fontsize=9)
        pairs = self.major_pairs
        style = {t: ("-", "--", ":", "-.")[j % 4] for j, t in enumerate(dict.fromkeys(t for _, t in pairs))}

        def label(x, t):
            return f"{feature_label(x)} as {feature_label(t)}" if self.all_lines else feature_label(x)

        peaks = []
        for x, t in pairs:
            z, f = self.fraction_interloper(x, t)
            ax2.plot(z, 100 * f, color=feature_color(x), ls=style[t], lw=1.8, label=label(x, t),
                     **({"marker": "o", "ms": 4} if z.size < 5 else {}))
            if np.any(f > 0):
                peaks.append(100 * float(f.max()))
        if len(peaks) > 1 and max(peaks) > 20 * min(peaks):      # curves of very different heights: a log axis
            from matplotlib.ticker import FuncFormatter

            ax2.set_yscale("log")
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            ax2.set_ylim(bottom=0)
        ax2.set_xlabel("true redshift of the interloper (z_rsd)")
        ax2.set_ylabel("single-line galaxies [% of the detected]")
        where = "another line" if self.all_lines else feature_label(self.line)
        ax2.set_title(f"(b) potential interlopers: their line where {where} is", loc="left")
        if pairs:
            ax2.legend(fontsize=9)
        z, f = self.contamination_at_most()
        ax3.plot(z, 100 * f, color=TEXT_PRIMARY, lw=2.0, label="all potential interlopers", **mark)
        for x, t in pairs:
            z, f = self.contamination_at_most(x, t)
            ax3.plot(z, 100 * f, color=feature_color(x), ls=style[t], lw=1.6, label=label(x, t), **mark)
        ax3.set_xlim(lo, hi)
        ax3.set_ylim(bottom=0)
        ax3.set_ylabel("[% of the catalog at that redshift]")
        if self.all_lines:
            ax3.set_xlabel("redshift if taken for that line")
            ax3.set_title("(c) at most: every one taken for that line", loc="left")
        else:
            ax3.set_xlabel(f"redshift if taken for {feature_label(self.line)}")
            ax3.set_title("(c) at most: every one taken for the target line", loc="left")
        ax3.legend(fontsize=9)
        return fig

    # ---- disk ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"settings": self.settings, "z_edges": self.z_edges.tolist(), "line_names": self.line_names,
                "rest_waves": self.rest_waves.tolist(), "area_deg2": self.area_deg2,
                "counts": {k: v.tolist() for k, v in self.counts.items()}}

    def save(self, path) -> Path:
        """Write the proposal (``.npz``); :meth:`load` reads it back."""
        from .catalog import _json_default

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"settings": self.settings, "line_names": self.line_names, "rest_waves": self.rest_waves.tolist(),
                "area_deg2": self.area_deg2}
        np.savez_compressed(path, __proposal__=np.array(json.dumps(meta, default=_json_default)), z_edges=self.z_edges,
                            **{f"count_{k}": v for k, v in self.counts.items()})
        return path

    @classmethod
    def load(cls, path, base=None) -> "Proposal":
        """A proposal written by :meth:`save`; ``base`` (a :class:`BaseCatalog`) is reopened from its settings when
        not given."""
        with np.load(path) as f:
            meta = json.loads(str(f["__proposal__"]))
            counts = {k[len("count_"):]: f[k] for k in f.files if k.startswith("count_")}
            edges = f["z_edges"]
        return cls(counts, edges, meta["settings"], meta["line_names"], meta["rest_waves"], meta["area_deg2"], base)


def _describe_doublets(cfg) -> str:
    """The doublet rule of a proposal in a few words."""
    if not cfg:
        return "no rule: every line a feature of its own"
    from .doublets import build_doublets

    rule = build_doublets(cfg)
    if rule.mode == "separated":
        return f"{rule.weak} a line of its own"
    if rule.mode == "merged":
        return f"{rule.weak} added to {rule.strong} in every galaxy"
    from .report import keep

    return (f"{rule.weak} a line of its own where the doublet is resolved for the galaxy's size "
            f"{keep(f'(dip ≥ {100 * rule.min_dip:g} %)')}, added to {rule.strong} where blended")


def new_settings(**kw) -> dict:
    """The settings of a proposal, stamped with the time it was made."""
    return {**kw, "built": _dt.datetime.now().isoformat(timespec="seconds")}

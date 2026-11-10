"""Step 3 of the plan for single-line galaxies: the confusion matrix.

``C[i, j] = P(assigned line = j | true detected line = i, N_det = 1)`` with an extra
"fail" column ``C[i, fail]`` (no reliable redshift) so that every row sums to one.
Rows and columns follow the bit ordering of the :class:`~romanmock.lines.LineSet`.

A :class:`ConfusionModel` receives a ``context`` dictionary (observed wavelength,
S/N, flux, plus any passthrough columns the pipeline is told to forward) so that
refinements ``C_ij(lambda_obs, S/N, photometry)`` can be added without touching the
rest of the pipeline. :class:`ConfusionMatrix` ignores the context;
:class:`WavelengthBinnedConfusion` selects a matrix by observed wavelength;
:class:`WindowedConfusion` takes a line for another only inside a window of observed
wavelength (the identification of a catalog designed by its interloper fractions);
:class:`LineFractionTable` (``type: empirical``) ignores the true line altogether and
draws the identity from the line mix of the mock's single-line galaxies at the same
observed wavelength and flux rank.

None of the matrices shipped here are calibrated Roman predictions; they are
explicitly assumed scenarios (plan comment 3). The line-fraction table is measured
on the mock, so it is only as good as the mock's line ratios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .lines import LineSet, get_line_set

FAIL = -1          #: assigned-line code for "no reliable redshift"
FAIL_KEY = "fail"  #: dictionary key of the failure column


@runtime_checkable
class ConfusionModel(Protocol):
    names: list[str]

    def assign(self, true_index: np.ndarray, context: Mapping[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
        """Draw an assigned feature index (or :data:`FAIL`) for each single-line galaxy."""
        ...


class ConfusionMatrix:
    """A fixed confusion matrix with a failure column.

    Parameters
    ----------
    names : sequence of str
        Feature names in bit order.
    matrix : (n, n+1) array
        ``matrix[i, j]`` for ``j < n`` is ``C_ij``; ``matrix[i, n]`` is ``C_i,fail``.
    """

    def __init__(self, names: Sequence[str], matrix: np.ndarray, rtol: float = 1e-6):
        self.names = list(names)
        n = len(self.names)
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (n, n + 1):
            raise ValueError(f"matrix must have shape ({n}, {n + 1}); got {m.shape}")
        if np.any(m < 0):
            raise ValueError("confusion probabilities must be non-negative")
        rowsum = m.sum(axis=1)
        if not np.allclose(rowsum, 1.0, rtol=0, atol=rtol):
            bad = [self.names[i] for i in np.where(~np.isclose(rowsum, 1.0, rtol=0, atol=rtol))[0]]
            raise ValueError(f"rows of the confusion matrix must sum to 1 (including fail); bad rows: {bad}")
        self.matrix = m
        self._cdf = np.cumsum(m, axis=1)
        self._cdf[:, -1] = 1.0  # guard against round-off

    # ---- constructors ----------------------------------------------------------------
    @classmethod
    def identity(cls, names: Sequence[str] | LineSet, p_fail: float = 0.0) -> "ConfusionMatrix":
        """Perfect identification with an optional uniform failure probability."""
        names = names.names if isinstance(names, LineSet) else list(names)
        n = len(names)
        m = np.zeros((n, n + 1))
        m[np.arange(n), np.arange(n)] = 1.0 - p_fail
        m[:, n] = p_fail
        return cls(names, m)

    @classmethod
    def from_dict(cls, names: Sequence[str] | LineSet, rows: Mapping[str, Mapping[str, float]],
                  default_p_fail: float = 0.0) -> "ConfusionMatrix":
        """Build from nested dicts ``{true: {assigned: prob, ..., "fail": prob}}``.

        Rows that are not listed are identity rows with ``default_p_fail``. Within a
        listed row, the diagonal entry is filled in so the row sums to one if it is
        omitted.
        """
        names = names.names if isinstance(names, LineSet) else list(names)
        n = len(names)
        idx = {nm: i for i, nm in enumerate(names)}
        m = np.zeros((n, n + 1))
        m[np.arange(n), np.arange(n)] = 1.0 - default_p_fail
        m[:, n] = default_p_fail
        for true_name, row in rows.items():
            if true_name not in idx:
                raise KeyError(f"unknown true line {true_name!r} in confusion matrix; known: {names}")
            i = idx[true_name]
            m[i, :] = 0.0
            for assigned, p in row.items():
                if assigned == FAIL_KEY:
                    m[i, n] = float(p)
                elif assigned in idx:
                    m[i, idx[assigned]] = float(p)
                else:
                    raise KeyError(f"unknown assigned line {assigned!r} in row {true_name!r}")
            if true_name not in row:  # fill diagonal with the remainder
                m[i, i] = max(0.0, 1.0 - m[i, :].sum())
        return cls(names, m)

    @classmethod
    def from_file(cls, path, names: Sequence[str] | LineSet | None = None) -> "ConfusionMatrix":
        """Load from YAML/JSON: either ``{names: [...], matrix: [[...], ...]}`` or the
        nested-dict form of :meth:`from_dict` under ``rows``."""
        import yaml

        with open(Path(path)) as f:
            d = yaml.safe_load(f)
        names = names.names if isinstance(names, LineSet) else names
        names = list(d.get("names", names or []))
        if not names:
            raise ValueError("feature names must be given in the file or as an argument")
        if "matrix" in d:
            return cls(names, np.asarray(d["matrix"], dtype=float))
        return cls.from_dict(names, d.get("rows", {}), default_p_fail=float(d.get("default_p_fail", 0.0)))

    # ---- accessors ------------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def p_fail(self) -> np.ndarray:
        return self.matrix[:, -1]

    def to_dict(self) -> dict:
        rows = {}
        for i, nm in enumerate(self.names):
            row = {self.names[j]: float(self.matrix[i, j]) for j in range(self.n) if self.matrix[i, j] > 0}
            if self.matrix[i, -1] > 0:
                row[FAIL_KEY] = float(self.matrix[i, -1])
            rows[nm] = row
        return {"names": list(self.names), "rows": rows}

    def to_config(self) -> dict:
        return {"type": "matrix", **self.to_dict()}

    def __repr__(self):
        return f"ConfusionMatrix({self.names}, diag={np.round(np.diag(self.matrix), 3).tolist()}, p_fail={np.round(self.p_fail, 3).tolist()})"

    # ---- sampling -------------------------------------------------------------------
    def assign(self, true_index, context=None, rng: np.random.Generator | None = None) -> np.ndarray:
        """Vectorised draw of the assigned identity for each galaxy."""
        true_index = np.asarray(true_index, dtype=np.int64)
        if true_index.size == 0:
            return np.zeros(0, dtype=np.int64)
        if np.any((true_index < 0) | (true_index >= self.n)):
            raise ValueError("true_index out of range for this confusion matrix")
        rng = rng if rng is not None else np.random.default_rng()
        u = rng.random(true_index.size)
        cdf = self._cdf[true_index]                     # (m, n+1)
        j = (u[:, None] > cdf).sum(axis=1)              # first column with cdf > u
        j = np.minimum(j, self.n)
        return np.where(j == self.n, FAIL, j).astype(np.int64)


class WavelengthBinnedConfusion:
    """A confusion matrix per bin of observed wavelength of the detected line.

    ``edges`` are bin edges in Angstrom (length ``n_bins + 1``); ``matrices`` has one
    :class:`ConfusionMatrix` per bin. Wavelengths outside the edges use the nearest
    bin. This is the simplest example of ``C_ij(lambda_obs)`` from plan comment 3.
    """

    def __init__(self, edges: Sequence[float], matrices: Sequence[ConfusionMatrix]):
        self.edges = np.asarray(edges, dtype=np.float64)
        if self.edges.ndim != 1 or self.edges.size < 2 or np.any(np.diff(self.edges) <= 0):
            raise ValueError("edges must be an increasing 1-D array of length >= 2")
        if len(matrices) != self.edges.size - 1:
            raise ValueError("need one confusion matrix per wavelength bin")
        names = matrices[0].names
        if any(m.names != names for m in matrices):
            raise ValueError("all matrices must share the same feature names/order")
        self.matrices = list(matrices)
        self.names = list(names)

    def assign(self, true_index, context, rng):
        true_index = np.asarray(true_index, dtype=np.int64)
        wave = np.asarray(context["wave_obs"], dtype=np.float64)
        b = np.clip(np.searchsorted(self.edges, wave, side="right") - 1, 0, len(self.matrices) - 1)
        out = np.full(true_index.shape, FAIL, dtype=np.int64)
        for k, cm in enumerate(self.matrices):
            sel = b == k
            if np.any(sel):
                out[sel] = cm.assign(true_index[sel], None, rng)
        return out

    def to_config(self) -> dict:
        return {"type": "wavelength_binned", "edges": self.edges.tolist(),
                "matrices": [m.to_dict() for m in self.matrices]}


class WindowedConfusion:
    """Single lines taken for another line only inside a window of observed wavelength (``type: windowed``).

    ``pairs`` lists ``(true line, assigned line, (min, max) observed wavelength [Å], p)``, or dicts with the keys
    ``line``, ``as``, ``window``, ``p``: a single line of the true line observed inside the window (ends included) is
    taken for the assigned line with probability ``p``. Everywhere else, and for the lines not listed, the
    identification is correct. A single line inside the windows of several pairs is taken for one of them with their
    probabilities, which must not add up to more than one there. One uniform deviate per single-line galaxy, as
    :class:`ConfusionMatrix` draws them.

    It is the identification of a catalog designed by its interloper fractions
    (:meth:`romanmock.proposal.Proposal.design`): the window of a pair is where the true line, taken for the other,
    gets a redshift inside the range of the catalog, and ``p`` is what makes the interlopers of the pair the share
    of the catalog asked for. ``meta`` (the fractions asked for, the secondary limit) is kept in the config only.
    """

    def __init__(self, names: Sequence[str] | LineSet, pairs, meta: Mapping | None = None):
        self.names = names.names if isinstance(names, LineSet) else list(names)
        idx = {n: i for i, n in enumerate(self.names)}
        rows = []
        for pair in pairs:
            if isinstance(pair, Mapping):
                true, assigned, window, p = pair["line"], pair["as"], pair["window"], pair["p"]
            else:
                true, assigned, window, p = pair
            for name in (true, assigned):
                if name not in idx:
                    raise KeyError(f"unknown line {name!r} in a windowed confusion pair; the lines are {self.names}")
            if true == assigned:
                raise ValueError(f"a pair takes a line for another one, not for itself ({true!r})")
            w0, w1, p = float(window[0]), float(window[1]), float(p)
            if not w1 > w0:
                raise ValueError(f"the window of {true} -> {assigned} must be (min, max) with min < max; got {list(window)}")
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"the probability of {true} -> {assigned} must be in [0, 1]; got {p}")
            rows.append((true, assigned, w0, w1, p))
        self.pairs = rows
        self.meta = dict(meta or {})
        for true in dict.fromkeys(r[0] for r in rows):         # overlapping windows of one line: at most one in all
            own = [r for r in rows if r[0] == true]
            cuts = sorted({w for r in own for w in r[2:4]})
            for w in cuts + [0.5 * (a + b) for a, b in zip(cuts[:-1], cuts[1:])]:
                total = sum(r[4] for r in own if r[2] <= w <= r[3])
                if total > 1.0 + 1e-9:
                    raise ValueError(f"a single {true} line at {w:.0f} Å would be taken for another line with probability "
                                     f"{total:.3g} > 1 (overlapping windows)")

    @classmethod
    def from_config(cls, line_set, pairs, meta=None, names=None) -> "WindowedConfusion":
        line_set = get_line_set(line_set)
        if names is not None and list(names) != line_set.names:
            raise ValueError(f"the windowed confusion is for the lines {list(names)}, not {line_set.names}")
        return cls(line_set, pairs, meta)

    def pair_probabilities(self) -> np.ndarray:
        """``P[i, j]``: the probability that a single line ``i`` inside the window of the pair is taken for ``j`` (0 for
        the pairs not listed)."""
        idx = {n: i for i, n in enumerate(self.names)}
        out = np.zeros((len(self.names), len(self.names)))
        for true, assigned, _, _, p in self.pairs:
            out[idx[true], idx[assigned]] = max(out[idx[true], idx[assigned]], p)
        return out

    def assign(self, true_index, context, rng: np.random.Generator | None = None) -> np.ndarray:
        true_index = np.asarray(true_index, dtype=np.int64)
        if true_index.size == 0:
            return np.zeros(0, dtype=np.int64)
        rng = rng if rng is not None else np.random.default_rng()
        u = rng.random(true_index.size)
        wave = np.asarray(context["wave_obs"], dtype=np.float64)
        idx = {n: i for i, n in enumerate(self.names)}
        out = true_index.copy()
        low = np.zeros(true_index.size)               # probability already given to earlier pairs of the same galaxy
        for true, assigned, w0, w1, p in self.pairs:
            inside = (true_index == idx[true]) & (wave >= w0) & (wave <= w1)
            out[inside & (u >= low) & (u < low + p)] = idx[assigned]
            low[inside] += p
        return out

    def to_config(self) -> dict:
        cfg = {"type": "windowed", "pairs": [{"line": t, "as": a, "window": [w0, w1], "p": p} for t, a, w0, w1, p in self.pairs]}
        if self.meta:
            cfg["meta"] = dict(self.meta)
        return cfg

    def __repr__(self) -> str:
        pairs = ", ".join(f"{t}->{a} {p:.3g} at {w0:.0f}-{w1:.0f} A" for t, a, w0, w1, p in self.pairs)
        return f"WindowedConfusion({pairs or 'no pairs'})"


#: Default single-line confusion scenario, P(assigned j | true i), keyed by the feature names
#: of the ``roman_transitions`` preset. Rows not listed are identity; there are no failures.
#: E.g. a true Ha line is called [O III] 5007 with probability 0.5 %. This is an assumed
#: scenario, not a Roman calibration.
DEFAULT_CONFUSION_ROWS = {
    "Ha":        {"OIII_5007": 0.005, "fail": 0.0},
    "OIII_5007": {"Ha": 0.05, "fail": 0.0},
    "SIII_9532": {"Ha": 0.05, "fail": 0.0},
    "Hb":        {"OIII_5007": 0.05, "fail": 0.0},
}
#: Equivalent feature names in other presets (e.g. ``plan_species`` sums the [O III] doublet).
_FEATURE_ALIASES = {"OIII_5007": ("OIII_5007", "OIII"), "SIII_9532": ("SIII_9532", "SIII"),
                    "Ha": ("Ha",), "Hb": ("Hb",)}


def default_confusion_rows(line_set) -> dict:
    """:data:`DEFAULT_CONFUSION_ROWS` translated to the feature names of ``line_set``;
    lines the set does not contain are dropped (their probability goes back to the diagonal)."""
    line_set = get_line_set(line_set)

    def resolve(name):
        for cand in _FEATURE_ALIASES.get(name, (name,)):
            if cand in line_set:
                return cand
        return None

    rows = {}
    for true_name, row in DEFAULT_CONFUSION_ROWS.items():
        t = resolve(true_name)
        if t is None:
            continue
        new_row = {}
        for assigned, p in row.items():
            a = FAIL_KEY if assigned == FAIL_KEY else resolve(assigned)
            if a is not None:
                new_row[a] = p
        rows[t] = new_row
    return rows


def default_confusion(line_set) -> "ConfusionMatrix":
    line_set = get_line_set(line_set)
    return ConfusionMatrix.from_dict(line_set, default_confusion_rows(line_set), default_p_fail=0.0)


CONFUSION_MODELS = {}


def build_confusion(cfg, line_set) -> ConfusionModel:
    """Build a confusion model from a config dict.

    ``type`` may be ``"default"`` (:data:`DEFAULT_CONFUSION_ROWS` adapted to the line set),
    ``"identity"`` (optional ``p_fail``), ``"matrix"`` (``rows`` nested dict or ``matrix``
    array, or ``file``), ``"wavelength_binned"`` (``edges`` and a list of ``matrices``
    in either of the ``matrix`` forms), ``"empirical"`` (a :class:`LineFractionTable`
    from ``file`` or ``table``, optional ``mode`` and ``p_fail``) or ``"windowed"``
    (:class:`WindowedConfusion`, ``pairs``). Other types come from
    :data:`CONFUSION_MODELS`. A string is taken as a type without options.
    """
    if isinstance(cfg, str):
        cfg = {"type": cfg}
    if isinstance(cfg, ConfusionModel):
        return cfg
    line_set = get_line_set(line_set)
    cfg = dict(cfg or {})
    kind = cfg.pop("type", "default")
    if kind == "default":
        return default_confusion(line_set)
    if kind == "identity":
        return ConfusionMatrix.identity(line_set, p_fail=float(cfg.get("p_fail", 0.0)))
    if kind == "matrix":
        return _matrix_from_cfg(cfg, line_set)
    if kind == "wavelength_binned":
        mats = [_matrix_from_cfg(dict(m), line_set) for m in cfg["matrices"]]
        return WavelengthBinnedConfusion(cfg["edges"], mats)
    try:
        return CONFUSION_MODELS[kind](line_set=line_set, **cfg)
    except KeyError:
        raise KeyError(f"unknown confusion model {kind!r}")


def _matrix_from_cfg(cfg: dict, line_set: LineSet) -> ConfusionMatrix:
    if "file" in cfg:
        return ConfusionMatrix.from_file(cfg["file"], names=line_set)
    names = cfg.get("names", line_set.names)
    if list(names) != line_set.names:
        raise ValueError(f"confusion matrix names {list(names)} do not match line set {line_set.names}")
    if "matrix" in cfg:
        return ConfusionMatrix(names, np.asarray(cfg["matrix"], dtype=float))
    return ConfusionMatrix.from_dict(names, cfg.get("rows", {}), default_p_fail=float(cfg.get("default_p_fail", 0.0)))


# ---------------------------------------------------------------------------------
# Empirical identification: the line mix of the mock per observed wavelength and flux rank
# ---------------------------------------------------------------------------------
class LineFractionTable:
    """Identify single lines from the line mix of the mock itself (``type: empirical``).

    A single-line galaxy shows one line at an observed wavelength with a flux, and nothing
    else. The mock knows which line it is, and so knows, at every wavelength, how the single
    lines of a given flux are shared among Hα, [S III], [O II], ... This table records that
    share and uses it to identify the single lines: an alternative to a
    :class:`ConfusionMatrix`, whose probabilities are assumed instead of measured.

    The single-line galaxies are binned in the observed wavelength of their line
    (``wave_edges``) and, inside each wavelength bin, in the rank of its flux: the galaxies of
    a wavelength bin are sorted by flux and cut into bins of equal count, as many as hold at
    least ``galaxies_per_bin`` galaxies each (so a populated wavelength bin gets more, finer
    flux bins than a sparse one), or into a fixed number ``n_flux_bins``, or at the quantiles
    ``flux_quantiles``. ``fraction[w, f, j]`` is the share of the galaxies of cell ``(w, f)``
    whose single line is line ``j``. Lines that are the single line of fewer than ``min_count``
    galaxies altogether are left out and the shares renormalised over the others. A
    single-line galaxy is then given line ``j`` with the probability ``fraction[w, f, j]`` of
    its own cell (``mode="draw"``), or the most common line of the cell (``mode="max"``),
    whatever its true line: what an observer who knows the line mix of the survey would do
    with the wavelength and the flux of a single line. ``p_fail`` is a uniform probability of
    obtaining no redshift at all.

    Build it from the base catalog with :meth:`romanmock.catalog.BaseCatalog.line_fractions`
    (at the flux limits of the catalog to be made) or from arrays with
    :meth:`from_single_lines`; :meth:`save` / :meth:`load` keep it in an ``.npz`` file for
    ``confusion: {type: empirical, file: ...}``. :meth:`expected_confusion` is the confusion
    matrix the table implies, averaged over the single-line galaxies of each line.

    Parameters
    ----------
    names : sequence of str
        Feature names in bit order.
    wave_edges : (n_w + 1,) array
        Edges of the observed-wavelength bins [Angstrom].
    flux_edges : (n_w, n_f + 1) array
        Flux edges of the flux-rank bins of each wavelength bin [erg/s/cm^2]: the lowest flux,
        the inner edges, then ``inf`` for the inner edges a wavelength bin with fewer than
        ``n_f`` bins does not have, and the highest flux last.
    counts : (n_w, n_f, n_lines) int array
        Single-line galaxies per cell and line (every line, also the ones left out); zero in
        the cells a wavelength bin does not have.
    used : (n_lines,) bool array, optional
        Lines the identification chooses among (default: all).
    """

    MODES = ("draw", "max")

    def __init__(self, names: Sequence[str] | LineSet, wave_edges, flux_edges, counts, used=None, mode: str = "draw",
                 p_fail: float = 0.0, settings: Mapping | None = None, source=None):
        self.names = names.names if isinstance(names, LineSet) else list(names)
        n = len(self.names)
        self.wave_edges = np.asarray(wave_edges, dtype=np.float64)
        self.flux_edges = np.asarray(flux_edges, dtype=np.float64)
        self.counts = np.asarray(counts, dtype=np.int64)
        if self.wave_edges.ndim != 1 or self.wave_edges.size < 2 or np.any(np.diff(self.wave_edges) <= 0):
            raise ValueError("wave_edges must be an increasing 1-D array of length >= 2")
        n_w = self.wave_edges.size - 1
        if self.flux_edges.ndim != 2 or self.flux_edges.shape[0] != n_w or self.flux_edges.shape[1] < 2:
            raise ValueError(f"flux_edges must have shape ({n_w}, n_flux_bins + 1); got {self.flux_edges.shape}")
        with np.errstate(invalid="ignore"):                                  # inf - inf in the padding
            bad = np.any(np.diff(self.flux_edges[:, :-1], axis=1) < 0) or np.any(self.flux_edges[:, -1] < self.flux_edges[:, 0])
        if bad:
            raise ValueError("the flux edges of a wavelength bin must not decrease (inf padding before the last edge)")
        n_f = self.flux_edges.shape[1] - 1
        if self.counts.shape != (n_w, n_f, n):
            raise ValueError(f"counts must have shape ({n_w}, {n_f}, {n}); got {self.counts.shape}")
        if np.any(self.counts < 0):
            raise ValueError("counts must be non-negative")
        self.used = np.ones(n, dtype=bool) if used is None else np.asarray(used, dtype=bool)
        if self.used.shape != (n,):
            raise ValueError(f"used must have shape ({n},)")
        self.mode = mode
        self.p_fail = p_fail
        self.settings = dict(settings or {})
        self.source = None if source is None else str(source)
        self._check_options()
        self.fraction = self._fractions()

    def _check_options(self):
        if self.mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}; got {self.mode!r}")
        self.p_fail = float(self.p_fail)
        if not 0.0 <= self.p_fail <= 1.0:
            raise ValueError("p_fail must be between 0 and 1")

    def _fractions(self) -> np.ndarray:
        """``fraction[w, f, j]`` over the lines in use; an empty cell takes the mix of its
        wavelength bin, an empty wavelength bin the mix of the whole table."""
        c = np.where(self.used[None, None, :], self.counts, 0).astype(np.float64)
        total = c.sum()
        if total == 0:
            raise ValueError("the table holds no single-line galaxy of the lines in use")
        overall = c.sum(axis=(0, 1)) / total
        per_wave = c.sum(axis=1)
        n_wave = per_wave.sum(axis=1)
        per_wave = np.where(n_wave[:, None] > 0, per_wave / np.maximum(n_wave, 1)[:, None], overall[None, :])
        n_cell = c.sum(axis=2)
        return np.where(n_cell[..., None] > 0, c / np.maximum(n_cell, 1)[..., None], per_wave[:, None, :])

    # ---- construction ---------------------------------------------------------------
    @classmethod
    def from_single_lines(cls, names: Sequence[str] | LineSet, wave, flux, line, wave_edges, galaxies_per_bin: int = 500,
                          n_flux_bins: int | None = None, flux_quantiles=None, min_count: int = 100, mode: str = "draw",
                          p_fail: float = 0.0, settings: Mapping | None = None) -> "LineFractionTable":
        """Build the table from the single-line galaxies: the observed wavelength ``wave`` and the
        flux ``flux`` of the single line of each, and ``line``, its index in ``names``.

        ``wave_edges`` are the wavelength bins. Inside each, the galaxies are ranked by flux and
        cut into bins of equal count: as many as hold at least ``galaxies_per_bin`` galaxies
        each (``N // galaxies_per_bin``, at least one, so every cell has between
        ``galaxies_per_bin`` and twice that), or exactly ``n_flux_bins`` in every wavelength
        bin, or the bins between the quantiles ``flux_quantiles`` (from 0 to 1, e.g.
        ``[0, 0.5, 0.9, 0.99, 1]`` to resolve the bright tail). Lines that are the single line
        of fewer than ``min_count`` galaxies in all are left out.
        """
        names = names.names if isinstance(names, LineSet) else list(names)
        n = len(names)
        wave = np.asarray(wave, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)
        line = np.asarray(line, dtype=np.int64)
        if not wave.shape == flux.shape == line.shape or wave.ndim != 1:
            raise ValueError("wave, flux and line must be 1-D arrays of the same length")
        if wave.size == 0:
            raise ValueError("no single-line galaxies to build the table from")
        if np.any((line < 0) | (line >= n)):
            raise ValueError(f"line indices must be in 0..{n - 1}")
        if not np.all(np.isfinite(wave) & np.isfinite(flux)):
            raise ValueError("wave and flux must be finite")
        wave_edges = np.asarray(wave_edges, dtype=np.float64)
        if wave_edges.ndim != 1 or wave_edges.size < 2 or np.any(np.diff(wave_edges) <= 0):
            raise ValueError("wave_edges must be an increasing 1-D array of length >= 2")
        n_w = wave_edges.size - 1
        iw = np.clip(np.searchsorted(wave_edges, wave, side="right") - 1, 0, n_w - 1)
        n_per_wave = np.bincount(iw, minlength=n_w)
        if flux_quantiles is not None:
            q = np.asarray(flux_quantiles, dtype=np.float64)
            if q.ndim != 1 or q.size < 2 or q[0] != 0.0 or q[-1] != 1.0 or np.any(np.diff(q) <= 0):
                raise ValueError("flux_quantiles must increase from 0 to 1")
            quantiles = [q] * n_w
        elif n_flux_bins is not None:
            if int(n_flux_bins) < 1:
                raise ValueError("n_flux_bins must be at least 1")
            quantiles = [np.linspace(0.0, 1.0, int(n_flux_bins) + 1)] * n_w
        else:
            if int(galaxies_per_bin) < 1:
                raise ValueError("galaxies_per_bin must be at least 1")
            quantiles = [np.linspace(0.0, 1.0, max(1, int(n // int(galaxies_per_bin))) + 1) for n in n_per_wave]
        n_f = max(q.size - 1 for q in quantiles)

        def edges_of(f, q):
            """Flux at the ranks of the quantiles (bin k holds the ranks from q[k] n to q[k+1] n), padded
            with inf before the last edge up to n_f bins."""
            s = np.sort(f)
            e = s[np.minimum((q * s.size).astype(np.int64), s.size - 1)]
            e[0], e[-1] = s[0], s[-1]
            return np.concatenate([e[:-1], np.full(n_f - (q.size - 1), np.inf), e[-1:]])

        flux_edges = np.empty((n_w, n_f + 1))
        for w in range(n_w):
            sel = iw == w
            flux_edges[w] = edges_of(flux[sel], quantiles[w]) if n_per_wave[w] else edges_of(flux, np.array([0.0, 1.0]))
        _, jf = cls._cells(wave_edges, flux_edges, wave, flux)
        counts = np.bincount((iw * n_f + jf) * n + line, minlength=n_w * n_f * n).reshape(n_w, n_f, n)
        used = counts.sum(axis=(0, 1)) >= int(min_count)
        if not used.any():
            raise ValueError(f"no line is the single line of at least min_count = {min_count} galaxies")
        settings = {**dict(settings or {}), "n_single": int(wave.size), "min_count": int(min_count),
                    "flux_bins": {"flux_quantiles": q.tolist()} if flux_quantiles is not None else
                    {"n_flux_bins": int(n_flux_bins)} if n_flux_bins is not None else {"galaxies_per_bin": int(galaxies_per_bin)}}
        return cls(names, wave_edges, flux_edges, counts, used, mode, p_fail, settings)

    @staticmethod
    def _cells(wave_edges, flux_edges, wave, flux, chunk: int = 500_000) -> tuple[np.ndarray, np.ndarray]:
        """Wavelength-bin and flux-rank-bin index of each galaxy; outside the edges, the nearest bin.
        The inf padding of the inner edges never counts, so the index stays below the number of
        bins of the wavelength bin."""
        wave = np.asarray(wave, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)
        n_w = wave_edges.size - 1
        iw = np.clip(np.searchsorted(wave_edges, wave, side="right") - 1, 0, n_w - 1)
        jf = np.empty(wave.size, dtype=np.int64)
        for start in range(0, wave.size, chunk):
            stop = min(start + chunk, wave.size)
            inner = flux_edges[iw[start:stop], 1:-1]                         # (m, n_f - 1)
            jf[start:stop] = (flux[start:stop, None] >= inner).sum(axis=1)
        return iw, jf

    def cells(self, wave, flux) -> tuple[np.ndarray, np.ndarray]:
        """``(wavelength bin, flux-rank bin)`` of galaxies with a single line at ``wave`` with flux ``flux``."""
        return self._cells(self.wave_edges, self.flux_edges, wave, flux)

    # ---- properties -------------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def n_wave_bins(self) -> int:
        return self.wave_edges.size - 1

    @property
    def n_flux_bins(self) -> int:
        """Flux-rank bins of the wavelength bin that has most."""
        return self.flux_edges.shape[1] - 1

    @property
    def n_flux_bins_per_wave(self) -> np.ndarray:
        """``(n_w,)`` flux-rank bins of each wavelength bin."""
        return 1 + np.isfinite(self.flux_edges[:, 1:-1]).sum(axis=1)

    @property
    def valid_cells(self) -> np.ndarray:
        """``(n_w, n_f)`` True for the cells a wavelength bin has (False for the padding)."""
        return np.arange(self.n_flux_bins)[None, :] < self.n_flux_bins_per_wave[:, None]

    @property
    def n_single(self) -> int:
        """Single-line galaxies the table was built from (every line)."""
        return int(self.counts.sum())

    @property
    def lines_used(self) -> list[str]:
        return [nm for nm, u in zip(self.names, self.used) if u]

    @property
    def n_per_cell(self) -> np.ndarray:
        """``(n_w, n_f)`` single-line galaxies per cell, lines in use only (zero in the padding)."""
        return np.where(self.used[None, None, :], self.counts, 0).sum(axis=2)

    def _bins_text(self) -> str:
        nf = self.n_flux_bins_per_wave
        return f"{nf.min()}" if nf.min() == nf.max() else f"{nf.min()}-{nf.max()}"

    def __repr__(self) -> str:
        return (f"LineFractionTable({self.n_single:,} single-line galaxies, {self.n_wave_bins} wavelength x "
                f"{self._bins_text()} flux-rank bins, lines {self.lines_used}, mode={self.mode!r}, p_fail={self.p_fail})")

    # ---- sampling ---------------------------------------------------------------------
    def assign(self, true_index, context, rng: np.random.Generator | None = None) -> np.ndarray:
        """Assigned feature index (or :data:`FAIL`) of each single-line galaxy, from the mix of its
        cell; the true line is not used."""
        m = np.asarray(true_index).size
        if m == 0:
            return np.zeros(0, dtype=np.int64)
        iw, jf = self.cells(context["wave_obs"], context["flux"])
        p = self.fraction[iw, jf]                                            # (m, n)
        if self.mode == "max":
            j = np.argmax(p, axis=1).astype(np.int64)
            if self.p_fail > 0:
                rng = rng if rng is not None else np.random.default_rng()
                j = np.where(rng.random(m) < self.p_fail, FAIL, j)
            return j
        rng = rng if rng is not None else np.random.default_rng()
        cdf = np.cumsum(p, axis=1) * (1.0 - self.p_fail)
        cdf[:, -1] = 1.0 - self.p_fail                                       # beyond: fail
        u = rng.random(m)
        j = (u[:, None] > cdf).sum(axis=1)
        return np.where(j >= self.n, FAIL, j).astype(np.int64)

    def expected_confusion(self) -> ConfusionMatrix:
        """The confusion matrix the table implies: ``P(assigned j | true i)`` averaged over the
        single-line galaxies of line ``i`` (a line without any gets an identity row)."""
        n = self.n
        if self.mode == "max":
            q = np.zeros_like(self.fraction)
            np.put_along_axis(q, np.argmax(self.fraction, axis=2)[..., None], 1.0, axis=2)
        else:
            q = self.fraction
        c = self.counts.astype(np.float64)
        expected = np.einsum("wfi,wfj->ij", c, q)
        total = c.sum(axis=(0, 1))
        m = np.zeros((n, n + 1))
        has = total > 0
        m[has, :n] = expected[has] / total[has, None] * (1.0 - self.p_fail)
        m[~has, np.arange(n)[~has]] = 1.0 - self.p_fail
        m[:, n] = self.p_fail
        return ConfusionMatrix(self.names, m)

    # ---- text -------------------------------------------------------------------------
    def summary(self) -> str:
        """The table as text: how it was built, then per wavelength bin the single-line galaxies, the flux-rank bins,
        the share of each line in use (%) and the most common line of the brightest flux-rank bin."""
        from .report import fields, notes, section, table

        s = self.settings
        lim = (f"F > {s['flux_limit']:.2g}" + (f", other lines < {s['secondary_flux_limit']:.2g}" if s.get("secondary_flux_limit") else "")
               ) if "flux_limit" in s else "flux limits not recorded"
        dropped = [f"{nm} ({int(self.counts[..., i].sum()):,})" for i, nm in enumerate(self.names)
                   if not self.used[i] and self.counts[..., i].sum() > 0]
        how = ", ".join(f"{k} {v}" for k, v in s.get("flux_bins", {}).items())
        out = [section("Line mix of the single-line galaxies (empirical identification)")]
        out += fields([("galaxies", f"{self.n_single:,} single-line galaxies ({lim})"),
                       ("bins", f"{self.n_wave_bins} wavelength bins × {self._bins_text().replace('-', '–')} flux-rank bins"
                                + (f" ({how})" if how else "")),
                       ("assignment", f"mode {self.mode}" + (f", p_fail {self.p_fail}" if self.p_fail else "")),
                       ("lines used", ", ".join(self.lines_used) + (f"; left out, fewer than {s.get('min_count', '?')} galaxies: "
                                                                    + ", ".join(dropped) if dropped else ""))])
        used = [i for i in range(self.n) if self.used[i]]
        c = np.where(self.used[None, None, :], self.counts, 0)
        nf = self.n_flux_bins_per_wave
        rows = []
        for w in range(self.n_wave_bins):
            n_w = int(c[w].sum())
            row = [f"{self.wave_edges[w]:.0f}–{self.wave_edges[w + 1]:.0f}", f"{n_w:,}", f"{nf[w]:d}"]
            if n_w > 0:
                share = c[w].sum(axis=0) / n_w
                top = self.fraction[w, nf[w] - 1]
                k = int(np.argmax(top))
                row += [f"{100 * share[i]:.1f}" if share[i] > 0 else "·" for i in used] + [f"{self.names[k]} {100 * top[k]:.0f} %"]
            else:
                row += ["·"] * len(used) + ["–"]
            rows.append(row)
        out += [""] + table([("λ_obs [Å]", "<"), ("galaxies", ">"), ("flux bins", ">")] + [(self.names[i], ">") for i in used]
                            + [("brightest bin", "<")], rows)
        per_cell = self.n_per_cell[self.valid_cells]
        out += notes(f"line shares in % of the single-line galaxies of the wavelength bin (· none; a bin without any uses the "
                     f"mix of the whole table); brightest bin: the most common line of the brightest flux-rank bin; {int(per_cell.min()):,}–{int(per_cell.max()):,} galaxies "
                     f"per cell (median {int(np.median(per_cell)):,}), {int((per_cell == 0).sum())} of {per_cell.size:,} cells empty")
        return "\n".join(out)

    # ---- serialisation ------------------------------------------------------------------
    def content_hash(self) -> str:
        """Hash of the numbers that decide an assignment (edges, counts, lines in use, mode, p_fail)."""
        import hashlib

        h = hashlib.sha256()
        for a in (self.wave_edges, self.flux_edges, self.counts, self.used):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(f"{self.mode}:{self.p_fail!r}:{self.names}".encode())
        return h.hexdigest()[:16]

    def to_dict(self) -> dict:
        """Everything, as plain lists (``from_dict`` reverses it; also the ``table:`` form of the config)."""
        return {"names": list(self.names), "wave_edges": self.wave_edges.tolist(), "flux_edges": self.flux_edges.tolist(),
                "counts": self.counts.tolist(), "used": self.used.tolist(), "mode": self.mode, "p_fail": self.p_fail,
                "settings": dict(self.settings)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "LineFractionTable":
        return cls(d["names"], d["wave_edges"], d["flux_edges"], d["counts"], d.get("used"), d.get("mode", "draw"),
                   d.get("p_fail", 0.0), d.get("settings"), d.get("source"))

    def to_config(self) -> dict:
        """A compact description for settings and manifests: how the table was built, the lines in
        use, the file it came from and a hash of its content (not the numbers themselves)."""
        nf = self.n_flux_bins_per_wave
        return {"type": "empirical", "file": self.source, "hash": self.content_hash(), "mode": self.mode, "p_fail": self.p_fail,
                "lines": self.lines_used, "n_single": self.n_single, "wave_edges": [round(float(w), 3) for w in self.wave_edges],
                "n_flux_bins": [int(nf.min()), int(nf.max())], "settings": dict(self.settings)}

    def save(self, path) -> Path:
        """Write the table to an ``.npz`` file."""
        import json

        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        np.savez(path, names=np.array(self.names), wave_edges=self.wave_edges, flux_edges=self.flux_edges, counts=self.counts,
                 used=self.used, mode=np.array(self.mode), p_fail=np.array(self.p_fail), settings=np.array(json.dumps(self.settings)))
        self.source = str(path)
        return path

    @classmethod
    def load(cls, path) -> "LineFractionTable":
        import json

        path = Path(path)
        with np.load(path) as f:
            return cls([str(s) for s in f["names"]], f["wave_edges"], f["flux_edges"], f["counts"], f["used"], str(f["mode"]),
                       float(f["p_fail"]), json.loads(str(f["settings"])), source=path)

    @classmethod
    def from_config(cls, line_set, file=None, table=None, mode: str | None = None, p_fail: float | None = None,
                    **build_options) -> "LineFractionTable":
        """``confusion: {type: empirical, file: table.npz}`` (or ``table:`` the dict of :meth:`to_dict`), with
        optional ``mode`` and ``p_fail`` overriding the ones stored. Without a table the model cannot
        be built here: ``BaseCatalog.line_fractions`` / ``romanmock line-fractions`` make one, and
        ``BaseCatalog.make_catalog(confusion="empirical")`` builds it on the fly."""
        if file is not None:
            t = cls.load(file)
        elif table is not None:
            t = cls.from_dict(table)
        else:
            extra = f" (options given: {sorted(build_options)})" if build_options else ""
            raise ValueError("the empirical confusion model needs its line-fraction table: give `file:` (written by "
                             "`romanmock line-fractions` or BaseCatalog.line_fractions(...).save) or `table:`; "
                             "BaseCatalog.make_catalog(confusion='empirical') builds it on the fly" + extra)
        line_set = get_line_set(line_set)
        if t.names != line_set.names:
            raise ValueError(f"the line-fraction table was built for the lines {t.names}, not {line_set.names}")
        if mode is not None:
            t.mode = mode
        if p_fail is not None:
            t.p_fail = p_fail
        t._check_options()
        return t


CONFUSION_MODELS["empirical"] = LineFractionTable.from_config
CONFUSION_MODELS["windowed"] = WindowedConfusion.from_config

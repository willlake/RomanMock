"""Step 3 of the plan for single-line galaxies: the confusion matrix.

``C[i, j] = P(assigned line = j | true detected line = i, N_det = 1)`` with an extra
"fail" column ``C[i, fail]`` (no reliable redshift) so that every row sums to one.
Rows and columns follow the bit ordering of the :class:`~romanmock.lines.LineSet`.

A :class:`ConfusionModel` receives a ``context`` dictionary (observed wavelength,
S/N, flux, plus any passthrough columns the pipeline is told to forward) so that
refinements ``C_ij(lambda_obs, S/N, photometry)`` can be added without touching the
rest of the pipeline. :class:`ConfusionMatrix` ignores the context;
:class:`WavelengthBinnedConfusion` selects a matrix by observed wavelength.

None of the matrices shipped here are calibrated Roman predictions; they are
explicitly assumed scenarios (plan comment 3).
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
    array, or ``file``), or ``"wavelength_binned"`` (``edges`` and a list of ``matrices``
    in either of the ``matrix`` forms).
    """
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

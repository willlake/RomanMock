"""Checks suggested by the plan: empirical confusion, completeness and contamination.

All functions work on dictionaries of numpy arrays with the column names produced by
the pipeline (``z_obs``, ``line_true``, ``line_assigned``, ``id_outcome``, ``n_det``,
``redshift_obs``), so they can be applied to an OpenCosmo dataset via
``ds.select(...).get_data("numpy")``.
"""

from __future__ import annotations

import numpy as np

from .lines import get_line_set
from .redshift import IdOutcome


def empirical_confusion(data, line_set) -> np.ndarray:
    """``(n, n+1)`` matrix of assignment frequencies among single-line galaxies,
    directly comparable to the input :class:`~romanmock.confusion.ConfusionMatrix`."""
    line_set = get_line_set(line_set)
    n = len(line_set)
    single = data["n_det"] == 1
    lt = np.asarray(data["line_true"])[single].astype(np.int64)
    la = np.asarray(data["line_assigned"])[single].astype(np.int64)
    la = np.where(la < 0, n, la)
    m = np.zeros((n, n + 1))
    np.add.at(m, (lt, la), 1.0)
    tot = m.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, m / tot, 0.0)


def contamination_fraction(data, line_set, z_edges, assigned_line: str) -> dict:
    """``f_{i->j}(B) = N_{i->j}(B) / N_sample_j(B)`` for the sample of galaxies assigned
    identity ``j = assigned_line`` in bins ``B`` of assigned redshift (plan comment 4).

    The denominator includes correctly identified multi-line galaxies. Returns a dict
    with ``z_edges``, ``n_sample`` and ``fraction`` of shape ``(n_lines, n_bins)`` whose
    row ``i`` is the fraction of the sample whose true line is ``i``; the diagonal row
    is the purity of the sample, ``fail``/``multi`` are not counted as interlopers.
    """
    line_set = get_line_set(line_set)
    j = line_set.bit(assigned_line)
    z_edges = np.asarray(z_edges, dtype=np.float64)
    z = np.asarray(data["z_obs"])
    la = np.asarray(data["line_assigned"]).astype(np.int64)
    lt = np.asarray(data["line_true"]).astype(np.int64)
    out = np.asarray(data["id_outcome"])
    with_z = np.isfinite(z) & np.isin(out, [int(IdOutcome.MULTI_LINE), int(IdOutcome.SINGLE_CORRECT), int(IdOutcome.SINGLE_MISID)])
    in_sample = with_z & (la == j)
    n_sample, _ = np.histogram(z[in_sample], bins=z_edges)
    frac = np.zeros((len(line_set), z_edges.size - 1))
    for i in range(len(line_set)):
        sel = in_sample & (lt == i) & (out == int(IdOutcome.SINGLE_MISID) if i != j else in_sample & (lt == i))
        n_ij, _ = np.histogram(z[sel], bins=z_edges)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac[i] = np.where(n_sample > 0, n_ij / n_sample, 0.0)
    return {"z_edges": z_edges, "n_sample": n_sample, "fraction": frac, "lines": line_set.names}


def completeness(data, line_set, z_edges, line: str, true_z_column: str = "redshift_obs") -> dict:
    """Fraction of galaxies whose *true* brightest detected feature is ``line`` that end
    up with a correct redshift (multi-line or single correct), in bins of true redshift."""
    line_set = get_line_set(line_set)
    i = line_set.bit(line)
    z_edges = np.asarray(z_edges, dtype=np.float64)
    zt = np.asarray(data[true_z_column])
    lt = np.asarray(data["line_true"]).astype(np.int64)
    out = np.asarray(data["id_outcome"])
    parent = lt == i
    good = parent & np.isin(out, [int(IdOutcome.MULTI_LINE), int(IdOutcome.SINGLE_CORRECT)])
    n_parent, _ = np.histogram(zt[parent], bins=z_edges)
    n_good, _ = np.histogram(zt[good], bins=z_edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(n_parent > 0, n_good / n_parent, np.nan)
    return {"z_edges": z_edges, "n_parent": n_parent, "n_good": n_good, "completeness": c}


def outcome_summary(data) -> dict:
    """Counts per :class:`IdOutcome`."""
    out = np.asarray(data["id_outcome"])
    return {o.name: int(np.sum(out == int(o))) for o in IdOutcome}


def dndz(z, z_edges, area_deg2: float) -> np.ndarray:
    """``dN/dz`` per deg^2 of an array of redshifts (NaNs ignored)."""
    z = np.asarray(z)
    counts, _ = np.histogram(z[np.isfinite(z)], bins=z_edges)
    return counts / (np.diff(z_edges) * area_deg2)

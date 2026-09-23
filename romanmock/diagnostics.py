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


def contamination_fraction(data, line_set, z_edges, assigned_line: str, membership: str = "assigned") -> dict:
    """``f_{i->j}(B) = N_{i->j}(B) / N_sample_j(B)`` for the sample of galaxies assigned
    identity ``j = assigned_line`` in bins ``B`` of assigned redshift (plan comment 4).

    The denominator includes correctly identified multi-line galaxies; ``membership`` is
    the tracer definition of :class:`~romanmock.selection.SampleDefinition`. Returns a dict
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
    from .selection import tracer_membership

    with_z = np.isfinite(z) & np.isin(out, [int(IdOutcome.MULTI_LINE), int(IdOutcome.SINGLE_CORRECT), int(IdOutcome.SINGLE_MISID)])
    in_sample = with_z & tracer_membership(data, line_set, [assigned_line], membership)
    n_sample, _ = np.histogram(z[in_sample], bins=z_edges)
    frac = np.zeros((len(line_set), z_edges.size - 1))
    for i in range(len(line_set)):
        sel = in_sample & (lt == i) & (out == int(IdOutcome.SINGLE_MISID) if i != j else in_sample & (lt == i))
        n_ij, _ = np.histogram(z[sel], bins=z_edges)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac[i] = np.where(n_sample > 0, n_ij / n_sample, 0.0)
    return {"z_edges": z_edges, "n_sample": n_sample, "fraction": frac, "lines": line_set.names}


def detection_efficiency(data, line_set, z_edges, line: str, reference_mask=None, true_z_column: str = "redshift_obs") -> dict:
    """Fraction of a *reference population* with ``line`` detected (bit set), in bins of
    true redshift (review M3, item 1). The reference population is ``reference_mask``
    (e.g. a stellar-mass cut) over all rows of ``data``; it only makes sense on a parent
    catalog written with ``output.keep_undetected: true``, otherwise undetected galaxies are
    absent and the result is 1 by construction."""
    line_set = get_line_set(line_set)
    i = line_set.bit(line)
    z_edges = np.asarray(z_edges, dtype=np.float64)
    zt = np.asarray(data[true_z_column])
    ref = np.ones(zt.size, dtype=bool) if reference_mask is None else np.asarray(reference_mask, dtype=bool)
    det = ((np.asarray(data["line_bitmask"]).astype(np.uint64) >> np.uint64(i)) & np.uint64(1)) > 0
    n_ref, _ = np.histogram(zt[ref], bins=z_edges)
    n_det, _ = np.histogram(zt[ref & det], bins=z_edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = np.where(n_ref > 0, n_det / n_ref, np.nan)
    return {"z_edges": z_edges, "n_reference": n_ref, "n_detected": n_det, "efficiency": eff}


def sample_completeness(data, line_set, z_edges, line: str, in_sample, true_z_column: str = "redshift_obs") -> dict:
    """Fraction of detected emitters whose brightest feature is ``line`` that are in the final
    analysis sample ``in_sample`` (boolean mask over ``data``), in bins of true redshift
    (review M3, item 3: completeness after the actual selection, including thinning)."""
    line_set = get_line_set(line_set)
    i = line_set.bit(line)
    z_edges = np.asarray(z_edges, dtype=np.float64)
    zt = np.asarray(data[true_z_column])
    parent = np.asarray(data["line_true"]).astype(np.int64) == i
    n_parent, _ = np.histogram(zt[parent], bins=z_edges)
    n_in, _ = np.histogram(zt[parent & np.asarray(in_sample, dtype=bool)], bins=z_edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(n_parent > 0, n_in / n_parent, np.nan)
    return {"z_edges": z_edges, "n_parent": n_parent, "n_in_sample": n_in, "completeness": c}


def completeness(data, line_set, z_edges, line: str, true_z_column: str = "redshift_obs") -> dict:
    """Identification efficiency conditional on detection (review M3, item 2): fraction of
    galaxies whose *brightest detected* feature is ``line`` that end up with a correct
    redshift (multi-line or single correct), in bins of true redshift. Undetected galaxies
    are not in the denominator; see :func:`detection_efficiency` for that and
    :func:`sample_completeness` for the completeness of a final sample."""
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


identification_efficiency = completeness


def outcome_summary(data) -> dict:
    """Counts per :class:`IdOutcome`."""
    out = np.asarray(data["id_outcome"])
    return {o.name: int(np.sum(out == int(o))) for o in IdOutcome}


def dndz(z, z_edges, area_deg2: float) -> np.ndarray:
    """``dN/dz`` per deg^2 of an array of redshifts (NaNs ignored)."""
    z = np.asarray(z)
    counts, _ = np.histogram(z[np.isfinite(z)], bins=z_edges)
    return counts / (np.diff(z_edges) * area_deg2)

"""Step 3 of the plan: assign a measured redshift and record the outcome.

* ``N_det >= 2``: baseline assumes the lines identify the redshift correctly,
  ``z_obs = z_rsd`` (optionally with a Gaussian error ``sigma_z (1+z)``). A
  :class:`MultiLineModel` hook allows replacing this later.
* ``N_det == 1``: draw an assigned identity ``j`` from the confusion model, then
  ``z_obs = lambda_{i,0} (1 + z_rsd) / lambda_{j,0} - 1``. Failures get ``z_obs = NaN``.

Outcome codes are in :class:`IdOutcome` and stored as ``id_outcome``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Protocol

import numpy as np

from .confusion import FAIL, ConfusionMatrix, ConfusionModel
from .detection import DetectionResult
from .lines import LineSet, get_line_set

NO_LINE = -1


class IdOutcome(IntEnum):
    """Identification outcome codes stored in the ``id_outcome`` column."""

    NOT_DETECTED = 0     # bitmask == 0: no line detected
    MULTI_LINE = 1       # N_det >= 2: redshift from multiple lines (assumed correct)
    SINGLE_CORRECT = 2   # N_det == 1 and the assigned identity is the true one
    SINGLE_MISID = 3     # N_det == 1 and the assigned identity is wrong (interloper)
    SINGLE_FAILED = 4    # N_det == 1 and no reliable redshift was obtained

    @classmethod
    def describe(cls) -> str:
        return "; ".join(f"{m.value}={m.name}" for m in cls)


@dataclass
class RedshiftAssignment:
    z_obs: np.ndarray          # (n,) assigned redshift, NaN when none
    line_true: np.ndarray      # (n,) int8: index of the (brightest) detected feature, -1 if none
    line_assigned: np.ndarray  # (n,) int8: assigned identity, -1 if none/fail
    outcome: np.ndarray        # (n,) int8: IdOutcome


class MultiLineModel(Protocol):
    def assign(self, det: DetectionResult, z_rsd: np.ndarray, sel: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return z_obs for the galaxies selected by the boolean mask ``sel``."""
        ...


class PerfectMultiLine:
    """``z_obs = z_rsd`` plus optional Gaussian scatter ``sigma_z (1 + z_rsd)``."""

    def __init__(self, sigma_z: float = 0.0):
        self.sigma_z = float(sigma_z)

    def assign(self, det, z_rsd, sel, rng):
        z = np.asarray(z_rsd, dtype=np.float64)[sel]
        if self.sigma_z > 0:
            z = z + self.sigma_z * (1.0 + z) * rng.standard_normal(z.size)
        return z

    def to_config(self):
        return {"type": "perfect", "sigma_z": self.sigma_z}


def brightest_detected(det: DetectionResult) -> np.ndarray:
    """Index of the detected feature with the highest S/N (``-1`` when none)."""
    score = np.where(det.detected, det.snr, -np.inf)
    idx = np.argmax(score, axis=1)
    return np.where(det.n_det > 0, idx, NO_LINE).astype(np.int64)


def assign_redshifts(
    det: DetectionResult,
    z_rsd: np.ndarray,
    line_set="roman_transitions",
    confusion: ConfusionModel | None = None,
    rng: np.random.Generator | None = None,
    multi_line: MultiLineModel | None = None,
    sigma_z: float = 0.0,
    extra_context: Mapping[str, np.ndarray] | None = None,
) -> RedshiftAssignment:
    """Assign ``z_obs`` and identification outcomes.

    Parameters
    ----------
    det : DetectionResult
    z_rsd : (n,) array
        Redshift including peculiar velocities (``redshift_obs``).
    line_set : LineSet or preset
    confusion : ConfusionModel, optional
        Defaults to perfect identification.
    multi_line : MultiLineModel, optional
        Defaults to :class:`PerfectMultiLine` with ``sigma_z``.
    extra_context : mapping, optional
        Additional per-galaxy arrays (photometry, ...) forwarded to the confusion
        model for single-line galaxies.
    """
    line_set = get_line_set(line_set)
    rng = rng if rng is not None else np.random.default_rng()
    confusion = confusion if confusion is not None else ConfusionMatrix.identity(line_set)
    if list(confusion.names) != line_set.names:
        raise ValueError(f"confusion model names {list(confusion.names)} do not match line set {line_set.names}")
    multi_line = multi_line if multi_line is not None else PerfectMultiLine(sigma_z)

    z_rsd = np.asarray(z_rsd, dtype=np.float64)
    n = z_rsd.size
    rest = line_set.rest_waves

    z_obs = np.full(n, np.nan)
    outcome = np.full(n, int(IdOutcome.NOT_DETECTED), dtype=np.int8)
    line_true = brightest_detected(det)
    line_assigned = np.full(n, NO_LINE, dtype=np.int64)

    # multi-line galaxies
    multi = det.n_det >= 2
    if np.any(multi):
        z_obs[multi] = multi_line.assign(det, z_rsd, multi, rng)
        line_assigned[multi] = line_true[multi]
        outcome[multi] = IdOutcome.MULTI_LINE

    # single-line galaxies
    single = det.n_det == 1
    if np.any(single):
        i = line_true[single]
        rows = np.arange(n)[single]
        context = {
            "wave_obs": det.wave_obs[rows, i],
            "snr": det.snr[rows, i],
            "flux": det.flux_measured[rows, i],
            "z_rsd": z_rsd[single],
        }
        if extra_context:
            context.update({k: np.asarray(v)[single] for k, v in extra_context.items()})
        j = confusion.assign(i, context, rng)
        ok = j != FAIL
        zj = np.full(i.size, np.nan)
        zj[ok] = rest[i[ok]] * (1.0 + z_rsd[single][ok]) / rest[j[ok]] - 1.0
        z_obs[single] = zj
        line_assigned[single] = j
        out = np.where(~ok, IdOutcome.SINGLE_FAILED, np.where(j == i, IdOutcome.SINGLE_CORRECT, IdOutcome.SINGLE_MISID))
        outcome[single] = out.astype(np.int8)

    return RedshiftAssignment(
        z_obs=z_obs,
        line_true=line_true.astype(np.int8),
        line_assigned=line_assigned.astype(np.int8),
        outcome=outcome,
    )

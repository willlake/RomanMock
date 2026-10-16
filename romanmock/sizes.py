"""Angular half-light sizes of the galaxies, for the size-dependent sensitivity.

The Roman grism line-flux limits depend on the size of the source
(:class:`~romanmock.sensitivity.RomanGrismSensitivity`). A :class:`SizeSpec` says where the
size of a galaxy comes from:

* a number: the same angular size for every galaxy, in arcsec (``0.3``; ``0`` = point source);
* a column of the input mock, e.g. ``"r50_disk_2d"`` (the 2-D half-mass radius of the disk in
  proper kpc), converted to arcsec with the angular-diameter distance at ``redshift_column``
  (``redshift_true``) and the cosmology of the catalog; a column already in arcsec takes
  ``unit: arcsec``. ``scale`` multiplies the size (the line-emitting region need not have the
  size of the stellar disk) and ``floor`` is the smallest size used. A size that is not finite
  or negative is invalid: it is an error unless ``fallback`` gives the size [arcsec] such
  galaxies take; :meth:`SizeSpec.n_invalid` counts them, so that a catalog can report how many
  galaxies used the fallback (the floor is never a missing-data policy).

In YAML / ``make_catalog(size=...)``::

    size: 0.3
    size: r50_disk_2d
    size: {column: r50_disk_2d, unit: kpc, redshift_column: redshift_true, scale: 1.0, floor: 0.0, fallback: 0.3}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

SIZE_UNITS = ("kpc", "arcsec")


@dataclass(frozen=True)
class SizeSpec:
    fixed: float | None = None          # arcsec, the same for every galaxy
    column: str | None = None           # column of the input mock with the size of each galaxy
    unit: str = "kpc"                   # unit of the column: proper kpc (converted with D_A) or arcsec
    redshift_column: str = "redshift_true"   # redshift of the kpc -> arcsec conversion
    scale: float = 1.0                  # multiplies the size of the column
    floor: float = 0.0                  # smallest size [arcsec]
    fallback: float | None = None       # size [arcsec] of galaxies whose column is not finite or negative; None: an error

    def __post_init__(self):
        if (self.fixed is None) == (self.column is None):
            raise ValueError("a size is a number (arcsec) or a column name")
        if self.fixed is not None and not (np.isfinite(self.fixed) and self.fixed >= 0):
            raise ValueError(f"a fixed size must be a finite number >= 0 arcsec, not {self.fixed}")
        if self.unit not in SIZE_UNITS:
            raise ValueError(f"unit must be one of {SIZE_UNITS}")
        if self.scale <= 0 or self.floor < 0:
            raise ValueError("scale must be positive and floor >= 0")
        if self.fallback is not None and not (np.isfinite(self.fallback) and self.fallback >= 0):
            raise ValueError(f"fallback must be a finite size >= 0 arcsec or None, not {self.fallback}")

    @classmethod
    def parse(cls, spec) -> "SizeSpec":
        """A number (arcsec), a column name, or a dict with ``column`` (+ ``unit``, ``redshift_column``,
        ``scale``, ``floor``, ``fallback``) or ``fixed``."""
        if isinstance(spec, SizeSpec):
            return spec
        if spec is None:
            raise ValueError("size is required: a number in arcsec or a column name")
        if isinstance(spec, str):
            return cls(column=spec)
        if isinstance(spec, Mapping):
            d = dict(spec)
            unknown = sorted(set(d) - {"fixed", "column", "unit", "redshift_column", "scale", "floor", "fallback"})
            if unknown:
                raise KeyError(f"unknown size options {unknown}")
            if "fixed" in d:
                d["fixed"] = float(d["fixed"])
            return cls(**d)
        return cls(fixed=float(spec))

    @property
    def is_fixed(self) -> bool:
        return self.fixed is not None

    @property
    def required_columns(self) -> list[str]:
        """Columns of the input catalog the sizes are computed from."""
        if self.is_fixed:
            return []
        return [self.column] + ([self.redshift_column] if self.unit == "kpc" else [])

    def same_source(self, other: "SizeSpec") -> bool:
        """Do the two specs read the size of a galaxy from the same quantity (column, unit and, for kpc,
        the redshift column), whatever their ``scale``, ``floor`` and ``fallback``?"""
        if self.is_fixed or other.is_fixed:
            return False
        same = self.column == other.column and self.unit == other.unit
        return same and (self.unit != "kpc" or self.redshift_column == other.redshift_column)

    def transform(self, raw):
        """The angular size [arcsec] for a raw size ``raw`` in arcsec (the column converted to arcsec, before
        ``scale`` and ``floor``): ``max(scale * raw, floor)``; a fixed size for a fixed spec."""
        if self.is_fixed:
            return np.full_like(np.asarray(raw, dtype=np.float64), self.fixed)
        return np.maximum(np.asarray(raw, dtype=np.float64) * self.scale, self.floor)

    def _raw(self, data: Mapping[str, np.ndarray] | None, cosmology) -> np.ndarray:
        """The column converted to arcsec (NaN where it is not finite or negative), before scale and floor."""
        if data is None:
            raise ValueError(f"the size comes from the column {self.column!r}: pass the data")
        size = np.asarray(data[self.column], dtype=np.float64)
        if self.unit == "kpc":
            if cosmology is None:
                raise ValueError("a size in kpc needs the cosmology for the angular-diameter distance")
            z = np.asarray(data[self.redshift_column], dtype=np.float64)
            size = size * arcsec_per_kpc_proper(z, cosmology)
        return np.where(np.isfinite(size) & (size >= 0), size, np.nan)

    def n_invalid(self, data: Mapping[str, np.ndarray] | None = None, cosmology=None) -> int:
        """Number of galaxies whose size is not finite or negative (0 for a fixed size)."""
        if self.is_fixed:
            return 0
        return int(np.isnan(self._raw(data, cosmology)).sum())

    def angular_size(self, data: Mapping[str, np.ndarray] | None = None, cosmology=None):
        """Angular half-light size [arcsec]: a float for a fixed size, else an ``(n,)`` array.

        A size that is not finite or negative raises ``ValueError`` unless ``fallback`` is set, in which
        case the galaxy takes the fallback size (not the floor).
        """
        if self.is_fixed:
            return float(self.fixed)
        raw = self._raw(data, cosmology)
        invalid = np.isnan(raw)
        if invalid.any():
            n_bad = int(invalid.sum())
            if self.fallback is None:
                raise ValueError(f"{n_bad:,} of {raw.size:,} galaxies have a size from {self.column!r} that is not finite "
                                 f"or negative; give size={{'column': {self.column!r}, ..., 'fallback': <arcsec>}} to assign "
                                 "them a size, or clean the column")
        size = np.maximum(raw * self.scale, self.floor)
        if invalid.any():
            size = np.where(invalid, self.fallback, size)
        return size

    def to_config(self):
        if self.is_fixed:
            return float(self.fixed)
        d = {"column": self.column, "unit": self.unit}
        if self.unit == "kpc":
            d["redshift_column"] = self.redshift_column
        if self.scale != 1.0:
            d["scale"] = float(self.scale)
        if self.floor != 0.0:
            d["floor"] = float(self.floor)
        if self.fallback is not None:
            d["fallback"] = float(self.fallback)
        return d

    def describe(self) -> str:
        if self.is_fixed:
            return f"{self.fixed:g} arcsec" if self.fixed > 0 else "point source"
        s = f"{self.column}" + (f" [kpc] / D_A({self.redshift_column})" if self.unit == "kpc" else " [arcsec]")
        if self.scale != 1.0:
            s = f"{self.scale:g} x " + s
        if self.floor > 0:
            s += f", at least {self.floor:g} arcsec"
        if self.fallback is not None:
            s += f", {self.fallback:g} arcsec where invalid"
        return s

    def __repr__(self):
        return f"SizeSpec({self.describe()})"


def arcsec_per_kpc_proper(z, cosmology) -> np.ndarray:
    """Arcseconds per proper kpc at redshift ``z`` (``cosmology.arcsec_per_kpc_proper``)."""
    from astropy import units as u

    z = np.asarray(z, dtype=np.float64)
    return np.asarray(cosmology.arcsec_per_kpc_proper(z).to_value(u.arcsec / u.kpc))

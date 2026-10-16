"""Angular half-light sizes of the galaxies, for the size-dependent sensitivity.

The Roman grism line-flux limits depend on the size of the source
(:class:`~romanmock.sensitivity.RomanGrismSensitivity`). A :class:`SizeSpec` says where the
size of a galaxy comes from:

* a number: the same angular size for every galaxy, in arcsec (``0.3``; ``0`` = point source);
* a column of the input mock, e.g. ``"r50_disk_2d"`` (the 2-D half-mass radius of the disk in
  proper kpc), converted to arcsec with the angular-diameter distance at ``redshift_column``
  (``redshift_true``) and the cosmology of the catalog; a column already in arcsec takes
  ``unit: arcsec``. ``scale`` multiplies the size (the line-emitting region need not have the
  size of the stellar disk) and ``floor`` is the smallest size used; sizes that are not finite
  take the floor.

In YAML / ``make_catalog(size=...)``::

    size: 0.3
    size: r50_disk_2d
    size: {column: r50_disk_2d, unit: kpc, redshift_column: redshift_true, scale: 1.0, floor: 0.0}
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
    floor: float = 0.0                  # smallest size [arcsec]; also used where the column is not finite

    def __post_init__(self):
        if (self.fixed is None) == (self.column is None):
            raise ValueError("a size is a number (arcsec) or a column name")
        if self.fixed is not None and not (np.isfinite(self.fixed) and self.fixed >= 0):
            raise ValueError(f"a fixed size must be a finite number >= 0 arcsec, not {self.fixed}")
        if self.unit not in SIZE_UNITS:
            raise ValueError(f"unit must be one of {SIZE_UNITS}")
        if self.scale <= 0 or self.floor < 0:
            raise ValueError("scale must be positive and floor >= 0")

    @classmethod
    def parse(cls, spec) -> "SizeSpec":
        """A number (arcsec), a column name, or a dict with ``column`` (+ ``unit``, ``redshift_column``,
        ``scale``, ``floor``) or ``fixed``."""
        if isinstance(spec, SizeSpec):
            return spec
        if spec is None:
            raise ValueError("size is required: a number in arcsec or a column name")
        if isinstance(spec, str):
            return cls(column=spec)
        if isinstance(spec, Mapping):
            d = dict(spec)
            unknown = sorted(set(d) - {"fixed", "column", "unit", "redshift_column", "scale", "floor"})
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

    def angular_size(self, data: Mapping[str, np.ndarray] | None = None, cosmology=None):
        """Angular half-light size [arcsec]: a float for a fixed size, else an ``(n,)`` array."""
        if self.is_fixed:
            return float(self.fixed)
        if data is None:
            raise ValueError(f"the size comes from the column {self.column!r}: pass the data")
        size = np.asarray(data[self.column], dtype=np.float64) * self.scale
        if self.unit == "kpc":
            if cosmology is None:
                raise ValueError("a size in kpc needs the cosmology for the angular-diameter distance")
            z = np.asarray(data[self.redshift_column], dtype=np.float64)
            size = size * arcsec_per_kpc_proper(z, cosmology)
        size = np.where(np.isfinite(size), size, self.floor)
        return np.maximum(size, self.floor)

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
        return d

    def describe(self) -> str:
        if self.is_fixed:
            return f"{self.fixed:g} arcsec" if self.fixed > 0 else "point source"
        s = f"{self.column}" + (f" [kpc] / D_A({self.redshift_column})" if self.unit == "kpc" else " [arcsec]")
        if self.scale != 1.0:
            s = f"{self.scale:g} x " + s
        if self.floor > 0:
            s += f", at least {self.floor:g} arcsec"
        return s

    def __repr__(self):
        return f"SizeSpec({self.describe()})"


def arcsec_per_kpc_proper(z, cosmology) -> np.ndarray:
    """Arcseconds per proper kpc at redshift ``z`` (``cosmology.arcsec_per_kpc_proper``)."""
    from astropy import units as u

    z = np.asarray(z, dtype=np.float64)
    return np.asarray(cosmology.arcsec_per_kpc_proper(z).to_value(u.arcsec / u.kpc))

"""The sky footprint of the lightcone, and uniform random points inside it.

A HACC lightcone is cut into sky patches, rectangles in the spherical angles ``theta`` (from the
pole) and ``phi``, listed in ``lc_cores-decomposition.txt`` next to the mock. Every galaxy carries
the number of its patch (column ``lc_patch``). The footprint of a mock is the union of the
patches it contains, so it is known exactly: no mask has to be estimated from the galaxies.

Random points uniform on the sky inside a rectangle are uniform in ``phi`` and in ``cos(theta)``;
a patch is drawn with a probability proportional to its solid angle. This is what
``diffsky.experimental.lc_utils.mc_lightcone_random_ra_dec`` does for one rectangle.

In the diffsky mock ``dec = 90 - theta`` and ``ra = phi - 180`` (degrees, modulo 360); the offset
is measured from the galaxies when the footprint is built (:meth:`PatchFootprint.from_mock`).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

DECOMPOSITION_FILE = "lc_cores-decomposition.txt"
FULL_SKY_DEG2 = 4 * np.pi * np.degrees(1.0) ** 2


def read_patch_decomposition(path) -> np.ndarray:
    """Rows ``(patch, theta_lo, theta_hi, phi_lo, phi_hi)`` [radian] of ``lc_cores-decomposition.txt``."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.replace("[", " ").replace("]", " ").replace(",", " ").split()
            if len(parts) == 5 and parts[0].isdigit():
                rows.append([float(x) for x in parts])
    if not rows:
        raise ValueError(f"no patches in {path}")
    return np.array(rows)


class PatchFootprint:
    """A union of ``theta``-``phi`` rectangles on the sky.

    Parameters
    ----------
    theta_lo, theta_hi, phi_lo, phi_hi : arrays [radian]
    ra_offset : float [degree]
        ``phi = ra + ra_offset`` modulo 360.
    patches : array of int, optional
        Patch numbers, for reference.
    """

    def __init__(self, theta_lo, theta_hi, phi_lo, phi_hi, ra_offset: float = 180.0, patches=None):
        self.theta_lo, self.theta_hi, self.phi_lo, self.phi_hi = (
            np.atleast_1d(np.asarray(x, dtype=np.float64)) for x in (theta_lo, theta_hi, phi_lo, phi_hi))
        self.ra_offset = float(ra_offset)
        self.patches = None if patches is None else np.asarray(patches, dtype=np.int64)
        self.solid_angle = (self.phi_hi - self.phi_lo) * (np.cos(self.theta_lo) - np.cos(self.theta_hi))   # steradian
        if np.any(self.solid_angle <= 0):
            raise ValueError("empty patch")

    # ---- construction ------------------------------------------------------------------
    @classmethod
    def from_mock(cls, catalog_files, decomposition=None, ra_offset: float | None = None) -> "PatchFootprint":
        """Footprint of the mock that ``catalog_files`` (one slice or several) belong to.

        The patches are the values of ``lc_patch`` in the files, their rectangles come from
        ``decomposition`` (default: ``lc_cores-decomposition.txt`` in the same directory). Nothing
        about the footprint is assumed: a mock with other patches gives another footprint.
        ``ra_offset`` is measured unless given: the multiple of 90 degrees that puts the galaxies
        inside their own patches. Several slices guard against a slice that misses a patch; a
        warning is given when the slices do not contain the same patches.
        """
        import opencosmo as oc

        files = [Path(catalog_files)] if isinstance(catalog_files, (str, Path)) else [Path(f) for f in catalog_files]
        table = read_patch_decomposition(decomposition or files[0].parent / DECOMPOSITION_FILE)
        rows = {int(r[0]): r for r in table}
        per_file, own, ra, dec = [], [], [], []
        for f in files:
            data = oc.open(str(f)).select("lc_patch", "ra", "dec").get_data("numpy")
            patch = np.asarray(data["lc_patch"]).astype(np.int64)
            found = sorted(np.unique(patch).tolist())
            missing = sorted(set(found) - set(rows))
            if missing:
                raise ValueError(f"patches {missing} of {f.name} are not in the decomposition")
            per_file.append(found)
            step = max(patch.size // 200_000, 1)
            own.append(np.array([rows[int(p)] for p in patch[::step]]))
            ra.append(np.asarray(data["ra"], dtype=np.float64)[::step])
            dec.append(np.asarray(data["dec"], dtype=np.float64)[::step])
        if any(p != per_file[0] for p in per_file):
            warnings.warn("the slices do not contain the same sky patches (" + ", ".join(
                f"{f.name}: {len(p)}" for f, p in zip(files, per_file)) + "): the footprint is their union; if the "
                "footprint of the mock depends on redshift, build one footprint per redshift range", RuntimeWarning, stacklevel=2)
        own, ra, dec = np.concatenate(own), np.concatenate(ra), np.concatenate(dec)
        theta = np.radians(90.0 - dec)

        def inside(offset):
            phi = np.radians((ra + offset) % 360.0)
            return np.mean((theta >= own[:, 1]) & (theta <= own[:, 2]) & (phi >= own[:, 3]) & (phi <= own[:, 4]))

        offsets = [ra_offset] if ra_offset is not None else [180.0, 0.0, 90.0, 270.0]
        found = {o: inside(o) for o in offsets}
        best = max(found, key=found.get)
        if found[best] < 0.98:
            raise ValueError(f"the galaxies are not inside their patches (fraction inside per ra offset: {found})")
        use = np.array([rows[p] for p in sorted(set().union(*per_file))])
        return cls(use[:, 1], use[:, 2], use[:, 3], use[:, 4], ra_offset=best, patches=use[:, 0])

    def check(self, ra, dec, nside: int | None = None) -> dict:
        """Compare the footprint with the positions of galaxies.

        ``inside``: fraction of the galaxies inside the footprint (a low value: the footprint
        misses part of the mock). ``empty``: fraction of the HEALPix pixels lying fully inside the
        footprint that contain no galaxy (a high value: the footprint covers sky the mock does
        not). ``nside`` defaults to the resolution with about 50 galaxies per pixel.
        """
        import healpy as hp

        ra, dec = np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64)
        if nside is None:
            per_deg2 = ra.size / self.area_deg2
            nside = int(2 ** np.clip(np.floor(np.log2(np.sqrt(per_deg2 * FULL_SKY_DEG2 / 12 / 50.0))), 2, 8))
        pixels = np.arange(hp.nside2npix(nside))
        lon, lat = hp.pix2ang(nside, pixels, lonlat=True)
        near = (lat > 90 - np.degrees(self.theta_hi.max()) - 1) & (lat < 90 - np.degrees(self.theta_lo.min()) + 1)
        pixels = pixels[near & self.contains(lon, lat)]
        corners = hp.boundaries(nside, pixels, step=2)                       # (n, 3, 8)
        c_lon, c_lat = hp.vec2ang(np.moveaxis(corners, 1, 2).reshape(-1, 3), lonlat=True)
        interior = pixels[self.contains(c_lon, c_lat).reshape(pixels.size, -1).all(axis=1)]
        counts = np.bincount(hp.ang2pix(nside, ra, dec, lonlat=True), minlength=hp.nside2npix(nside))
        return {"inside": float(np.mean(self.contains(ra, dec))) if ra.size else float("nan"),
                "empty": float(np.mean(counts[interior] == 0)) if interior.size else float("nan"),
                "nside": int(nside), "n_pixels": int(interior.size), "n_galaxies": int(ra.size)}

    def to_dict(self) -> dict:
        return {"ra_offset": self.ra_offset, "area_deg2": self.area_deg2,
                "patches": None if self.patches is None else self.patches.tolist(),
                "theta_lo": self.theta_lo.tolist(), "theta_hi": self.theta_hi.tolist(),
                "phi_lo": self.phi_lo.tolist(), "phi_hi": self.phi_hi.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "PatchFootprint":
        return cls(d["theta_lo"], d["theta_hi"], d["phi_lo"], d["phi_hi"], d.get("ra_offset", 180.0), d.get("patches"))

    def save(self, path) -> Path:
        Path(path).write_text(json.dumps(self.to_dict()))
        return Path(path)

    @classmethod
    def load(cls, path) -> "PatchFootprint":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ---- use ---------------------------------------------------------------------------
    @property
    def n_patches(self) -> int:
        return self.theta_lo.size

    @property
    def area_deg2(self) -> float:
        return float(self.solid_angle.sum() * np.degrees(1.0) ** 2)

    def __repr__(self) -> str:
        return (f"<PatchFootprint: {self.n_patches} patches, {self.area_deg2:.2f} deg2, "
                f"{90 - np.degrees(self.theta_hi.max()):.2f} < dec < {90 - np.degrees(self.theta_lo.min()):.2f}>")

    def rectangles(self) -> np.ndarray:
        """The patches as rows ``(ra_lo, ra_hi, dec_lo, dec_hi)`` in degrees, for drawing. A patch
        that contains ra = 0 is given as two rectangles."""
        rows = []
        for t_lo, t_hi, p_lo, p_hi in zip(self.theta_lo, self.theta_hi, self.phi_lo, self.phi_hi):
            lo = (np.degrees(p_lo) - self.ra_offset) % 360.0
            hi = lo + np.degrees(p_hi - p_lo)
            dec = (90.0 - np.degrees(t_hi), 90.0 - np.degrees(t_lo))
            rows += [(lo, hi, *dec)] if hi <= 360.0 + 1e-9 else [(lo, 360.0, *dec), (0.0, hi - 360.0, *dec)]
        return np.array(rows)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
        """``n`` points uniform on the sky inside the footprint: ``ra, dec`` in degrees."""
        rng = rng if rng is not None else np.random.default_rng()
        n = int(n)
        k = rng.choice(self.n_patches, size=n, p=self.solid_angle / self.solid_angle.sum())
        phi = self.phi_lo[k] + (self.phi_hi[k] - self.phi_lo[k]) * rng.random(n)
        c_lo, c_hi = np.cos(self.theta_hi[k]), np.cos(self.theta_lo[k])
        theta = np.arccos(c_lo + (c_hi - c_lo) * rng.random(n))
        return (np.degrees(phi) - self.ra_offset) % 360.0, 90.0 - np.degrees(theta)

    def contains(self, ra, dec) -> np.ndarray:
        """True for the points inside the footprint."""
        ra, dec = np.broadcast_arrays(np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64))
        shape = ra.shape
        theta = np.radians(90.0 - dec.ravel())
        phi = np.radians((ra.ravel() + self.ra_offset) % 360.0)
        inside = np.zeros(theta.size, dtype=bool)
        rings = np.unique(np.stack([self.theta_lo, self.theta_hi], axis=1), axis=0)
        for lo, hi in rings:                                 # the patches of a ring do not overlap
            ring = (self.theta_lo == lo) & (self.theta_hi == hi)
            order = np.argsort(self.phi_lo[ring])
            p_lo, p_hi = self.phi_lo[ring][order], self.phi_hi[ring][order]
            sel = np.flatnonzero((theta >= lo) & (theta <= hi))
            i = np.searchsorted(p_lo, phi[sel], side="right") - 1
            ok = (i >= 0) & (phi[sel] <= p_hi[np.maximum(i, 0)])
            inside[sel[ok]] = True
        return inside.reshape(shape)

"""Random catalogs.

A random catalog belongs to one data catalog and depends on nothing else: its points are uniform inside the footprint
of the lightcone the data come from (and inside the data's cuts on ra and dec), and its redshifts are drawn, with
replacement, from the redshifts of the data. The data catalog is any :class:`~romanmock.catalog.Catalog`: a whole
catalog, or a part of one, which then gets randoms of its own::

    cat.make_random(10)                                    # the whole catalog
    cat.subset("OIII_5007_as_Ha").make_random(10)          # one kind of interloper (Catalog.subset, Catalog.components)
    cat[cat["redshift_assigned"] < 1.1].make_random(10)    # any selection

``make_random(data, n_times)`` makes ``n_times`` as many points as the data, in blocks of the size of the data: block
``k`` comes from the random stream ``(seed, RANDOM_STREAM, key, k)``, ``key`` a hash of the data's redshifts. The same
data and seed always give the same randoms, more blocks add points and keep the first ones, and two different data
catalogs (two parts of one catalog) get independent randoms with the same seed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

RANDOM_STREAM = 30_000
ANGULAR_COLUMNS = ("ra", "dec")


def data_key(z) -> int:
    """The part of the random stream that comes from the data: a 63-bit hash of its redshifts, in their order."""
    z = np.ascontiguousarray(np.asarray(z, dtype=np.float64))
    return int.from_bytes(hashlib.blake2b(z.tobytes(), digest_size=8).digest(), "little") >> 1


def angular_cuts(settings: dict) -> dict:
    """The cuts of a catalog that act on the sky position, ``{column: (min, max)}``."""
    return {k: tuple(v) for k, v in (settings.get("cuts") or {}).items() if k in ANGULAR_COLUMNS}


def sample_sky(footprint, n: int, rng: np.random.Generator, cuts: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``n`` points uniform inside the footprint and inside the ``cuts`` on ra and dec."""
    if not cuts:
        return footprint.sample(n, rng)
    ra, dec = np.empty(0), np.empty(0)
    accepted, drawn = 0, 0
    while ra.size < n:
        rate = accepted / drawn if accepted else 0.2
        m = int(1.1 * (n - ra.size) / rate) + 1000
        a, d = footprint.sample(m, rng)
        keep = np.ones(m, dtype=bool)
        for col, (lo, hi) in cuts.items():
            v = a if col == "ra" else d
            if lo is not None:
                keep &= v >= lo
            if hi is not None:
                keep &= v <= hi
        accepted, drawn = accepted + int(keep.sum()), drawn + m
        if drawn > 50 * n + 10**6 and accepted == 0:
            raise ValueError(f"the cuts {cuts} leave nothing of the footprint")
        ra, dec = np.concatenate([ra, a[keep]]), np.concatenate([dec, d[keep]])
    return ra[:n], dec[:n]


def make_random(data, footprint, n_times: int = 1, z_column: str | None = None, seed: int = 12345, write_to=None,
                verbose: bool = True):
    """The random catalog of ``data``; see :meth:`romanmock.catalog.Catalog.make_random`."""
    from .catalog import Catalog, catalog_attributes

    z_column = z_column or data.settings.get("z_column") or "redshift_assigned"
    if z_column not in data:
        raise KeyError(f"the catalog has no column {z_column!r}: make it with columns=[..., {z_column!r}]")
    n_times = int(n_times)
    if n_times < 1:
        raise ValueError("n_times must be at least 1")
    z_data = np.asarray(data[z_column], dtype=np.float64)
    z_data = z_data[np.isfinite(z_data)]
    n = int(z_data.size)
    if n == 0:
        raise ValueError(f"the data catalog has no galaxy with a finite {z_column}: there is nothing to make randoms for")
    key, cuts = data_key(z_data), angular_cuts(data.settings)
    settings = {**data.settings, "z_column": z_column,
                "random": {"n_times": n_times, "n_data": n, "seed": int(seed), "data_key": key}}
    info = {"footprint": footprint.to_dict()}
    columns = ["ra", "dec", z_column]
    if verbose:
        where = f" and inside {cuts}" if cuts else ""
        print(f"[romanmock] random catalog: {n_times} x {n:,} points, uniform inside {footprint.area_deg2:.1f} deg²{where}, "
              f"{z_column} drawn from the data", flush=True)

    def block(k: int):
        rng = np.random.default_rng([int(seed), RANDOM_STREAM, key, k])
        ra, dec = sample_sky(footprint, n, rng, cuts)
        return ra, dec, z_data[rng.integers(0, n, size=n)]

    path = None if write_to is None else Path(write_to)
    if path is not None and path.suffix.lower() not in (".hdf5", ".h5", ".npz", ".fits"):
        path = path / "random.hdf5"
    if path is None or path.suffix.lower() not in (".hdf5", ".h5"):
        parts = [block(k) for k in range(n_times)]
        cat = Catalog({c: np.concatenate([p[i] for p in parts]) for i, c in enumerate(columns)}, settings, info,
                      data.line_names, footprint.area_deg2)
        return cat if path is None else cat.write(path)
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:                    # block by block: the memory of one block only
        for name, value in catalog_attributes(settings, info, data.line_names, columns, footprint.area_deg2,
                                              n * n_times).items():
            f.attrs[name] = value
        sets = [f.create_dataset(c, shape=(n * n_times,), dtype="f8") for c in columns]
        for k in range(n_times):
            for d, values in zip(sets, block(k)):
                d[k * n:(k + 1) * n] = values
    if verbose:
        print(f"[romanmock] wrote {path}", flush=True)
    return path

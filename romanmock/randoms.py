"""Random catalogs for the catalogs made by :meth:`romanmock.catalog.BaseCatalog.make_catalog`.

A random catalog has the angular distribution of the footprint (uniform inside it) and the
redshift distribution of its data catalog: the redshifts are drawn, with replacement, from the
redshifts of the data. Three data catalogs come out of a contaminated catalog, each with its
own randoms:

``"all"``          every galaxy of the catalog (galaxies and interlopers),
``"galaxies"``     the galaxies with a correct redshift (multi-line, single-line correct),
``"interlopers"``  the galaxies with a wrong redshift (single-line, misidentified).

Randoms ``n_times`` as numerous as the data are made as ``n_times`` catalogs of the size of the
data, each with its own random stream ``(seed, RANDOM_STREAM, kind, index)``: any of them can be
made again, or more added, without touching the others.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .redshift import IdOutcome

RANDOM_STREAM = 30_000
#: data catalogs of a contaminated catalog: identification outcomes of each
KINDS = {
    "all": None,
    "galaxies": (int(IdOutcome.MULTI_LINE), int(IdOutcome.SINGLE_CORRECT)),
    "interlopers": (int(IdOutcome.SINGLE_MISID),),
}
ANGULAR_COLUMNS = ("ra", "dec")


def kind_mask(catalog, kind: str) -> np.ndarray:
    """Rows of ``catalog`` that belong to ``kind`` (``"all"``, ``"galaxies"`` or ``"interlopers"``)."""
    if kind not in KINDS:
        raise ValueError(f"which must be one of {list(KINDS)}; got {kind!r}")
    if KINDS[kind] is None:
        return np.ones(len(catalog), dtype=bool)
    if "id_outcome" not in catalog:
        raise KeyError(f"the catalog needs the column id_outcome to tell the {kind} apart")
    return np.isin(catalog["id_outcome"], KINDS[kind])


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


def random_files(write_to, kind: str, n_times: int, several_kinds: bool) -> list[Path]:
    """File names of the randoms: ``<stem>[_<kind>]_<index>.hdf5`` next to ``write_to``."""
    path = Path(write_to)
    if path.suffix.lower() not in (".hdf5", ".h5"):
        path = path / "random.hdf5" if path.is_dir() or not path.suffix else path.with_suffix(".hdf5")
    stem = f"{path.stem}_{kind}" if several_kinds else path.stem
    width = max(2, len(str(n_times - 1)))
    return [path.with_name(f"{stem}_{k:0{width}d}{path.suffix}") for k in range(n_times)]


def make_random(catalog, footprint, which: str | Sequence[str] = "all", n_times: int = 1, z_column: str | None = None,
                seed: int = 12345, write_to=None, first: int = 0, verbose: bool = True):
    """Random catalogs for ``catalog``; see :meth:`romanmock.catalog.Catalog.make_random`."""
    from .catalog import Catalog

    kinds = [which] if isinstance(which, str) else list(which)
    z_column = z_column or catalog.settings.get("z_column") or "redshift_assigned"
    if z_column not in catalog:
        raise KeyError(f"the catalog has no column {z_column!r}: make it with columns=[..., {z_column!r}]")
    cuts = angular_cuts(catalog.settings)
    n_times = int(n_times)
    out = {}
    for kind in kinds:
        z_data = np.asarray(catalog[z_column])[kind_mask(catalog, kind)]
        z_data = z_data[np.isfinite(z_data)]
        if verbose:
            print(f"[romanmock] randoms for {kind!r}: {n_times} x {z_data.size:,} points, uniform inside "
                  f"{footprint.area_deg2:.1f} deg2{f' and {cuts}' if cuts else ''}, {z_column} drawn from the data", flush=True)
        files = None if write_to is None else random_files(write_to, kind, first + n_times, len(kinds) > 1)[first:]
        made = []
        for k in range(first, first + n_times):
            rng = np.random.default_rng([int(seed), RANDOM_STREAM, list(KINDS).index(kind), k])
            ra, dec = sample_sky(footprint, z_data.size, rng, cuts)
            z = z_data[rng.integers(0, z_data.size, size=z_data.size)] if z_data.size else z_data.copy()
            settings = {**catalog.settings, "z_column": z_column,
                        "random": {"which": kind, "index": k, "seed": int(seed), "n_data": int(z_data.size)}}
            ran = Catalog({"ra": ra, "dec": dec, z_column: z}, settings, {"footprint": footprint.to_dict()},
                          catalog.line_names, footprint.area_deg2)
            if files is None:
                made.append(ran)
            else:
                made.append(ran.write(files[k - first]))
                if verbose:
                    print(f"[romanmock]   {made[-1]}", flush=True)
        out[kind] = made
    return out[kinds[0]] if isinstance(which, str) else out

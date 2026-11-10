"""Step 2 of the two-step workflow, the neat way: catalogs in memory from the base catalog.

Step 1 (``romanmock base``) stores, once, every galaxy with a line above the base flux limit
together with the flux of all its lines. Everything else is a choice made here, in one call
that can be repeated freely::

    from romanmock import BaseCatalog

    base = BaseCatalog("/path/to/base")
    cat = base.make_catalog(flux_limit=1.5e-16, secondary_flux_limit=8e-17,
                            confusion={"Ha": {"OIII_5007": 0.2}},
                            lines=["Ha"], z_range=(1.0, 1.9))
    cat["ra"], cat["redshift_assigned"], len(cat)     # numpy arrays
    cat.write("halpha.hdf5")                          # only if you want it on disk

The thresholds can be flux limits, as above, or S/N thresholds of the Roman grism
(:class:`~romanmock.sensitivity.RomanGrismSensitivity`: the 5-sigma line sensitivity of the
NASA WFI technical page against observed wavelength and source size, scaled to the exposure
time and background): ``make_catalog(snr=5, secondary_snr=2, size="r50_disk_2d")`` keeps
the galaxies with a line of S/N >= 5 and detects their other lines down to S/N >= 2, with
the size of each galaxy from the base catalog (``size=0.3`` for one size for all).

Single lines can instead be identified from the line mix of the mock itself
(``confusion="empirical"``: the share of each line among the single-line galaxies at the
same observed wavelength and flux rank, :meth:`BaseCatalog.line_fractions`).

Nothing is written unless asked for, and no setting is remembered between calls. The base
catalog is read slice by slice (in parallel); detection, bitmask and redshift assignment use
the same functions and the same per-slice random streams as the pipeline, so a catalog made
here equals the one ``romanmock assign`` + ``romanmock select`` would write.
"""

from __future__ import annotations

import datetime as _dt
import json
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import opencosmo as oc

from ._version import __version__
from .confusion import ConfusionMatrix, LineFractionTable, WindowedConfusion, build_confusion, default_confusion
from .detection import detect_lines
from .doublets import build_doublets
from .io import memory_budget_gb, read_manifest, suggest_n_workers
from .lines import get_line_set
from .redshift import IdOutcome, PerfectMultiLine, assign_redshifts, single_lines
from .report import PREFIX, Progress
from .rescaling import build_rescaling
from .roman import DEFAULT_LINE_FLUX_LIMIT, DEFAULT_SECONDARY_LINE_FLUX_LIMIT
from .samples import THINNING_STREAM
from .selection import TargetDNDZ, tracer_membership, dndz_from_counts, keep_probability, probability_per_galaxy, sky_area_deg2
from .sensitivity import (
    ConstantFluxLimit,
    RomanGrismSensitivity,
    build_detection,
    build_sensitivity,
    describe_sensitivity,
    limits_contained,
    NOISY_BASE_MARGIN,
    needs_size,
    nominal_flux_limit,
    nominal_secondary_limit,
)
#: the arguments of make_catalog / line_fractions that fix the detection thresholds
SENSITIVITY_ARGS = ("flux_limit", "secondary_flux_limit", "wave_range", "sensitivity", "snr", "secondary_snr", "size",
                    "exposure_time", "background")

#: columns of a catalog when ``columns`` is not given
DEFAULT_COLUMNS = ["ra", "dec", "redshift_true", "redshift_rsd", "redshift_assigned", "id_outcome"]
#: columns computed in step 2 (the others are read from the base catalog)
DERIVED_COLUMNS = ["redshift_assigned", "id_outcome", "line_bitmask", "n_det", "line_true", "line_assigned", "doublet_merged"]
DEFAULT_OUTCOMES = ("MULTI_LINE", "SINGLE_CORRECT", "SINGLE_MISID")     # every galaxy with a redshift
CHUNK_ROWS = 1_000_000  #: rows of a slice processed at a time by a worker (bounds its memory)
RSD_MARGIN = 0.012      #: largest |z_rsd - z_true| / (1 + z) allowed for in the line report
FRACTION_STREAM = 40000  #: random stream of the line-fraction scan: (seed, FRACTION_STREAM, step); used by noisy detection models only
#: options of ``confusion={"type": "empirical", ...}`` that build the table on the fly (``BaseCatalog.line_fractions``)
FRACTION_OPTIONS = ("wave_bin", "wave_edges", "galaxies_per_bin", "n_flux_bins", "flux_quantiles", "min_count", "mode", "p_fail", "write_to")
REDSHIFT_COLUMNS = ("redshift_assigned", "redshift_rsd", "redshift_true")

DESCRIPTIONS = {
    "redshift_rsd": "True redshift including the line-of-sight peculiar velocity",
    "redshift_assigned": "Assigned (measured) redshift; NaN when no reliable redshift",
    "id_outcome": "Identification outcome: " + IdOutcome.describe(),
    "line_bitmask": "Detected-line bitmask; bit i = feature i of the line set",
    "n_det": "Number of detected features = popcount(line_bitmask)",
    "line_true": "Index of the brightest detected feature",
    "line_assigned": "Feature index assigned by the redshift identification (-1 = failed)",
    "doublet_merged": "The doublet of the doublet rule ([O III] 4959 + 5007) was one feature for this galaxy (unresolved)",
}
#: the identification outcomes as the reports name them
OUTCOME_LABELS = {"NOT_DETECTED": "not detected", "MULTI_LINE": "multi-line", "SINGLE_CORRECT": "single line, correct redshift",
                  "SINGLE_MISID": "single line taken for another: interloper", "SINGLE_FAILED": "single line, no redshift"}
_Z_SHORT = {"redshift_assigned": "z_assigned", "redshift_rsd": "z_rsd", "redshift_true": "z_true"}


#: the parts of a contaminated catalog by identification outcome (Catalog.subset); one kind of interloper is
#: "<true line>_as_<assigned line>"
COMPONENTS = {
    "all": None,
    "multi_line": (int(IdOutcome.MULTI_LINE),),
    "single_correct": (int(IdOutcome.SINGLE_CORRECT),),
    "galaxies": (int(IdOutcome.MULTI_LINE), int(IdOutcome.SINGLE_CORRECT)),
    "interlopers": (int(IdOutcome.SINGLE_MISID),),
}
PAIR_SEP = "_as_"


def parse_pair(which: str, line_names) -> tuple[int, int] | None:
    """``(true, assigned)`` line indices of one kind of interloper, ``"<true>_as_<assigned>"``; ``None`` otherwise."""
    if PAIR_SEP not in which:
        return None
    true, assigned = which.split(PAIR_SEP, 1)
    names = list(line_names or [])
    unknown = [x for x in (true, assigned) if x not in names]
    if unknown:
        raise ValueError(f"unknown lines {unknown} in {which!r}; the lines are {names}")
    if true == assigned:
        raise ValueError(f"{which!r}: a line taken for itself is not an interloper")
    return names.index(true), names.index(assigned)


def component_mask(catalog, which: str) -> np.ndarray:
    """Rows of ``catalog`` in the part ``which``: one of :data:`COMPONENTS` or ``"<true>_as_<assigned>"``."""
    pair = parse_pair(which, catalog.line_names)
    if pair is not None:
        missing = [c for c in ("id_outcome", "line_true", "line_assigned") if c not in catalog]
        if missing:
            raise KeyError(f"the catalog needs the columns {missing} to tell the interlopers {which!r} apart: make it with "
                           "columns=[..., 'id_outcome', 'line_true', 'line_assigned']")
        return ((np.asarray(catalog["id_outcome"]) == int(IdOutcome.SINGLE_MISID))
                & (np.asarray(catalog["line_true"]) == pair[0]) & (np.asarray(catalog["line_assigned"]) == pair[1]))
    if which not in COMPONENTS:
        raise ValueError(f"which must be one of {list(COMPONENTS)} or '<true line>{PAIR_SEP}<assigned line>' (one kind of "
                         f"interloper, e.g. 'OIII_5007_as_Ha'); got {which!r}")
    if COMPONENTS[which] is None:
        return np.ones(len(catalog), dtype=bool)
    if "id_outcome" not in catalog:
        raise KeyError(f"the catalog needs the column id_outcome to tell the {which} apart")
    return np.isin(catalog["id_outcome"], COMPONENTS[which])


def catalog_attributes(settings, info, line_names, columns, area_deg2, n_galaxies) -> dict:
    """The file attributes of a catalog written to HDF5 (what :meth:`Catalog.read` needs)."""
    meta = {"settings": settings, "info": info, "line_names": line_names, "columns": list(columns), "area_deg2": area_deg2,
            "romanmock_version": __version__, "written": _dt.datetime.now().isoformat(timespec="seconds")}
    return {"romanmock_catalog": json.dumps(meta, default=_json_default), "n_galaxies": int(n_galaxies)}


def _describe_identification(cfg, max_entries: int = 6) -> str | None:
    """The identification of a catalog's single-line galaxies, from its confusion config: the kind of model and, for a
    confusion matrix, its entries that are not the identity (``OIII_5007 → Ha 0.05``)."""
    if cfg is None:
        return None
    if not isinstance(cfg, Mapping):
        return str(cfg)
    kind = cfg.get("type")
    if kind == "matrix":
        entries = [f"{t} → {a} {p:g}" for t, row in (cfg.get("rows") or {}).items() for a, p in row.items() if a != t and p > 0]
        if not entries:
            return "every single line identified correctly"
        more = f", … ({len(entries)} entries)" if len(entries) > max_entries else ""
        return "confusion matrix: " + ", ".join(entries[:max_entries]) + more
    if kind == "windowed":
        pairs = cfg.get("pairs") or []
        asked = (cfg.get("meta") or {}).get("fractions")
        if asked:
            p = {(r["line"], r["as"]): r["p"] for r in pairs}
            entries = [f"{x} → {t} {100 * f:.3g} % (P {p.get((x, t), 0):.3g})" for t, row in asked.items() for x, f in row.items()]
            return "by design, shares of the catalog: " + ", ".join(entries)
        return "inside the windows of the range: " + (", ".join(f"{r['line']} → {r['as']} {r['p']:.3g}" for r in pairs) or "none")
    return {"wavelength_binned": "confusion matrices per wavelength bin", "default": "the default confusion matrix",
            "empirical": "the line mix of the mock (empirical)"}.get(kind, str(kind))


# =====================================================================================
# the result
# =====================================================================================
class Catalog:
    """A catalog in memory: numpy arrays by column name, plus how it was made.

    ``cat["ra"]`` is an array, ``len(cat)`` the number of galaxies, ``cat.columns`` the
    column names, ``cat.settings`` the choices that produced it, ``cat.info`` what happened
    (counts, thinning probabilities) and ``cat.area_deg2`` the sky area of the lightcone.
    """

    def __init__(self, data: Mapping[str, np.ndarray], settings: dict | None = None, info: dict | None = None,
                 line_names: Sequence[str] | None = None, area_deg2: float | None = None):
        self.data = dict(data)
        n = {len(v) for v in self.data.values()}
        if len(n) > 1:
            raise ValueError("columns of different lengths")
        self.settings = dict(settings or {})
        self.info = dict(info or {})
        self.line_names = list(line_names or [])
        self.area_deg2 = area_deg2

    # ---- table-like access ---------------------------------------------------------
    def __len__(self) -> int:
        return len(next(iter(self.data.values()))) if self.data else 0

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        if isinstance(key, (list, tuple)) and all(isinstance(k, str) for k in key):
            return Catalog({k: self.data[k] for k in key}, self.settings, self.info, self.line_names, self.area_deg2)
        rows = np.asarray(key)                       # boolean mask or integer index
        return Catalog({k: v[rows] for k, v in self.data.items()}, self.settings, self.info, self.line_names, self.area_deg2)

    def __contains__(self, name) -> bool:
        return name in self.data

    @property
    def columns(self) -> list[str]:
        return list(self.data)

    def keys(self):
        return self.data.keys()

    def __repr__(self) -> str:
        s = self.settings
        if s.get("sensitivity"):
            lims = describe_sensitivity(s["sensitivity"])
        else:
            lims = f"F > {s.get('flux_limit', float('nan')):.2e}" + (
                f" (other lines > {s['secondary_flux_limit']:.2e})" if s.get("secondary_flux_limit") else "")
        cuts = ", ".join(str(x) for x in [
            f"lines {s['lines']}" if s.get("lines") else None,
            f"{s.get('z_column', 'redshift_assigned')} in {tuple(s['z_range'])}" if s.get("z_range") else None,
            f"lines rescaled ({s['rescaling']['type']})" if s.get("rescaling") else None,
            f"doublet {s['doublets']['mode']}" if s.get("doublets") else None,
            f"downsampled to {s['downsample']}" if s.get("downsample") else None] if x)
        return f"<Catalog: {len(self):,} galaxies, {len(self.data)} columns {self.columns}; {lims}{'; ' + cuts if cuts else ''}>"

    def bit(self, line: str) -> int:
        """Index of ``line`` in the line set (the value stored in ``line_assigned`` / ``line_true``)."""
        return self.line_names.index(line)

    # ---- summaries -----------------------------------------------------------------
    def outcomes(self) -> dict[str, int]:
        """Number of galaxies per identification outcome (needs the column ``id_outcome``)."""
        o = np.asarray(self.data["id_outcome"])
        return {x.name: int(np.sum(o == int(x))) for x in IdOutcome}

    def dndz(self, z_edges, column: str = "redshift_assigned") -> np.ndarray:
        """``dN/dz`` per deg^2 of ``column`` in the bins ``z_edges``."""
        if self.area_deg2 is None:
            raise ValueError("the catalog has no sky area")
        z = np.asarray(self.data[column], dtype=np.float64)
        z_edges = np.asarray(z_edges, dtype=np.float64)
        counts = np.histogram(z[np.isfinite(z)], bins=z_edges)[0]
        return counts / (np.diff(z_edges) * self.area_deg2)

    def summary(self) -> str:
        """The catalog as text: what it holds and how it was made (thresholds, line fluxes, identification, sky area,
        the base galaxies detected), then its galaxies per identification outcome."""
        from .report import count, fields, label, notes, percent, symbols, table, title, z_cut

        s, info = self.settings, self.info
        head = "every galaxy" if not s.get("lines") else " + ".join(label(x) for x in s["lines"])
        if s.get("z_range"):
            col = s.get("z_column", "redshift_assigned")
            head += " · " + z_cut(*s["z_range"], column=_Z_SHORT.get(col, col))
        r = s.get("random")
        out = [title(f"Random catalog · {r.get('n_times', 1)} × {r.get('n_data', 0):,} points · {head}" if r
                     else f"Catalog · {head}")]
        if s.get("sensitivity"):
            lims = symbols(describe_sensitivity(s["sensitivity"]))
        elif "flux_limit" in s:
            lims = f"F > {s['flux_limit']:.2e}" + (f" (other lines > {s['secondary_flux_limit']:.2e})"
                                                   if s.get("secondary_flux_limit") else "")
        else:
            lims = None
        det = s.get("detection") or {}
        if lims and det.get("single_line"):
            lims += "; one line per galaxy: no secondary limit"
        if lims and det.get("primary_line"):
            lims += f"; only galaxies with {label(det['primary_line'])} above the primary limit"
        fluxes = None
        if "rescaling" in s or "doublets" in s:
            fluxes = (f"rescaled ({s['rescaling']['type']})" if s.get("rescaling") else "as stored") + (
                f"; [O III] doublet {s['doublets']['mode']}" if s.get("doublets") else "")
        how = {"user": "given", "footprint": "estimated from the sky patches of the mock", "healpix": "estimated from the galaxies"}
        area = None if self.area_deg2 is None else (f"{self.area_deg2:.2f} deg²"
                                                    + (f" ({how[info['area_source']]})" if info.get("area_source") in how else ""))
        base = f"{info['n_detected']:,} of its {info['n_base']:,} galaxies detected" if {"n_detected", "n_base"} <= set(info) else None
        sizes = None
        if info.get("n_fallback_size"):
            sizes = (f"{info['n_fallback_size']:,} base galaxies ({100 * info['n_fallback_size'] / max(info['n_base'], 1):.2f} %) "
                     f"have no valid size and took the fallback of {info.get('size_fallback', float('nan')):g} arcsec")
        out += fields([("thresholds", lims), ("line fluxes", fluxes), ("identification", _describe_identification(s.get("confusion"))),
                       ("sky area", area), ("base catalog", base), ("sizes", sizes),
                       ("cuts", ", ".join(f"{k} in {tuple(v)}" for k, v in (s.get("cuts") or {}).items()) or None),
                       ("downsampled", str(s["downsample"]) if s.get("downsample") else None),
                       ("columns", ", ".join(self.columns))])
        if "id_outcome" in self.data and len(self):
            rows = [[OUTCOME_LABELS.get(name, name), count(n), percent(n / len(self))] for name, n in self.outcomes().items() if n]
            out += [""] + table([("galaxies", "<"), ("number", ">"), ("share", ">")], rows + [["all", count(len(self)), "100.00 %"]])
        else:
            out += fields([("galaxies", count(len(self)))])
        for name, t in (info.get("downsample") or {}).items():
            out += notes(f"population {name}: {t['n_before']:,} → {t['n_after']:,} emitters "
                         f"(p_keep {min(t['p_keep']):.3f}–{max(t['p_keep']):.3f})")
        return "\n".join(out)

    def subset(self, which: str) -> "Catalog":
        """One part of the catalog: ``"all"``; ``"multi_line"``; ``"single_correct"`` (single-line, correct redshift);
        ``"galaxies"`` (the two, every galaxy with a correct redshift); ``"interlopers"`` (single-line, misidentified:
        a wrong redshift); ``"<true>_as_<assigned>"``, one kind of interloper, e.g. ``"OIII_5007_as_Ha"``. Needs the
        column ``id_outcome`` (and ``line_true``, ``line_assigned`` for one kind of interloper). :meth:`components`
        lists them with their sizes; ``cat.subset(which).make_random(...)`` gives a part its randoms."""
        return self[component_mask(self, which)]

    def components(self) -> dict[str, int]:
        """``{which: size}`` of the parts of the catalog (see :meth:`subset`): ``all``, ``multi_line``,
        ``single_correct``, ``galaxies``, ``interlopers`` and one ``"<true>_as_<assigned>"`` per kind of interloper
        present (needs ``line_true`` and ``line_assigned``), most numerous first."""
        if "id_outcome" not in self:
            return {"all": len(self)}
        out = {k: int(component_mask(self, k).sum()) for k in COMPONENTS}
        if "line_true" in self and "line_assigned" in self:
            wrong = np.asarray(self["id_outcome"]) == int(IdOutcome.SINGLE_MISID)
            n = len(self.line_names)
            pairs = np.bincount(np.asarray(self["line_true"])[wrong].astype(np.int64) * n
                                + np.asarray(self["line_assigned"])[wrong].astype(np.int64), minlength=n * n)
            for k in np.argsort(-pairs, kind="stable"):
                if pairs[k]:
                    out[f"{self.line_names[k // n]}{PAIR_SEP}{self.line_names[k % n]}"] = int(pairs[k])
        return out

    def make_random(self, n_times: int = 1, seed: int = 12345, z_column: str | None = None, write_to=None,
                    footprint=None, verbose: bool = True):
        """The random catalog of this catalog: ``n_times`` as many points as it has galaxies with a redshift, uniform
        inside the footprint of the lightcone (and inside its cuts on ra and dec), with redshifts drawn, with
        replacement, from its own. A part of a catalog is a catalog too and gets randoms of its own:
        ``cat.subset("interlopers").make_random(10)``, ``cat.subset("OIII_5007_as_Ha").make_random(10)``,
        ``cat[mask].make_random(10)`` (:mod:`romanmock.randoms`).

        Parameters
        ----------
        n_times : int
            The random catalog holds ``n_times`` times as many points as the data, in blocks of the size of the data.
        seed : int
            The randoms depend on the seed and on the data only: the same data give the same randoms, more blocks
            (a larger ``n_times``) add points and keep the first ones, and two different data catalogs (two parts of
            one catalog) get independent randoms.
        z_column : str
            Redshift column; default: the one of the redshift cut (``redshift_assigned``).
        write_to : path
            Write the random catalog there and return the path: ``.hdf5`` (written block by block, so that the
            randoms of a large catalog never sit in memory together), ``.npz`` or ``.fits``; a directory gets
            ``random.hdf5``. Read it with :meth:`Catalog.read`.
        footprint : PatchFootprint
            Default: the footprint of the lightcone the catalog was made from.

        Returns
        -------
        A :class:`Catalog` with the columns ra, dec and the redshift (``settings["random"]`` says how it was made), or
        the path of the file.
        """
        from .randoms import make_random

        if footprint is None:
            if not self.settings.get("base"):
                raise ValueError("the catalog does not know its base catalog: pass footprint=")
            footprint = BaseCatalog(self.settings["base"], self.settings.get("base_tag", "romanmock_base")).footprint
        return make_random(self, footprint, n_times, z_column, seed, write_to, verbose)

    def line_report(self) -> str:
        """The lines inside the bandpass for the redshift cut of the catalog, and the interlopers.

        Interlopers are single-line galaxies that were given a wrong line. *Potential* ones
        follow from geometry (a line ``i`` inside the bandpass that, taken for line ``j``, passes
        the redshift cut) and from the confusion model (``P`` = probability of taking ``i`` for
        ``j``); the last columns count the ones that are in the catalog.
        """
        return format_line_report(self.info.get("line_report"))

    def to_table(self):
        """The catalog as an ``astropy.table.Table`` (metadata in ``table.meta``)."""
        from astropy.table import Table

        t = Table(self.data)
        t.meta["settings"] = self.settings
        t.meta["area_deg2"] = self.area_deg2
        return t

    # ---- disk ----------------------------------------------------------------------
    def write(self, path, overwrite: bool = True) -> Path:
        """Write the catalog to ``path``: ``.hdf5``/``.h5`` (one dataset per column, settings in
        the file attributes), ``.fits`` or ``.npz``. An existing file is replaced."""
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        attrs = catalog_attributes(self.settings, self.info, self.line_names, self.columns, self.area_deg2, len(self))
        blob = attrs["romanmock_catalog"]
        suffix = path.suffix.lower()
        if suffix in (".hdf5", ".h5"):
            import h5py

            with h5py.File(path, "w") as f:
                for key, value in attrs.items():
                    f.attrs[key] = value
                for name, arr in self.data.items():
                    d = f.create_dataset(name, data=arr)
                    d.attrs["description"] = DESCRIPTIONS.get(name, "")
        elif suffix == ".fits":
            t = self.to_table()
            t.meta = {"ROMANMOCK": blob} if len(blob) < 60 else {}
            t.write(path, overwrite=True)
            path.with_suffix(".json").write_text(blob)       # FITS headers cannot hold the settings
        elif suffix == ".npz":
            np.savez(path, __romanmock_catalog__=np.array(blob), **self.data)
        else:
            raise ValueError("write to .hdf5, .h5, .fits or .npz")
        return path

    @classmethod
    def read(cls, path) -> "Catalog":
        """Read a catalog written by :meth:`write` (``.hdf5``/``.h5`` or ``.npz``)."""
        path = Path(path)
        if path.suffix.lower() in (".hdf5", ".h5"):
            import h5py

            with h5py.File(path, "r") as f:
                meta = json.loads(f.attrs["romanmock_catalog"])
                data = {k: f[k][...] for k in meta.get("columns", list(f.keys()))}
        elif path.suffix.lower() == ".npz":
            with np.load(path) as f:
                meta = json.loads(str(f["__romanmock_catalog__"]))
                data = {k: f[k] for k in meta.get("columns", [k for k in f.files if k != "__romanmock_catalog__"])}
        else:
            raise ValueError("read from .hdf5, .h5 or .npz")
        return cls(data, meta["settings"], meta["info"], meta["line_names"], meta["area_deg2"])


# =====================================================================================
# the base catalog
# =====================================================================================
class BaseCatalog:
    """The base flux catalog written by step 1 (``romanmock base`` / ``build_base``).

    ``area_deg2`` is the sky area of the lightcone, if you know it. When it is not given it is
    estimated: from the sky patches of the mock, or else from the galaxies
    (:attr:`area_source` tells which). It can also be set later, ``base.area_deg2 = 2500.0``.
    """

    def __init__(self, base_dir, tag: str = "romanmock_base", area_deg2: float | None = None):
        self.dir = Path(base_dir)
        self.tag = tag
        manifest = read_manifest(self.dir, tag)
        if manifest is None or manifest.get("kind") != "base":
            raise FileNotFoundError(f"no base catalog in {self.dir} (manifest {tag}_manifest.yaml): run `romanmock base` first")
        self.description = manifest["base"]
        self.manifest_hash = manifest.get("config_hash")
        self.slices = {int(k): v for k, v in sorted(manifest["slices"].items(), key=lambda kv: int(kv[0]))
                       if v.get("n_galaxies", 0) > 0}
        self.line_set = get_line_set(self.description["line_set"])
        self.sensitivity = build_sensitivity(self.description["sensitivity"])
        self.flux_suffix = self.description["flux"]["suffix"]
        self._area = None
        self.area_source = None            # "user", "footprint" or "healpix", once the area is known
        self._footprint = None
        self._cosmology = None
        if area_deg2 is not None:
            self.area_deg2 = area_deg2

    # ---- what is in it -------------------------------------------------------------
    @property
    def n_galaxies(self) -> int:
        return sum(v["n_galaxies"] for v in self.slices.values())

    @property
    def flux_limit(self) -> float:
        """Primary flux limit of the base catalog: the deepest limit step 2 can use (for a
        wavelength- or size-dependent base sensitivity, its nominal value, in the middle of
        the bandpass; the actual check is made limit against limit)."""
        return nominal_flux_limit(self.sensitivity)

    @property
    def cosmology(self):
        """The cosmology of the lightcone (from the header of the base files), for the kpc -> arcsec
        conversion of the galaxy sizes."""
        if self._cosmology is None:
            self._cosmology = oc.open(str(self.files[0])).cosmology
        return self._cosmology

    @property
    def wave_range(self) -> tuple[float, float]:
        return tuple(self.sensitivity.wave_range)

    @property
    def flux_columns(self) -> list[str]:
        return [f"{name}{self.flux_suffix}" for name in self.line_set.names]

    @property
    def columns(self) -> list[str]:
        """Every column a catalog can contain."""
        stored = [c for c in self.description["columns"]]
        bitmask = self.description.get("bitmask_column") if self.description.get("store_bitmask") else None
        return [*stored, "redshift_rsd", *DERIVED_COLUMNS, *self.flux_columns, *([bitmask] if bitmask else [])]

    @property
    def files(self) -> list[Path]:
        return [self.dir / v["output"] for v in self.slices.values()]

    def read_slices(self, columns: Sequence[str], z_range: tuple[float | None, float | None] | None = None,
                    z_column: str = "redshift_rsd", verbose: bool = False):
        """Yield the ``columns`` of the base catalog slice by slice, as dicts of numpy arrays.

        For computations that stream through the base catalog, such as histograms of the stored
        columns: one slice at a time is in memory (a few hundred MB), whatever the size of the
        base catalog, and no worker processes are used. ``z_range`` keeps the rows with
        ``z_column`` inside it and skips the slices entirely outside it. The stored columns are
        :attr:`columns` without the ones step 2 derives (``base.description["columns"]``,
        ``redshift_rsd``, the line fluxes and the base bitmask).
        """
        columns = list(columns)
        need_z = z_range is not None
        read = list(dict.fromkeys([*columns, *([z_column] if need_z else [])]))
        lo, hi = (None, None) if z_range is None else z_range
        for step, v in self.slices.items():
            if need_z:
                margin = RSD_MARGIN * (1 + v["z_max"])
                if (lo is not None and v["z_max"] + margin < lo) or (hi is not None and v["z_min"] - margin > hi):
                    continue
            if verbose:
                print(f"[romanmock] slice {step}: {v['n_galaxies']:,} rows", flush=True)
            data = oc.open(str(self.dir / v["output"])).select(*read).get_data("numpy", wrap_single=True)
            if need_z:
                z = np.asarray(data[z_column], dtype=np.float64)
                keep = np.isfinite(z)
                if lo is not None:
                    keep &= z >= lo
                if hi is not None:
                    keep &= z <= hi
                yield {c: np.asarray(data[c])[keep] for c in columns}
            else:
                yield {c: np.asarray(data[c]) for c in columns}

    @property
    def area_deg2(self) -> float:
        """Sky area of the lightcone [deg^2]: the value given by the user, else an estimate. The
        estimate is the area of the sky patches of the mock (:attr:`footprint`) or, without them,
        the area covered by the galaxies in the HEALPix index (cached in the base directory)."""
        if self._area is None:
            try:
                self._area, self.area_source = self.footprint.area_deg2, "footprint"
                return self._area
            except (FileNotFoundError, KeyError, ValueError, OSError):
                pass
            warnings.warn("the footprint of the mock is not available: the sky area is estimated from the galaxies",
                          RuntimeWarning, stacklevel=2)
            cache = self.dir / f"{self.tag}_area.json"
            stored = json.loads(cache.read_text()) if cache.exists() else {}
            self.area_source = "healpix"
            if stored.get("base") == self._identity():
                self._area = float(stored["area_deg2"])
            else:
                self._area = float(sky_area_deg2(oc.open(*self.files)))
                try:
                    cache.write_text(json.dumps({"area_deg2": self._area, "base": self._identity()}))
                except OSError:          # read-only base catalog: just recompute next time
                    pass
        return self._area

    @area_deg2.setter
    def area_deg2(self, value):
        if value is None:                                  # back to the estimate
            self._area, self.area_source = None, None
            return
        value = float(value)
        if not 0 < value <= 4 * np.pi * np.degrees(1.0) ** 2:
            raise ValueError(f"area_deg2 = {value}: the sky area is between 0 and 41253 deg2")
        self._area, self.area_source = value, "user"

    @property
    def footprint(self):
        """Sky footprint of the lightcone: the union of the sky patches of the input mock.

        It is built from the mock the base catalog was made of (nothing is assumed about its
        size), checked against the galaxies of the base catalog, and cached in the base
        directory. The cache is dropped when the base catalog has been rebuilt.
        """
        if self._footprint is None:
            from .footprint import PatchFootprint

            cache = self.dir / f"{self.tag}_footprint.json"
            if cache.exists():
                stored = json.loads(cache.read_text())
                if stored.get("base") == self._identity():
                    self._footprint = PatchFootprint.from_dict(stored)
            if self._footprint is None:
                import yaml

                meta = yaml.safe_load((self.dir / f"{self.tag}_metadata.yaml").read_text())
                inputs = {Path(p).name: Path(p) for p in meta.get("input_files", [])}
                slices = sorted(self.slices.values(), key=lambda v: -v.get("n_input", 0))     # the most populated first
                files = [inputs[v["input"]] for v in slices if v.get("input") in inputs and inputs[v["input"]].exists()]
                if not files:
                    raise FileNotFoundError("the input mock of the base catalog is not there: the footprint cannot be "
                                            "built. Pass footprint=PatchFootprint(...)")
                footprint = PatchFootprint.from_mock(files[:3])
                big = max(self.slices.values(), key=lambda v: v["n_galaxies"])
                pos = oc.open(str(self.dir / big["output"])).select("ra", "dec").get_data("numpy")
                check = footprint.check(pos["ra"], pos["dec"])
                if not check["inside"] > 0.99 or check["empty"] > 0.01:
                    raise ValueError(f"the footprint built from the sky patches of the mock ({footprint}) does not match "
                                     f"the galaxies of {big['output']}: {check}. Pass footprint=PatchFootprint(...)")
                self._footprint = footprint
                try:
                    cache.write_text(json.dumps({**footprint.to_dict(), "base": self._identity(), "check": check}))
                except OSError:
                    pass
        return self._footprint

    def _identity(self) -> dict:
        """What a cached footprint or area belongs to: changes when the base catalog is rebuilt."""
        return {"config_hash": self.manifest_hash, "n_slices": len(self.slices), "n_galaxies": self.n_galaxies,
                "completed": max(str(v.get("completed", "")) for v in self.slices.values())}

    def make_random(self, catalog: "Catalog", **settings):
        """``catalog.make_random(footprint=self.footprint, **settings)``; see :meth:`Catalog.make_random`."""
        return catalog.make_random(footprint=self.footprint, **settings)

    def __repr__(self) -> str:
        z = [v["z_min"] for v in self.slices.values()] + [v["z_max"] for v in self.slices.values()]
        return (f"<BaseCatalog {self.dir}: {self.n_galaxies:,} galaxies in {len(self.slices)} slices, "
                f"{min(z):.2f} < z < {max(z):.2f}; flux limit {self.flux_limit:.2e} erg/s/cm2 in "
                f"{self.wave_range[0]:.0f}-{self.wave_range[1]:.0f} A; lines {self.line_set.names}>")

    # ---- step 2 --------------------------------------------------------------------
    def make_catalog(
        self,
        flux_limit: float | None = None,
        secondary_flux_limit: float | None | str = "default",
        snr: float | None = None,
        secondary_snr: float | None | str = "default",
        size=None,
        exposure_time: float | None = None,
        background: float | None = None,
        sensitivity=None,
        confusion=None,
        rescaling=None,
        doublets=None,
        z_range: tuple[float | None, float | None] | None = None,
        z_column: str = "redshift_assigned",
        lines: Sequence[str] | str | None = None,
        membership: str = "detected",
        outcomes: Sequence[str] | None = DEFAULT_OUTCOMES,
        n_det_min: int | None = None,
        cuts: Mapping[str, tuple] | None = None,
        downsample=None,
        columns: Sequence[str] | str | None = None,
        wave_range: tuple[float, float] | None = None,
        detection: Mapping | None = None,
        sigma_z: float = 0.0,
        seed: int = 12345,
        n_workers: int | None = None,
        write_to=None,
        area_deg2: float | None = None,
        verbose: bool = True,
    ) -> Catalog:
        """Build a catalog from the base catalog. Every argument is optional.

        Parameters
        ----------
        flux_limit : float
            Primary limit [erg/s/cm^2]: a galaxy needs one line above it inside the
            bandpass. Default: the limit of the base catalog. Cannot be deeper than that.
        secondary_flux_limit : float or None
            Limit for the other lines of such a galaxy. ``None`` = no second tier.
            Default: 4.0e-17 (never above the primary limit).
        snr, secondary_snr, size, exposure_time, background
            The other way of fixing the thresholds: S/N in the Roman grism
            (:class:`~romanmock.sensitivity.RomanGrismSensitivity`, the 5-sigma line
            sensitivity of the NASA WFI technical page against observed wavelength and
            source size). A galaxy needs one line with S/N >= ``snr`` (5) inside the bandpass;
            its other lines are detected with S/N >= ``secondary_snr`` (2; ``None`` = no second
            tier). ``size`` is the angular size of the galaxies: a number in arcsec for all
            (0.3 by default; 0 = point source) or a column of the base catalog in proper kpc
            (``"r50_disk_2d"``; it must be in ``base.columns``), converted at ``redshift_true``;
            ``exposure_time`` in seconds (3600) and ``background`` in units of the minimum
            zodiacal light (2; a typical HLWAS field is about 1.4) rescale the noise. Giving any
            of these selects the S/N route; the flux limits must then be left alone. The S/N of
            every line is available as the ``<line>_snr`` columns of the file route
            (``store_snr``) and in ``cat.settings["sensitivity"]`` as the model used.
        sensitivity : dict or model
            Full control: any sensitivity model config (``{"type": "roman_grism", ...}``,
            ``{"type": "tabulated", ...}``) or object, instead of the arguments above.
        confusion : dict, str, ConfusionMatrix, LineFractionTable or None
            How single-line galaxies are identified. A confusion matrix: ``{true line:
            {assigned line: probability, "fail": probability}}``, lines not listed identified
            correctly; ``None`` = the package default. Or the line mix of the mock:
            ``"empirical"`` builds a :class:`~romanmock.confusion.LineFractionTable` from the
            base catalog at these flux limits first (one more pass over it; the options of
            :meth:`line_fractions` go in a dict, ``{"type": "empirical", "wave_bin": 250.0,
            "galaxies_per_bin": 500, "min_count": 100, "mode": "draw"}``), a table made by
            :meth:`line_fractions` is used as it is, ``{"type": "empirical", "file": ...}``
            loads a saved one. A full config (``{"type": ...}``) or a model object also works.
        rescaling : str, dict or model
            Rescale the line fluxes of every galaxy before the detection: ``"galacticus_like"``
            (:class:`~romanmock.rescaling.GalacticusLike`, the diffsky lines toward the Galacticus
            mock), any config of :func:`~romanmock.rescaling.build_rescaling`, or a model. The flux
            columns of the catalog are then the rescaled ones. Default: the stored fluxes.
        doublets : str, dict or DoubletRule
            When [O III] λ4959 is a line of its own (:class:`~romanmock.doublets.DoubletRule`):
            ``"per_galaxy"`` (where the doublet is resolved for the galaxy's size, needs
            ``r50_disk_2d`` in ``base.columns``; elsewhere λ4959 is added to λ5007), ``"merged"``
            (added in every galaxy) or ``"separated"``. Default: separated, as without a rule.
            ``doublet_merged`` (a column) says where it was added.
        z_range : (min, max)
            Keep galaxies with ``z_column`` inside the range (either end may be None).
        z_column : {"redshift_assigned", "redshift_rsd", "redshift_true"}
        lines : list of str
            Keep the galaxies in which one of these lines is identified, e.g. ``["Ha"]``.
        membership : {"detected", "assigned"}
            What "identified" means for a galaxy with several detected lines. ``"detected"``
            (default): each of its detected lines is identified, since its redshift is right; a
            galaxy with Ha and [O III] belongs to both. ``"assigned"``: only its brightest line
            (``line_assigned``), which makes the samples of different lines disjoint. A
            single-line galaxy always counts under the line it was taken for.
        outcomes : list of str or None
            Identification outcomes to keep. Default: every galaxy with a redshift
            (multi-line, single-line correct, single-line misidentified). ``None`` = all.
        n_det_min : int
            Keep galaxies with at least this number of detected lines.
        cuts : dict
            ``{column: (min, max)}`` on any column, e.g. ``{"dec": (60, 70)}``.
        downsample : None, "wang2022" or dict
            Thin the emitter populations to a reference dN/dz in bins of true redshift.
            ``"wang2022"``: the Wang et al. (2022) forecast for Ha and [O III] 5007 at the
            primary flux limit. A dict customises it: ``{"reference": "wang2022",
            "dust_av": 1.92}``, or one target per line, ``{"Ha": {"z": [...], "dndz": [...]}}``.
            Default: no downsampling.
        columns : list of str, "all" or None
            Columns of the result. Default: ra, dec, redshift_true, redshift_rsd,
            redshift_assigned, id_outcome. See :attr:`columns` for what is available.
        wave_range : (min, max) in Angstrom
            Bandpass; default and maximum: the one of the base catalog.
        detection : dict
            Detection model config; default the deterministic threshold rule.
        sigma_z : float
            Gaussian error ``sigma_z (1 + z)`` on multi-line redshifts.
        seed : int
        n_workers : int
            Default: as many as fit in the memory this session may use, at most 32.
        write_to : path
            Also write the catalog there (see :meth:`Catalog.write`).
        area_deg2 : float
            Sky area of the lightcone [deg^2], used for dN/dz and for the downsampling.
            Default: the area of the base catalog (given by the user, else estimated).
        verbose : bool
            Print the progress and the line report: the lines inside the bandpass for the
            redshift cut, and the potential and actual line interlopers. The report is also
            kept: ``cat.line_report()`` (text) and ``cat.info["line_report"]`` (numbers).
        """
        if n_workers is None:
            n_workers = suggest_n_workers(max_workers=32, per_worker_gb=2.5)     # memory-limited: ~12 on a login node
        sens_kw = dict(flux_limit=flux_limit, secondary_flux_limit=secondary_flux_limit, wave_range=wave_range, sensitivity=sensitivity,
                       snr=snr, secondary_snr=secondary_snr, size=size, exposure_time=exposure_time, background=background)
        confusion = self._resolve_confusion(confusion, sens_kw, detection, seed, n_workers, verbose, rescaling, doublets)
        spec = self._spec(sens_kw, confusion, z_range, z_column, lines, membership, outcomes, n_det_min,
                          cuts, downsample, columns, detection, sigma_z, seed, rescaling, doublets)
        populations = spec["populations"]
        if area_deg2 is not None:
            self.area_deg2 = area_deg2
        area = self.area_deg2
        tasks = [(spec, str(self.dir / v["output"]), step) for step, v in self.slices.items()]
        with Progress(f"catalog: scanning {len(tasks)} slices with {n_workers} worker(s)", verbose):
            results = _run(tasks, n_workers)

        data = {c: np.concatenate([r["columns"][c] for r in results]) for c in spec["columns"]}
        info = {"n_base": int(sum(r["n_base"] for r in results)),
                "n_detected": int(sum(r["n_detected"] for r in results)),
                "n_fallback_size": int(sum(r["n_fallback_size"] for r in results)),
                "n_selected": int(len(next(iter(data.values())))) if data else 0}
        if populations:
            keep, info["downsample"] = _thin(results, populations, spec, area)
            data = {c: v[keep] for c, v in data.items()}
            ident = {k: np.concatenate([r["ident"][k] for r in results])[keep] for k in results[0]["ident"]}
            counts = line_counts(n_lines=len(self.line_set), **ident)
        else:
            counts = {k: np.sum([r["line_counts"][k] for r in results], axis=0) for k in results[0]["line_counts"]}
        info["n_galaxies"] = int(len(next(iter(data.values())))) if data else 0
        z_cover = (min(v["z_min"] for v in self.slices.values()), max(v["z_max"] for v in self.slices.values()))
        # the lines are observed at redshift_rsd: allow for peculiar velocities up to ~3600 km/s
        z_cover = (max(z_cover[0] - RSD_MARGIN * (1 + z_cover[0]), 0.0), z_cover[1] + RSD_MARGIN * (1 + z_cover[1]))
        info["line_report"] = line_report(self.line_set.names, self.line_set.rest_waves, spec["wave_range"], spec["z_range"],
                                          spec["z_column"], z_cover, spec["line_bits"], confusion_probabilities(spec["confusion"]),
                                          counts, spec["outcome_codes"], spec["membership"],
                                          probability_kind=_probability_kind(spec["confusion"]))
        settings = {k: spec[k] for k in ("sensitivity", "flux_limit", "secondary_flux_limit", "wave_range", "z_range", "z_column", "lines",
                                         "membership", "outcomes", "n_det_min", "cuts", "sigma_z", "seed", "detection")}
        settings.update(confusion=spec["confusion_config"], downsample=spec["downsample_label"], base=str(self.dir), base_tag=self.tag,
                        rescaling=spec["rescaling"], doublets=spec["doublets"])
        info["area_deg2"], info["area_source"] = area, self.area_source
        if info["n_fallback_size"]:
            info["size_fallback"] = float(spec["size"].fallback)
        cat = Catalog(data, settings, info, self.line_set.names, area)
        if verbose:
            print("\n" + cat.summary() + "\n\n" + cat.line_report(), flush=True)
        if cat.nbytes > 0.5e9 * memory_budget_gb():
            warnings.warn(f"the catalog takes {cat.nbytes / 1e9:.1f} GB of memory", stacklevel=2)
        if write_to is not None:
            cat.write(write_to)
        return cat

    # ---- settings -> a picklable description for the workers -------------------------
    def _sensitivity(self, flux_limit=None, secondary_flux_limit="default", wave_range=None, sensitivity=None, snr=None,
                     secondary_snr="default", size=None, exposure_time=None, background=None, detection=None):
        """The sensitivity model of a catalog: flux limits (``ConstantFluxLimit``), S/N thresholds
        (``RomanGrismSensitivity``) or a given model; validated against the base catalog. ``detection``
        (the detection config) only decides whether to warn about a noisy model close to the base limit."""
        wave_range = self.wave_range if wave_range is None else (float(wave_range[0]), float(wave_range[1]))
        flux_given = flux_limit is not None or not (isinstance(secondary_flux_limit, str) and secondary_flux_limit == "default")
        snr_given = {k: v for k, v in dict(snr=snr, size=size, exposure_time=exposure_time, background=background).items()
                     if v is not None}
        if not (isinstance(secondary_snr, str) and secondary_snr == "default"):
            snr_given["secondary_snr"] = secondary_snr
        if sensitivity is not None:
            if flux_given or snr_given:
                raise ValueError("give either sensitivity=... or the flux limits / S/N thresholds, not both")
            if isinstance(sensitivity, Mapping):
                sensitivity = {**sensitivity, "wave_range": list(wave_range)} if "wave_range" not in sensitivity else sensitivity
            sens = build_sensitivity(sensitivity)
        elif snr_given:
            if flux_given:
                raise ValueError("flux limits (flux_limit, secondary_flux_limit) and S/N thresholds (snr, secondary_snr, size, "
                                 "exposure_time, background) are two ways of fixing the detection; give one of them")
            sens = RomanGrismSensitivity(wave_range=wave_range, **snr_given)
        else:
            flux_limit = self.flux_limit if flux_limit is None else float(flux_limit)
            secondary = secondary_flux_limit
            if isinstance(secondary, str):
                if secondary != "default":
                    raise ValueError("secondary_flux_limit must be a number, None or 'default'")
                secondary = min(DEFAULT_SECONDARY_LINE_FLUX_LIMIT, flux_limit)
            sens = ConstantFluxLimit(flux_limit, wave_range, secondary_flux_limit=secondary)    # validates the pair
        problem, min_ratio = limits_contained(sens, self.sensitivity)
        if problem is not None:
            raise ValueError(f"{problem}; rebuild the base catalog" + (" deeper" if "deeper" in problem else ""))
        if (detection or {}).get("type", "threshold") != "threshold" and min_ratio < NOISY_BASE_MARGIN:
            warnings.warn("a noisy detection model scatters galaxies across the flux limit, but the base catalog has "
                          "none below its own limit (the selection is deterministic on the true fluxes); use a base "
                          f"catalog clearly deeper than the requested limit (it is {min_ratio:.2f} x deeper at best)",
                          stacklevel=4)
        missing = [c for c in getattr(sens, "required_columns", []) if c not in self.description["columns"]]
        if missing:
            raise KeyError(f"the sizes need the columns {missing}, which the base catalog does not store (it has "
                           f"{self.description['columns']}); add them to base.columns and rebuild it, or use size=<arcsec>")
        return sens

    def _resolve_confusion(self, confusion, sens_kw, detection, seed, n_workers, verbose, rescaling=None, doublets=None):
        """Build the line-fraction table when ``confusion`` asks for the empirical model without one (with the same
        thresholds, line rescaling and doublet rule as the catalog)."""
        if isinstance(confusion, str):
            confusion = {"type": confusion}
        if isinstance(confusion, Mapping) and confusion.get("type") == "empirical" and "file" not in confusion \
                and "table" not in confusion:
            options = {k: v for k, v in confusion.items() if k != "type"}
            unknown = sorted(set(options) - set(FRACTION_OPTIONS))
            if unknown:
                raise KeyError(f"unknown options {unknown} for the empirical confusion model; known: {list(FRACTION_OPTIONS)}, "
                               "or file / table")
            return self.line_fractions(**sens_kw, detection=detection, rescaling=rescaling, doublets=doublets, seed=seed,
                                       n_workers=n_workers, verbose=verbose, **options)
        return confusion

    def _flux_models(self, rescaling, doublets, sens) -> dict:
        """The line rescaling and the doublet rule of a catalog, checked against the base catalog, as entries of the
        worker spec: their configs (``None`` when not used), the extra columns they read and the cosmology of the
        doublet sizes."""
        model, rule = build_rescaling(rescaling), build_doublets(doublets)
        stored = self.description["columns"]
        columns = []
        if model is not None:
            if self.line_set.has_separated_components:
                raise NotImplementedError("the rescaling acts on the feature flux columns; line sets with separated "
                                          "multi-component features are not supported")
            missing = [n for n in model.required_features(self.line_set) if n not in self.line_set.names] \
                if model.galaxy_features(self.line_set) else []
            if missing:
                raise KeyError(f"the rescaling reads the fluxes of {missing}, which the base catalog does not have")
            if model.has_scatter(self.line_set):
                if "gal_id" not in stored:
                    raise KeyError(f"the scatter of the rescaling is keyed on gal_id, which the base catalog does not store ({stored})")
                columns.append("gal_id")
        cosmology = None
        if rule is not None:
            rule.check(self.line_set)
            missing = [c for c in rule.required_columns if c not in stored]
            if missing:
                raise KeyError(f"the doublet rule needs the columns {missing} (the size of each galaxy), which the base catalog "
                               f"does not store ({stored}); add them to base.columns and rebuild it, or give doublets="
                               "{'mode': 'per_galaxy', 'size': <arcsec>}")
            columns += rule.required_columns
            if rule.required_columns and rule.size.unit == "kpc":
                cosmology = self.cosmology
        self._check_rescaling_depth(model, rule, sens)
        return {"rescaling": None if model is None else model.to_config(), "doublets": None if rule is None else rule.to_config(),
                "model_columns": list(dict.fromkeys(columns)), "doublet_cosmology": cosmology}

    def _check_rescaling_depth(self, model, rule, sens) -> None:
        """Warn when the base catalog is not deep enough for the line rescaling and the doublet rule: they brighten a
        line before the selection (the rescaling by up to a 5-sigma deviate, the merged doublet by 1 + 1 / ratio), and a
        galaxy whose lines are all below the base limit before, and above the thresholds after, is not in the base."""
        boost, what = 1.0, []
        if model is not None:
            boost *= 10.0 ** model.max_log_boost(self.line_set)
            what.append("the rescaling (a 5-sigma deviate)")
        if rule is not None and rule.mode != "separated":
            boost *= 1.0 + 1.0 / rule.ratio
            what.append(f"the merged {rule.strong} + {rule.weak} doublet")
        if not what:
            return
        problem, min_ratio = limits_contained(sens, self.sensitivity)
        if problem is None and min_ratio < boost:
            warnings.warn(f"the base catalog is {min_ratio:.1f} times deeper than these thresholds, but {' with '.join(what)} "
                          f"brightens a line by up to {boost:.1f} times: galaxies lifted above the thresholds from below the base "
                          f"limit are missing. Thresholds at least {boost:.1f} times the base limit are complete; or rebuild the "
                          "base catalog deeper.", RuntimeWarning, stacklevel=4)

    def _spec(self, sens_kw, confusion, z_range, z_column, lines, membership, outcomes, n_det_min, cuts, downsample,
              columns, detection, sigma_z, seed, rescaling=None, doublets=None) -> dict:
        names = self.line_set.names
        sens = self._sensitivity(**sens_kw, detection=detection)
        flux_limit, wave_range = nominal_flux_limit(sens), tuple(sens.wave_range)
        secondary = nominal_secondary_limit(sens)
        models = self._flux_models(rescaling, doublets, sens)

        if confusion is None:
            model = default_confusion(self.line_set)
        elif isinstance(confusion, Mapping) and "type" not in confusion:
            unknown = sorted(set(confusion) - set(names))
            if unknown:
                raise KeyError(f"confusion rows for unknown lines {unknown}; the lines are {names}")
            model = ConfusionMatrix.from_dict(self.line_set, {k: dict(v) for k, v in confusion.items()}, default_p_fail=0.0)
        else:
            model = build_confusion(confusion, self.line_set)
        if list(model.names) != names:
            raise ValueError(f"the confusion model is for the lines {list(model.names)}, not {names}")
        if isinstance(model, LineFractionTable):
            _check_table(model, flux_limit, secondary, wave_range, sens.to_config(), models["rescaling"], models["doublets"])
        confusion_config = model.to_config() if hasattr(model, "to_config") else repr(model)

        if z_column not in REDSHIFT_COLUMNS:
            raise ValueError(f"z_column must be one of {REDSHIFT_COLUMNS}")
        if z_range is not None:
            z_range = tuple(None if z is None else float(z) for z in z_range)
            if len(z_range) != 2:
                raise ValueError("z_range is (min, max)")
        if membership not in ("detected", "assigned"):
            raise ValueError("membership must be 'detected' or 'assigned'")
        if isinstance(lines, str):
            lines = [lines]
        if lines is not None:
            unknown = sorted(set(lines) - set(names))
            if unknown:
                raise KeyError(f"unknown lines {unknown}; the lines are {names}")
        codes = None if outcomes is None else [int(IdOutcome[o]) if isinstance(o, str) else int(o) for o in outcomes]

        available = self.columns
        if columns is None:
            columns = [c for c in DEFAULT_COLUMNS if c in available]
        elif isinstance(columns, str):
            if columns != "all":
                raise ValueError("columns is a list of names, 'all' or None")
            columns = list(available)
        cuts = {k: tuple(v) for k, v in (cuts or {}).items()}
        missing = sorted({*columns, *cuts} - set(available))
        if missing:
            raise KeyError(f"columns {missing} are not available; the base catalog offers {available}. Add input columns "
                           "to base.columns when building it, or join on gal_id")

        populations, label = self._targets(downsample, flux_limit)
        return {
            "line_set": self.line_set.to_config(), "flux_columns": self.flux_columns, "flux_suffix": self.flux_suffix,
            "sensitivity": sens.to_config(), "flux_limit": flux_limit, "secondary_flux_limit": secondary,
            **self._size_spec(sens), **models,
            "wave_range": list(wave_range), "detection": dict(detection or {"type": "threshold"}),
            "confusion": model, "confusion_config": confusion_config, "sigma_z": float(sigma_z), "seed": int(seed),
            "z_range": None if z_range is None else list(z_range), "z_column": z_column,
            "lines": None if lines is None else list(lines),
            "line_bits": None if lines is None else [names.index(x) for x in lines], "membership": membership,
            "outcomes": None if outcomes is None else [IdOutcome(c).name for c in codes], "outcome_codes": codes,
            "n_det_min": n_det_min, "cuts": {k: list(v) for k, v in cuts.items()},
            "columns": list(columns), "populations": populations, "downsample_label": label,
        }

    def _size_spec(self, sens) -> dict:
        """What the workers need to compute the angular sizes: the size spec of the model (``None`` for
        models without one or with a fixed size) and the cosmology."""
        if not needs_size(sens):
            return {"size": None, "size_columns": [], "cosmology": None}
        return {"size": sens.size_spec, "size_columns": list(sens.required_columns), "cosmology": self.cosmology}

    def _targets(self, downsample, flux_limit: float) -> tuple[dict, object]:
        """Resolve ``downsample`` into ``{line: {"bit", "edges", "target"}}``."""
        if downsample is None or downsample is False:
            return {}, None
        from .reference_dndz import reference_target

        names = self.line_set.names
        if isinstance(downsample, str):
            downsample = {"reference": downsample}
        downsample = dict(downsample)
        if "reference" in downsample:                    # one reference for the lines it tabulates
            ref = downsample.pop("reference")
            lines = downsample.pop("lines", [n for n in ("Ha", "OIII_5007", "OIII") if n in names])
            label = {"reference": ref, **downsample}
            per_line = {line: {"reference": ref, **downsample} for line in lines}
        else:
            per_line, label = downsample, {k: (v if isinstance(v, (dict, str)) else "custom") for k, v in downsample.items()}
        out = {}
        for line, spec in per_line.items():
            if line not in names:
                raise KeyError(f"downsample target for unknown line {line!r}; the lines are {names}")
            if isinstance(spec, TargetDNDZ) or callable(spec):
                target = spec
            elif isinstance(spec, (str, Path)):
                target = TargetDNDZ.from_file(spec)
            elif "reference" in spec:
                kw = {k: v for k, v in spec.items() if k != "reference"}
                kw.setdefault("flux_limit", flux_limit)
                try:
                    target = reference_target(spec["reference"], line=kw.pop("line", line), **kw)
                except ValueError as err:
                    raise ValueError(f"{err}. Give the flux limit of the reference explicitly, e.g. "
                                     "downsample={'reference': 'wang2022', 'flux_limit': 1e-16}") from None
            elif "file" in spec:
                target = TargetDNDZ.from_file(spec["file"])
            else:
                target = TargetDNDZ(spec["z"], spec["dndz"])
            edges = getattr(target, "edges", None)
            if edges is None:
                lo, hi = target.z_range
                n = max(int(np.ceil((hi - lo) / 0.1 - 1e-9)), 1)
                edges = lo + 0.1 * np.arange(n + 1)
            edges = np.asarray(edges, dtype=np.float64)
            zc = 0.5 * (edges[1:] + edges[:-1])
            out[line] = {"bit": names.index(line), "edges": edges, "target": np.asarray(target(zc), dtype=np.float64)}
        return out, label

    # ---- the line mix of the single-line galaxies ------------------------------------
    def line_fractions(
        self,
        flux_limit: float | None = None,
        secondary_flux_limit: float | None | str = "default",
        wave_range: tuple[float, float] | None = None,
        snr: float | None = None,
        secondary_snr: float | None | str = "default",
        size=None,
        exposure_time: float | None = None,
        background: float | None = None,
        sensitivity=None,
        detection: Mapping | None = None,
        rescaling=None,
        doublets=None,
        wave_bin: float = 250.0,
        wave_edges=None,
        galaxies_per_bin: int = 500,
        n_flux_bins: int | None = None,
        flux_quantiles=None,
        min_count: int = 100,
        mode: str = "draw",
        p_fail: float = 0.0,
        seed: int = 12345,
        n_workers: int | None = None,
        write_to=None,
        verbose: bool = True,
    ) -> LineFractionTable:
        """The line mix of the single-line galaxies of the base catalog per bin of observed
        wavelength and flux rank: the :class:`~romanmock.confusion.LineFractionTable` of the
        empirical identification, ``make_catalog(confusion=table)``.

        The base catalog is scanned once at these flux limits (the same detection as
        :meth:`make_catalog`), and the single line of every single-line galaxy is kept: its
        observed wavelength, its flux and which line it is. The bandpass is cut into bins of
        ``wave_bin`` Angstrom (or at ``wave_edges``); inside each bin the galaxies are ranked by
        flux and cut into bins of equal count, as many as hold at least ``galaxies_per_bin``
        galaxies each (every cell then has between ``galaxies_per_bin`` and twice that, and a
        populated wavelength bin gets finer flux bins than a sparse one), or into exactly
        ``n_flux_bins`` bins, or at the quantiles ``flux_quantiles`` (``[0, 0.5, 0.9, 0.99, 1]``
        resolves the bright tail, where the rare lines are). Lines that are the single line of
        fewer than ``min_count`` galaxies in all
        are left out of the identification. Build the table at the thresholds of the catalog it
        is for (``flux_limit``, ``secondary_flux_limit`` and ``wave_range``, or the S/N route
        ``snr``, ``secondary_snr``, ``size``, ``exposure_time``, ``background``, or a
        ``sensitivity`` model, as in :meth:`make_catalog`), and with its ``rescaling`` of the
        line fluxes and ``doublets`` rule; ``make_catalog`` warns when they differ.

        ``mode`` is ``"draw"`` (a galaxy gets each line with the probability of its cell) or
        ``"max"`` (the most common line of its cell); ``p_fail`` a uniform probability of no
        redshift. ``write_to`` saves the table (``.npz``) for ``confusion={"type": "empirical",
        "file": ...}`` and the YAML configs of the pipeline. ``seed`` matters only for noisy
        detection models. ``verbose`` prints :meth:`LineFractionTable.summary`: the galaxies
        and the shares per wavelength bin, and the smallest cell, to judge the binning.
        """
        sens = self._sensitivity(flux_limit, secondary_flux_limit, wave_range, sensitivity, snr, secondary_snr, size,
                                 exposure_time, background, detection=detection)
        flux_limit, wave_range = nominal_flux_limit(sens), tuple(sens.wave_range)
        secondary = nominal_secondary_limit(sens)
        if wave_edges is None:
            n_bins = max(1, int(round((wave_range[1] - wave_range[0]) / float(wave_bin))))
            wave_edges = np.linspace(wave_range[0], wave_range[1], n_bins + 1)
        wave_edges = np.asarray(wave_edges, dtype=np.float64)
        comp_cols = {c: f"{c}{self.flux_suffix}" for feat in self.line_set if not feat.is_blend for c in feat.components} \
            if self.line_set.has_separated_components else {}
        spec = {"line_set": self.line_set.to_config(), "flux_columns": self.flux_columns, "component_columns": comp_cols,
                "flux_suffix": self.flux_suffix, "wave_range": list(wave_range),
                "sensitivity": sens.to_config(), "detection": dict(detection or {"type": "threshold"}), "seed": int(seed),
                **self._size_spec(sens), **self._flux_models(rescaling, doublets, sens)}
        if n_workers is None:
            n_workers = suggest_n_workers(max_workers=32, per_worker_gb=2.5)
        tasks = [(spec, str(self.dir / v["output"]), step) for step, v in self.slices.items()]
        with Progress(f"line mix of the single-line galaxies: scanning {len(tasks)} slices with {n_workers} worker(s)", verbose):
            results = _run(tasks, n_workers, _single_lines_slice)
        wave, flux, line = (np.concatenate([r[k] for r in results]) for k in ("wave", "flux", "line"))
        settings = {"base": str(self.dir), "base_tag": self.tag, "sensitivity": sens.to_config(), "flux_limit": flux_limit,
                    "secondary_flux_limit": secondary, "rescaling": spec["rescaling"], "doublets": spec["doublets"],
                    "wave_range": list(wave_range), "detection": spec["detection"], "wave_bin": float(np.median(np.diff(wave_edges))),
                    "n_base": int(sum(r["n"] for r in results)),
                    "built": _dt.datetime.now().isoformat(timespec="seconds")}
        table = LineFractionTable.from_single_lines(self.line_set, wave, flux, line, wave_edges, galaxies_per_bin, n_flux_bins,
                                                    flux_quantiles, min_count, mode, p_fail, settings)
        if verbose:
            print("\n" + table.summary() + "\n", flush=True)
        if write_to is not None:
            path = table.save(write_to)
            if verbose:
                print(f"[romanmock] wrote {path}", flush=True)
        return table

    # ---- step 1 of a catalog with interlopers: the proposal ----------------------------
    def make_proposal(
        self,
        z_range: tuple[float, float],
        line: str = "Ha",
        flux_limit: float | None = None,
        secondary_flux_limit: float | None | str = "default",
        snr: float | None = None,
        secondary_snr: float | None | str = "default",
        size=None,
        exposure_time: float | None = None,
        background: float | None = None,
        sensitivity=None,
        wave_range: tuple[float, float] | None = None,
        rescaling="galacticus_like",
        doublets="per_galaxy",
        dz: float = 0.01,
        min_count: int = 100,
        min_share: float = 1e-3,
        interloper_fraction: float | None = None,
        secondary_ratios: Sequence[float] = (0.3, 0.4, 0.5),
        detection: Mapping | None = None,
        seed: int = 12345,
        n_workers: int | None = None,
        verbose: bool = True,
    ):
        """Step 1 of a catalog with interlopers: the single-line galaxies it would hold at proposed thresholds, before
        their identification is chosen (:class:`~romanmock.proposal.Proposal`).

        Without a secondary limit and without ``interloper_fraction`` it is a proposal for single-line clean catalogs
        instead, without interlopers (see ``secondary_flux_limit`` below).

        One pass over the base catalog (the slices that matter) counts, per bin of true redshift: the single-line
        galaxies inside ``z_range`` and which line is single, and the potential interlopers, single-line galaxies at
        other redshifts whose line falls where ``line`` would be for a redshift inside ``z_range``. The same pass scans
        the secondary limit from the primary one down to a hundredth of it: the summary shows, for each potential
        interloper and in total, the most they can make of the catalog against the secondary limit
        (``proposal.plot_secondary()`` draws it), and where an expected ``interloper_fraction`` becomes possible.
        Step 2 is ``proposal.make_catalog(fractions={line: {interloper: share}}, out_dir=...)`` (a catalog designed by
        its interloper fractions, at a secondary limit of your choice) or ``proposal.make_catalog(confusion, out_dir)``.

        Parameters
        ----------
        z_range : (min, max)
            Redshift range of the catalog (on the assigned redshift).
        line : str
            Which galaxies the catalog holds. A line of ``base.line_set.names`` (``"Ha"``, the default, ``"OIII_5007"``,
            ...): the galaxies in which that line is identified, their redshift from it (observed wavelength / rest
            wavelength - 1) inside ``z_range``: the line detected, alone or with other lines (a galaxy with Hα and
            [O III] belongs to the catalog of each), or a single line taken for it (the interlopers). ``"all"``: every
            galaxy with a redshift inside ``z_range``, whichever line gave it; its sample is every detected galaxy
            inside the range, and its potential interlopers are the single lines that, taken for a line that is single
            in a major share of the galaxies inside the range (:attr:`Proposal.assignable`), get a redshift inside it.
            ``verbose`` prints this as a hint.
        flux_limit, secondary_flux_limit, snr, secondary_snr, size, exposure_time, background, sensitivity, wave_range
            The proposed thresholds, as in :meth:`make_catalog`: flux limits, or S/N in the Roman grism, or a
            sensitivity model. One of them must be given. Without a secondary limit (``secondary_flux_limit`` or
            ``secondary_snr`` left out, or None) and with ``interloper_fraction``, the proposal is for catalogs by
            design with a single tier of its own (the secondary limit at the primary one, the right end of the scan),
            and step 2 chooses the secondary limit. Without a secondary limit and without ``interloper_fraction``, the
            proposal is for a single-line clean catalog, without interlopers:
            the galaxies with ``line`` above the primary limit (``line="all"``: every detected galaxy), each with its
            true redshift. The summary gives, against redshift, the share of them that are true single-line galaxies,
            without another line above each of the ``secondary_ratios``. ``proposal.make_catalog(out_dir=...)`` makes
            the catalog with one line per galaxy, every one a single-line galaxy (the target line, even where
            [O III] is brighter; ``line="all"``: the brightest); ``proposal.make_catalog(secondary_flux_limit=...,
            out_dir=...)`` the same galaxies with their other lines above that limit, so that only those without
            another line are single-line galaxies.
        rescaling : str, dict, model or None
            Rescaling of the line fluxes; default ``"galacticus_like"`` (the diffsky lines toward Galacticus).
        doublets : str, dict, DoubletRule or None
            [O III] λ4959 a line of its own or not; default ``"per_galaxy"`` (where the doublet is resolved for the
            galaxy's size).
        dz : float
            Width of the redshift bins (``z_range`` is cut into equal bins of about ``dz``).
        min_count, min_share : int, float
            A line counts when it is the single line of at least ``min_count`` galaxies and ``min_share`` of the
            detected galaxies inside the range (single lines inside it), respectively of the target sample
            (potential interlopers); the others are minor and left out of the figures and of the confusion template.
        interloper_fraction : float
            The total interloper fraction expected in the catalog (a share of it, e.g. 0.01): the summary and
            ``plot_secondary()`` say from which secondary limit the potential interlopers can make it, and
            ``proposal.fraction_template()`` shares it among them. The secondary limit given here only fixes the
            tables against redshift (and the default of step 2; without one, a single tier); the scan covers every
            other.
        secondary_ratios : list of float
            Secondary limits, as fractions of the primary one, at which a single-line proposal (no secondary limit)
            reports the true single-line fraction against redshift (0.3, 0.4 and 0.5 times the primary limit).
        detection, seed, n_workers, verbose : as in :meth:`make_catalog`
        """
        from .proposal import (ALL_LINES, PROPOSAL_STREAM, SCAN, Proposal, add_counts, line_hint, new_settings, proposal_grid,
                               scan_edges, secondary_ratio, target_window, target_windows, window_intervals)

        names = self.line_set.names
        if line != ALL_LINES and line not in names:
            raise KeyError(f"unknown line {line!r}; the lines are {names}, or {ALL_LINES!r} for every galaxy")
        if z_range is None or len(z_range) != 2 or None in tuple(z_range) or not float(z_range[0]) < float(z_range[1]):
            raise ValueError("z_range is (min, max) with min < max")
        z_range = (float(z_range[0]), float(z_range[1]))
        if interloper_fraction is not None and not 0.0 < float(interloper_fraction) < 1.0:
            raise ValueError(f"interloper_fraction is a share of the catalog, between 0 and 1; got {interloper_fraction}")
        given = [flux_limit is not None, snr is not None, sensitivity is not None, size is not None,
                 not (isinstance(secondary_flux_limit, str) and secondary_flux_limit == "default"),
                 not (isinstance(secondary_snr, str) and secondary_snr == "default")]
        if not any(given):
            raise ValueError("propose the thresholds: flux_limit and secondary_flux_limit, or snr, secondary_snr and size "
                             "(S/N in the Roman grism), or a sensitivity model")
        ratios = [float(r) for r in secondary_ratios]
        if not ratios or not all(0.0 < r <= 1.0 for r in ratios):
            raise ValueError(f"secondary_ratios are secondary limits as fractions of the primary one, in (0, 1]; got {ratios}")
        unset = lambda v: v is None or (isinstance(v, str) and v == "default")
        no_secondary = sensitivity is None and unset(secondary_flux_limit) and unset(secondary_snr)
        single_line = no_secondary and interloper_fraction is None
        if no_secondary:                                 # a single tier: no secondary limit of its own
            if any(v is not None for v in (snr, size, exposure_time, background)) or secondary_snr is None:
                secondary_snr = None
            else:
                secondary_flux_limit = None
        if single_line:                                  # single-line clean catalogs: one line per galaxy
            detection = {**dict(detection or {"type": "threshold"}), "single_line": True if line == ALL_LINES else line}
        sens = self._sensitivity(flux_limit, secondary_flux_limit, wave_range, sensitivity, snr, secondary_snr, size,
                                 exposure_time, background, detection=detection)
        models = self._flux_models(rescaling, doublets, sens)
        wave_range = tuple(sens.wave_range)
        rest = self.line_set.rest_waves
        if line == ALL_LINES:
            windows = target_windows(names, rest, z_range, wave_range)
            if not windows:
                raise ValueError(f"no line is inside the bandpass {list(wave_range)} at {list(z_range)}")
        else:
            windows = {line: target_window(rest, names.index(line), z_range, wave_range)}
        z_hi = max(v["z_max"] for v in self.slices.values())
        z_edges = proposal_grid(z_range, dz, z_hi + RSD_MARGIN * (1 + z_hi))
        # the slices that can hold a galaxy of the range or a potential interloper (with peculiar velocities)
        intervals = [z_range]
        for window in windows.values():
            intervals += [(max(a, wave_range[0] / r - 1), min(b, wave_range[1] / r - 1))
                          for (a, b), r in zip(window_intervals(names, rest, window).values(), rest)]
        intervals = [(a, b) for a, b in intervals if b > a]

        def needed(v):
            margin = RSD_MARGIN * (1 + v["z_max"])
            return any(v["z_max"] + margin >= a and v["z_min"] - margin <= b for a, b in intervals)

        comp_cols = {c: f"{c}{self.flux_suffix}" for feat in self.line_set if not feat.is_blend for c in feat.components} \
            if self.line_set.has_separated_components else {}
        spec = {"line_set": self.line_set.to_config(), "flux_columns": self.flux_columns, "component_columns": comp_cols,
                "flux_suffix": self.flux_suffix, "wave_range": list(wave_range), "sensitivity": sens.to_config(),
                "detection": dict(detection or {"type": "threshold"}), "seed": int(seed), **self._size_spec(sens), **models,
                "targets": list(windows), "windows": [list(w) for w in windows.values()], "z_edges": z_edges,
                "stream": PROPOSAL_STREAM, "z_range": list(z_range), "q_show": ratios}
        scan = {**SCAN, "extra": [] if single_line else [secondary_ratio(sens.to_config())]}
        spec["q_edges"] = scan_edges(**scan)
        if n_workers is None:
            n_workers = suggest_n_workers(max_workers=32, per_worker_gb=2.5)
        tasks = [(spec, str(self.dir / v["output"]), step) for step, v in self.slices.items() if needed(v)]
        if verbose:
            print(line_hint(line, names, rest), flush=True)
            if single_line:
                print(f"{PREFIX} no secondary limit: a single-line clean catalog, no interlopers; the summary gives the true "
                      f"single-line share at {', '.join(f'{r:g}' for r in ratios)} times the primary limit", flush=True)
        with Progress(f"proposal: scanning {len(tasks)} of {len(self.slices)} slices with {n_workers} worker(s)", verbose):
            results = _run(tasks, n_workers, _proposal_slice)
        counts = None
        for r in results:
            if r["counts"] is not None:
                counts = add_counts(counts, r["counts"])
        if counts is None:
            raise ValueError("no galaxy of the base catalog falls in the redshifts of this proposal")
        settings = new_settings(line=line, z_range=list(z_range), targets=list(windows),
                                windows=[list(w) for w in windows.values()], sensitivity=sens.to_config(),
                                flux_limit=nominal_flux_limit(sens), secondary_flux_limit=nominal_secondary_limit(sens),
                                wave_range=list(wave_range), rescaling=models["rescaling"], doublets=models["doublets"],
                                detection=spec["detection"], seed=int(seed), dz=float(dz), min_count=int(min_count),
                                min_share=float(min_share), base=str(self.dir), base_tag=self.tag,
                                n_base=int(sum(r["n"] for r in results)), scan=scan, secondary_ratios=ratios,
                                interloper_fraction=None if interloper_fraction is None else float(interloper_fraction))
        proposal = Proposal(counts, z_edges, settings, names, rest, self.area_deg2, base=self)
        if verbose:
            print("\n" + proposal.summary(), flush=True)
        return proposal


def make_catalog(base_dir, **settings) -> Catalog:
    """``BaseCatalog(base_dir).make_catalog(**settings)``."""
    return BaseCatalog(base_dir).make_catalog(**settings)


def make_proposal(base_dir, z_range, **settings):
    """``BaseCatalog(base_dir).make_proposal(z_range, **settings)``: step 1 of a catalog with interlopers."""
    return BaseCatalog(base_dir).make_proposal(z_range, **settings)


def _check_table(table: LineFractionTable, flux_limit: float, secondary: float | None, wave_range, sensitivity: dict | None = None,
                 rescaling: dict | None = None, doublets: dict | None = None) -> None:
    """Warn when a line-fraction table was built at other limits, or with another line rescaling or doublet rule,
    than the catalog uses: its flux-rank bins then describe another population."""
    s = table.settings
    if "flux_limit" not in s and "sensitivity" not in s:
        return
    mismatch = []
    same = lambda a, b: json.dumps(a, sort_keys=True, default=_json_default) == json.dumps(b, sort_keys=True, default=_json_default)
    for key, value in (("rescaling", rescaling), ("doublets", doublets)):
        if not same(s.get(key), value):
            mismatch.append(f"{key} {s.get(key)} (table) vs {value} (catalog)")
    if sensitivity is not None and s.get("sensitivity") is not None:
        same = json.dumps(s["sensitivity"], sort_keys=True, default=_json_default) == json.dumps(sensitivity, sort_keys=True, default=_json_default)
        if not same:
            mismatch.append(f"sensitivity {describe_sensitivity(s['sensitivity'])} (table) vs {describe_sensitivity(sensitivity)} (catalog)")
    elif "flux_limit" in s:
        if not np.isclose(s["flux_limit"], flux_limit, rtol=1e-6, atol=0.0):       # fluxes ~1e-16: the default atol would hide it
            mismatch.append(f"flux_limit {s['flux_limit']:.3g} (table) vs {flux_limit:.3g} (catalog)")
        if (s.get("secondary_flux_limit") is None) != (secondary is None) or \
                (secondary is not None and not np.isclose(s["secondary_flux_limit"], secondary, rtol=1e-6, atol=0.0)):
            mismatch.append(f"secondary_flux_limit {s.get('secondary_flux_limit')} (table) vs {secondary} (catalog)")
    if "wave_range" in s and not np.allclose(s["wave_range"], wave_range, rtol=1e-6, atol=0.0):
        mismatch.append(f"wave_range {s['wave_range']} (table) vs {list(wave_range)} (catalog)")
    if mismatch:
        warnings.warn("the line-fraction table was built for other single-line galaxies than this catalog has: "
                      + "; ".join(mismatch) + ". Build it with BaseCatalog.line_fractions at the limits of the catalog.",
                      RuntimeWarning, stacklevel=4)


def make_random(catalog, **settings):
    """``catalog.make_random(**settings)``: the random catalog of a data catalog, a :class:`Catalog` (a whole catalog or
    a part of one) or a file written by one."""
    if not isinstance(catalog, Catalog):
        catalog = Catalog.read(catalog)
    return catalog.make_random(**settings)


# =====================================================================================
# workers
# =====================================================================================
def _catalog_slice(spec: dict, path: str, step: int) -> dict:
    """Detection, redshift assignment and cuts for one slice of the base catalog.

    The slice is read and processed in chunks of :data:`CHUNK_ROWS` rows, so that the memory of a
    worker does not grow with the slice. The random draws are consumed in row order, so the
    result equals that of one pass over the slice.
    """
    import h5py

    line_set = get_line_set(spec["line_set"])
    flux_columns = spec["flux_columns"]
    comp_cols = {c: f"{c}{spec['flux_suffix']}" for feat in line_set if not feat.is_blend for c in feat.components} \
        if line_set.has_separated_components else {}
    rescaling, doublets = build_rescaling(spec.get("rescaling")), build_doublets(spec.get("doublets"))
    derived = set(DERIVED_COLUMNS)
    wanted = [c for c in dict.fromkeys([*spec["columns"], *spec["cuts"], spec["z_column"]]) if c not in derived]
    read = list(dict.fromkeys([*flux_columns, *comp_cols.values(), "redshift_rsd", *spec["size_columns"],
                               *spec.get("model_columns", []), *wanted]))
    rng = np.random.default_rng([spec["seed"], int(step)])              # the pipeline's stream for this slice
    rng_thin = np.random.default_rng([spec["seed"], THINNING_STREAM, int(step)])   # the stream `romanmock select` uses
    model = build_detection(spec["detection"], build_sensitivity(spec["sensitivity"]))
    multi_line = PerfectMultiLine(spec["sigma_z"])
    pops = spec["populations"]
    n_lines = len(line_set)

    parts = []
    with h5py.File(path, "r") as h:
        group = base_data_group(h)
        n = group["redshift_rsd"].shape[0]
        for start in range(0, n, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, n)
            data = {c: group[c][start:stop] for c in read}
            data, z_rsd, flux, merged = _chunk_fluxes(spec, line_set, data, rescaling, doublets)
            comp = {lum: np.asarray(data[name], dtype=np.float64) for lum, name in comp_cols.items()} or None
            det = detect_lines(flux, z_rsd, line_set, model, rng, component_flux=comp, size=_sizes(spec, data))
            za = assign_redshifts(det, z_rsd, line_set, spec["confusion"], rng, multi_line)
            cols = dict(data)
            cols.update(redshift_assigned=za.z_assigned, id_outcome=za.outcome, line_bitmask=det.bitmask, n_det=det.n_det,
                        line_true=za.line_true, line_assigned=za.line_assigned,
                        doublet_merged=np.zeros(stop - start, dtype=bool) if merged is None else merged)
            detected = det.bitmask > 0
            part = {"n": stop - start, "n_detected": int(detected.sum()), "n_fallback_size": _n_fallback_size(spec, data)}
            if pops:
                part["hist"], part["n_pop"] = {}, {}
                for name, pop in pops.items():
                    zs = z_rsd[detected & (za.line_true == pop["bit"])]
                    part["n_pop"][name] = int(zs.size)
                    part["hist"][name] = np.histogram(zs[np.isfinite(zs)], bins=pop["edges"])[0]
                u = np.zeros(stop - start)                       # one uniform deviate per detected galaxy, in order
                u[detected] = rng_thin.random(int(detected.sum()))

            keep = detected.copy()
            if spec["outcome_codes"] is not None:
                keep &= np.isin(za.outcome, spec["outcome_codes"])
            if spec["z_range"] is not None:
                z = np.asarray(cols[spec["z_column"]], dtype=np.float64)
                lo, hi = spec["z_range"]
                keep &= np.isfinite(z)
                if lo is not None:
                    keep &= z >= lo
                if hi is not None:
                    keep &= z <= hi
            if spec["line_bits"] is not None:
                keep &= tracer_membership(cols, line_set, spec["lines"], spec["membership"])
            if spec["n_det_min"] is not None:
                keep &= det.n_det >= spec["n_det_min"]
            for col, (lo, hi) in spec["cuts"].items():
                v = np.asarray(cols[col])
                if lo is not None:
                    keep &= v >= lo
                if hi is not None:
                    keep &= v <= hi

            part["columns"] = {c: np.ascontiguousarray(np.asarray(cols[c])[keep]) for c in spec["columns"]}
            part["ident"] = {"line_true": za.line_true[keep], "line_assigned": za.line_assigned[keep],
                             "outcome": za.outcome[keep], "bitmask": np.asarray(det.bitmask)[keep]}
            if pops:
                part["u"], part["population"], part["z_pop"] = u[keep], za.line_true[keep].astype(np.int16), z_rsd[keep]
            parts.append(part)

    cat = lambda key: np.concatenate([p[key] for p in parts]) if parts else np.zeros(0)
    out = {"step": int(step), "n_base": int(sum(p["n"] for p in parts)), "n_detected": int(sum(p["n_detected"] for p in parts)),
           "n_fallback_size": int(sum(p["n_fallback_size"] for p in parts)),
           "columns": {c: (np.concatenate([p["columns"][c] for p in parts]) if parts else np.zeros(0)) for c in spec["columns"]}}
    ident = {k: (np.concatenate([p["ident"][k] for p in parts]) if parts else np.zeros(0, dtype=np.int64)) for k in
             ("line_true", "line_assigned", "outcome", "bitmask")}
    if pops:
        out["ident"] = ident                       # counted after the thinning
        out["hist"] = {name: np.sum([p["hist"][name] for p in parts], axis=0) for name in pops}
        out["n_pop"] = {name: int(sum(p["n_pop"][name] for p in parts)) for name in pops}
        out["u"], out["population"], out["z_pop"] = cat("u"), cat("population"), cat("z_pop")
    else:
        out["line_counts"] = line_counts(n_lines=n_lines, **ident)
    return out


def _chunk_fluxes(spec: dict, line_set, data: dict, rescaling, doublets):
    """The line fluxes of a chunk as the detection sees them: the rescaling applied to the flux columns of ``data``
    (the returned ``data`` carries the rescaled fluxes), the feature fluxes stacked, the doublet rule applied.
    Returns ``(data, z_rsd, flux, merged)``; ``merged`` is ``None`` without a doublet rule."""
    if rescaling is not None:
        data = rescaling.apply_columns(data, line_set, suffix=spec["flux_suffix"])
    z_rsd = np.asarray(data["redshift_rsd"], dtype=np.float64)
    flux = np.stack([np.asarray(data[c], dtype=np.float64) for c in spec["flux_columns"]], axis=1)
    merged = None
    if doublets is not None:
        flux, merged = doublets.apply(flux, z_rsd, line_set, spec["wave_range"], data, spec.get("doublet_cosmology"))
    return data, z_rsd, flux, merged


def _sizes(spec: dict, data: Mapping[str, np.ndarray]):
    """Angular sizes [arcsec] of the galaxies of a chunk for a size-dependent sensitivity, else ``None``."""
    if spec.get("size") is None:
        return None
    return spec["size"].angular_size(data, spec["cosmology"])


def _n_fallback_size(spec: dict, data: Mapping[str, np.ndarray]) -> int:
    """Galaxies of a chunk whose size is invalid and took the fallback size (0 without a size column or fallback)."""
    size = spec.get("size")
    if size is None or size.is_fixed or size.fallback is None:
        return 0
    return size.n_invalid(data, spec["cosmology"])


def base_data_group(h):
    """The HDF5 group with the columns of a base-catalog file (``<step>_data/data`` as OpenCosmo writes it)."""
    import h5py

    for key in ["data", *h.keys()]:
        node = h.get(key)
        if isinstance(node, h5py.Group):
            if "redshift_rsd" in node:
                return node
            if isinstance(node.get("data"), h5py.Group) and "redshift_rsd" in node["data"]:
                return node["data"]
    raise KeyError(f"{h.filename}: no data group with the column redshift_rsd")


def _single_lines_slice(spec: dict, path: str, step: int) -> dict:
    """Worker of :meth:`BaseCatalog.line_fractions`: the single line (wavelength, flux, index) of every
    single-line galaxy of one slice, detected as :func:`_catalog_slice` does it."""
    import h5py

    line_set = get_line_set(spec["line_set"])
    flux_columns, comp_cols = spec["flux_columns"], spec["component_columns"]
    rescaling, doublets = build_rescaling(spec.get("rescaling")), build_doublets(spec.get("doublets"))
    read = list(dict.fromkeys([*flux_columns, *comp_cols.values(), "redshift_rsd", *spec["size_columns"],
                               *spec.get("model_columns", [])]))
    rng = np.random.default_rng([spec["seed"], FRACTION_STREAM, int(step)])
    model = build_detection(spec["detection"], build_sensitivity(spec["sensitivity"]))
    waves, fluxes, lines, n = [], [], [], 0
    with h5py.File(path, "r") as h:
        group = base_data_group(h)
        n = group["redshift_rsd"].shape[0]
        for start in range(0, n, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, n)
            data = {c: group[c][start:stop] for c in read}
            data, z_rsd, flux, _ = _chunk_fluxes(spec, line_set, data, rescaling, doublets)
            comp = {lum: np.asarray(data[name], dtype=np.float64) for lum, name in comp_cols.items()} or None
            det = detect_lines(flux, z_rsd, line_set, model, rng, component_flux=comp, size=_sizes(spec, data))
            w, f, i = single_lines(det)
            waves.append(w); fluxes.append(f); lines.append(i.astype(np.int8))
    cat = lambda parts, dtype: np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)
    return {"step": int(step), "n": int(n), "wave": cat(waves, np.float64), "flux": cat(fluxes, np.float64), "line": cat(lines, np.int8)}


def _proposal_slice(spec: dict, path: str, step: int) -> dict:
    """Worker of :meth:`BaseCatalog.make_proposal`: the counts of :func:`~romanmock.proposal.proposal_counts` for one
    slice, detected as :func:`_catalog_slice` does it."""
    import h5py

    from .proposal import add_counts, primary_snr, proposal_counts

    line_set = get_line_set(spec["line_set"])
    flux_columns, comp_cols = spec["flux_columns"], spec["component_columns"]
    rescaling, doublets = build_rescaling(spec.get("rescaling")), build_doublets(spec.get("doublets"))
    read = list(dict.fromkeys([*flux_columns, *comp_cols.values(), "redshift_rsd", *spec["size_columns"],
                               *spec.get("model_columns", [])]))
    rng = np.random.default_rng([spec["seed"], spec["stream"], int(step)])
    model = build_detection(spec["detection"], build_sensitivity(spec["sensitivity"]))
    lo, hi = spec["wave_range"]
    counts, n = None, 0
    with h5py.File(path, "r") as h:
        group = base_data_group(h)
        n = group["redshift_rsd"].shape[0]
        for start in range(0, n, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, n)
            data = {c: group[c][start:stop] for c in read}
            data, z_rsd, flux, merged = _chunk_fluxes(spec, line_set, data, rescaling, doublets)
            comp = {lum: np.asarray(data[name], dtype=np.float64) for lum, name in comp_cols.items()} or None
            det = detect_lines(flux, z_rsd, line_set, model, rng, component_flux=comp, size=_sizes(spec, data))
            both = None
            if merged is not None:
                s, w = doublets.check(line_set)
                both = np.all([(line_set.rest_waves[k] * (1 + z_rsd) > lo) & (line_set.rest_waves[k] * (1 + z_rsd) < hi)
                               for k in (s, w)], axis=0)
            part = proposal_counts(det, z_rsd, line_set, spec["targets"], spec["windows"], spec["z_edges"], merged, both,
                                   primary_snr=None if spec.get("q_edges") is None else primary_snr(model),
                                   q_edges=spec.get("q_edges"), z_range=spec.get("z_range"), q_show=spec.get("q_show"))
            counts = add_counts(counts, part)
    return {"step": int(step), "n": int(n), "counts": counts}


def _run(tasks, n_workers: int, worker=_catalog_slice) -> list[dict]:
    if n_workers <= 1 or len(tasks) <= 1:
        return [worker(*t) for t in tasks]
    import concurrent.futures as cf
    import multiprocessing as mp

    with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(worker, *t) for t in tasks]
        return [f.result() for f in futures]           # in slice order


def _thin(results, populations: dict, spec: dict, area: float) -> tuple[np.ndarray, dict]:
    """Keep-mask of the selected rows after thinning the emitter populations to their targets."""
    u = np.concatenate([r["u"] for r in results])
    member = np.concatenate([r["population"] for r in results])
    z = np.concatenate([r["z_pop"] for r in results])
    keep = np.ones(u.size, dtype=bool)
    info = {}
    for name, pop in populations.items():
        n_before = sum(r["n_pop"][name] for r in results)
        if n_before == 0:
            continue
        edges = pop["edges"]
        zc = 0.5 * (edges[1:] + edges[:-1])
        counts = np.sum([r["hist"][name] for r in results], axis=0)
        mock = dndz_from_counts(counts, edges, area)
        empty = (mock == 0) & (pop["target"] > 0)
        if np.any(empty):
            warnings.warn(f"{name}: {empty.sum()} of {empty.size} target bins (z {edges[:-1][empty].min():.2f}-"
                          f"{edges[1:][empty].max():.2f}) contain no galaxies whose brightest line is {name}; "
                          "the target cannot be applied there.", RuntimeWarning, stacklevel=3)
        p_keep = keep_probability(mock, pop["target"], zc, warn_empty=False, stacklevel=4)
        sel = member == pop["bit"]
        in_range = (z >= edges[0]) & (z <= edges[-1])
        p = np.where(in_range, probability_per_galaxy(z, edges, p_keep), 1.0)     # outside the target: kept
        keep[sel] = (u < p)[sel]
        expected = float(np.sum(counts * p_keep))
        info[name] = {"z_edges": edges.tolist(), "p_keep": p_keep.tolist(), "mock_dndz": mock.tolist(),
                      "target_dndz": pop["target"].tolist(), "n_before": int(n_before),
                      "n_after": int(round(n_before - counts.sum() + expected)), "area_deg2": float(area)}
    return keep, info


# =====================================================================================
# the line report
# =====================================================================================
def line_counts(line_true, line_assigned, outcome, bitmask, n_lines: int) -> dict:
    """Counts behind the line report, for the galaxies of a catalog.

    ``detected[i]``: galaxies in which line ``i`` is detected; ``identified[i]``: those of them in
    which it is identified as line ``i`` (galaxies with a correct redshift: every detected line of a
    multi-line galaxy, the line of a correctly identified single-line galaxy); ``brightest[i]``:
    galaxies with a correct redshift whose brightest line is ``i``; ``misid[i, j]``: single-line
    galaxies with true line ``i`` taken for ``j``.
    """
    line_true = np.asarray(line_true, dtype=np.int64)
    line_assigned = np.asarray(line_assigned, dtype=np.int64)
    bitmask = np.asarray(bitmask).astype(np.int64)
    wrong = np.asarray(outcome) == int(IdOutcome.SINGLE_MISID)
    pair = line_true[wrong] * n_lines + line_assigned[wrong]
    right = np.isin(outcome, [int(IdOutcome.MULTI_LINE), int(IdOutcome.SINGLE_CORRECT)])
    return {"detected": np.array([int(np.count_nonzero((bitmask >> i) & 1)) for i in range(n_lines)], dtype=np.int64),
            "identified": np.array([int(np.count_nonzero((bitmask[right] >> i) & 1)) for i in range(n_lines)], dtype=np.int64),
            "brightest": np.bincount(line_true[right & (line_true >= 0)], minlength=n_lines).astype(np.int64),
            "misid": np.bincount(pair, minlength=n_lines * n_lines).reshape(n_lines, n_lines).astype(np.int64)}


def confusion_probabilities(model) -> np.ndarray | None:
    """``P[i, j]`` = probability that a single line ``i`` is taken for ``j`` (the largest one over the
    wavelength bins of a binned model; for a :class:`LineFractionTable` the expectation over the
    single-line galaxies of line ``i``; for a :class:`WindowedConfusion` the probability inside the window of the
    pair); ``None`` for a model that is none of these."""
    if isinstance(model, ConfusionMatrix):
        return model.matrix[:, :-1].copy()
    if hasattr(model, "pair_probabilities"):
        return model.pair_probabilities()
    if hasattr(model, "expected_confusion"):
        return model.expected_confusion().matrix[:, :-1].copy()
    matrices = getattr(model, "matrices", None)
    if matrices and all(isinstance(m, ConfusionMatrix) for m in matrices):
        return np.max([m.matrix[:, :-1] for m in matrices], axis=0)
    return None


def _probability_kind(model) -> str:
    """What the probabilities of the line report are: a matrix entry, the expectation of a line-fraction table, or the
    probability inside the window of a pair."""
    if isinstance(model, LineFractionTable):
        return "expected"
    return "windowed" if isinstance(model, WindowedConfusion) else "matrix"


def line_report(names, rest_waves, wave_range, z_range, z_column, z_cover, assigned_bits=None, probabilities=None,
                counts=None, outcome_codes=None, membership: str = "detected", probability_kind: str = "matrix") -> dict:
    """Lines inside the bandpass for a redshift cut, and the interlopers the cut lets in.

    A line ``i`` observed at ``wave`` and taken for line ``j`` gets the redshift ``wave / rest_j - 1``.
    With the cut on the assigned redshift, the interlopers of ``j`` are the lines ``i`` whose observed
    wavelength lies where ``j`` would be for a redshift inside the cut; with the cut on a true
    redshift, they are the lines inside the cut that are taken for another line. ``z_cover`` is
    the redshift range of the lightcone.
    """
    rest = np.asarray(rest_waves, dtype=np.float64)
    n = rest.size
    z_lo = -1.0 if z_range is None or z_range[0] is None else float(z_range[0])
    z_hi = np.inf if z_range is None or z_range[1] is None else float(z_range[1])
    on_assigned = z_column == "redshift_assigned"

    def window(true, cut):
        """Observed wavelengths of line ``true`` with line ``cut`` inside the redshift cut."""
        lo = max(wave_range[0], rest[cut] * (1 + z_lo), rest[true] * (1 + z_cover[0]))
        hi = min(wave_range[1], rest[cut] * (1 + z_hi), rest[true] * (1 + z_cover[1]))
        return (lo, hi) if hi > lo else None

    def z_of(wave, k):
        return [float(wave[0] / rest[k] - 1), float(wave[1] / rest[k] - 1)]

    lines = []
    for k in range(n):
        wave = window(k, k)
        if wave is None:
            continue
        row = {"line": names[k], "rest_wave": float(rest[k]), "z": z_of(wave, k), "wave_obs": list(map(float, wave)),
               "selected": assigned_bits is None or k in assigned_bits}
        if counts is not None:
            row.update(n_detected=int(counts["detected"][k]), n_identified=int(counts["identified"][k]),
                       n_brightest=int(counts["brightest"][k]), n_mistaken=int(counts["misid"][k].sum()),
                       n_interlopers=int(counts["misid"][:, k].sum()))
        lines.append(row)

    interlopers = []
    for j in range(n):
        if assigned_bits is not None and j not in assigned_bits:
            continue
        for i in range(n):
            wave = None if i == j else window(i, j if on_assigned else i)
            if wave is None:
                continue
            row = {"line_true": names[i], "line_assigned": names[j], "z_true": z_of(wave, i), "z_assigned": z_of(wave, j),
                   "wave_obs": list(map(float, wave)),
                   "probability": None if probabilities is None else float(probabilities[i, j])}
            if counts is not None:
                own = counts["identified" if membership == "detected" else "brightest"][j]
                row.update(n=int(counts["misid"][i, j]), n_line=int(own + counts["misid"][:, j].sum()))
            interlopers.append(row)
    kept = outcome_codes is None or int(IdOutcome.SINGLE_MISID) in outcome_codes
    return {"wave_range": list(map(float, wave_range)), "z_range": None if z_range is None else list(z_range),
            "z_column": z_column, "z_cover": list(map(float, z_cover)), "lines": lines, "interlopers": interlopers,
            "misidentified_kept": bool(kept), "membership": membership, "probability_kind": probability_kind}


def format_line_report(report: dict | None) -> str:
    """The text of :func:`line_report`: a table of the lines inside the bandpass for the redshift cut, then a table of
    the interlopers, each with notes on its columns."""
    from .report import count, fields, keep, notes, section, span, table, z_cut

    if not report:
        return "no line report (the catalog was not made by make_catalog)"
    lo, hi = (None, None) if report["z_range"] is None else report["z_range"]
    cut = z_cut(lo, hi, _Z_SHORT.get(report["z_column"], report["z_column"]))
    band = f"{report['wave_range'][0]:.0f}–{report['wave_range'][1]:.0f} Å"
    counted = bool(report["lines"]) and "n_identified" in report["lines"][0]
    waves = lambda w: f"{w[0]:.0f}–{w[1]:.0f}"

    out = [section(f"Lines in the bandpass, {band}, for {cut}")]
    if report["lines"]:
        cols = [("line", "<"), ("redshift", "<"), ("λ_obs [Å]", "<")]
        if counted:
            cols += [("detected", ">"), ("identified", ">"), ("brightest", ">"), ("mistaken", ">"), ("interlopers", ">")]
        rows = []
        for r in report["lines"]:
            row = [r["line"] + ("" if r["selected"] else " ○"), span(*r["z"]), waves(r["wave_obs"])]
            if counted:
                row += [count(r[k]) for k in ("n_detected", "n_identified", "n_brightest", "n_mistaken", "n_interlopers")]
            rows.append(row)
        out += table(cols, rows)
        if counted:
            out += notes("galaxies of the catalog, a galaxy with several lines counted under each of them:")
            out += fields([("detected", "the line is detected"),
                           ("identified", "… and gives the right redshift (every detected line of a multi-line galaxy)"),
                           ("brightest", "… and is the brightest detected line (line_assigned)"),
                           ("mistaken", "the only line, taken for another"),
                           ("interlopers", "another line taken for this one")], indent=4)
        if not all(r["selected"] for r in report["lines"]):
            out += notes("○ not in `lines`: inside the bandpass, but its galaxies are not in the catalog")
        out += notes(f"redshift of the lightcone: {span(*report['z_cover'], 2)} (z_rsd)")
    else:
        out += notes("none")

    rows = report["interlopers"]
    likely = [r for r in rows if r["probability"] is None or r["probability"] > 0 or r.get("n", 0) > 0]
    out += ["", section("Interlopers: single lines given a wrong line, and so a wrong redshift")]
    if likely:
        cols = [("true line", "<"), ("taken for", "<"), ("true z", "<"), ("assigned z", "<"), ("λ_obs [Å]", "<"), ("P¹", ">")]
        if counted:
            cols += [("in catalog", ">"), ("share²", ">")]
        body = []
        for r in likely:
            row = [r["line_true"], r["line_assigned"], span(*r["z_true"]), span(*r["z_assigned"]), waves(r["wave_obs"]),
                   "model" if r["probability"] is None else f"{r['probability']:.3f}"]
            if counted:
                row += [count(r["n"]), f"{100 * r['n'] / r['n_line']:.2f} %" if r["n_line"] else "–"]
            body.append(row)
        out += table(cols, body)
        if report.get("probability_kind") == "expected":
            out += notes("expected over the single-line galaxies of the true line, from the line mix of the mock at their "
                         "observed wavelength and flux rank", mark="¹ ")
        elif report.get("probability_kind") == "windowed":
            out += notes("the probability that a single line of the true line is taken for the other where that puts it "
                         "inside the range (elsewhere it keeps its line)", mark="¹ ")
        else:
            out += notes("the probability that a single line of the true line is taken for the other", mark="¹ ")
        if counted:
            own = "identified" if report.get("membership", "detected") == "detected" else "brightest"
            out += notes(f"of the sample of the line it is taken for: its {own} galaxies plus its interlopers", mark="² ")
    if counted:
        total = sum(r["n"] for r in rows)
        out += notes(f"{total:,} interlopers in the catalog"
                     + ("" if report["misidentified_kept"] else " (`outcomes` excludes the misidentified galaxies)"))
    idle = {}
    targets = {r["line_assigned"] for r in likely} | ({r["line"] for r in report["lines"] if r["selected"]}
                                                      if not all(r["selected"] for r in report["lines"]) else set())
    for r in rows:
        if r not in likely and r["line_assigned"] in targets:       # lines nothing is taken for are left out
            idle.setdefault(r["line_assigned"], []).append(keep(f"{r['line_true']} ({span(*r['z_true'], 2)})"))
    if idle:
        out += notes("possible by wavelength, but with P = 0 (true redshifts in brackets):")
        for name, v in idle.items():
            out += notes(f"taken for {name}: " + ", ".join(v), indent=4)
    if not rows:
        out += notes("none possible: no other line falls at these wavelengths")
    return "\n".join(out)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


Catalog.nbytes = property(lambda self: int(sum(v.nbytes for v in self.data.values())))

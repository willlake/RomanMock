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
from .confusion import ConfusionMatrix, build_confusion, default_confusion
from .detection import detect_lines
from .io import memory_budget_gb, read_manifest, suggest_n_workers
from .lines import get_line_set
from .redshift import IdOutcome, PerfectMultiLine, assign_redshifts
from .roman import DEFAULT_LINE_FLUX_LIMIT, DEFAULT_SECONDARY_LINE_FLUX_LIMIT
from .samples import THINNING_STREAM
from .selection import TargetDNDZ, tracer_membership, dndz_from_counts, keep_probability, probability_per_galaxy, sky_area_deg2
from .sensitivity import ConstantFluxLimit, build_detection, build_sensitivity

#: columns of a catalog when ``columns`` is not given
DEFAULT_COLUMNS = ["ra", "dec", "redshift_true", "redshift_rsd", "redshift_assigned", "id_outcome"]
#: columns computed in step 2 (the others are read from the base catalog)
DERIVED_COLUMNS = ["redshift_assigned", "id_outcome", "line_bitmask", "n_det", "line_true", "line_assigned"]
DEFAULT_OUTCOMES = ("MULTI_LINE", "SINGLE_CORRECT", "SINGLE_MISID")     # every galaxy with a redshift
CHUNK_ROWS = 1_000_000  #: rows of a slice processed at a time by a worker (bounds its memory)
RSD_MARGIN = 0.012      #: largest |z_rsd - z_true| / (1 + z) allowed for in the line report
REDSHIFT_COLUMNS = ("redshift_assigned", "redshift_rsd", "redshift_true")

DESCRIPTIONS = {
    "redshift_rsd": "True redshift including the line-of-sight peculiar velocity",
    "redshift_assigned": "Assigned (measured) redshift; NaN when no reliable redshift",
    "id_outcome": "Identification outcome: " + IdOutcome.describe(),
    "line_bitmask": "Detected-line bitmask; bit i = feature i of the line set",
    "n_det": "Number of detected features = popcount(line_bitmask)",
    "line_true": "Index of the brightest detected feature",
    "line_assigned": "Feature index assigned by the redshift identification (-1 = failed)",
}


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
        lims = f"F > {s.get('flux_limit', float('nan')):.2e}" + (
            f" (other lines > {s['secondary_flux_limit']:.2e})" if s.get("secondary_flux_limit") else "")
        cuts = ", ".join(str(x) for x in [
            f"lines {s['lines']}" if s.get("lines") else None,
            f"{s.get('z_column', 'redshift_assigned')} in {tuple(s['z_range'])}" if s.get("z_range") else None,
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
        lines = [repr(self)]
        if "id_outcome" in self.data and len(self):
            for name, n in self.outcomes().items():
                if n:
                    lines.append(f"  {name:<15s} {n:>12,}  ({100 * n / len(self):6.2f}%)")
        for name, t in (self.info.get("downsample") or {}).items():
            lines.append(f"  population {name}: {t['n_before']:,} -> {t['n_after']:,} emitters "
                         f"(p_keep {min(t['p_keep']):.3f}-{max(t['p_keep']):.3f})")
        return "\n".join(lines)

    def subset(self, which: str) -> "Catalog":
        """``"galaxies"``: the galaxies with a correct redshift; ``"interlopers"``: the ones with a wrong
        redshift (single-line, misidentified); ``"all"``: both. Needs the column ``id_outcome``."""
        from .randoms import kind_mask

        return self[kind_mask(self, which)]

    def make_random(self, which="all", n_times: int = 1, z_column: str | None = None, seed: int = 12345, write_to=None,
                    first: int = 0, footprint=None, verbose: bool = True):
        """Random catalogs for this catalog: uniform inside the footprint of the lightcone, redshifts
        drawn from the data.

        Parameters
        ----------
        which : "all", "galaxies", "interlopers", or a list of them
            The data catalog the randoms belong to: the whole (contaminated) catalog, the
            galaxies with a correct redshift, or the interlopers (see :meth:`subset`).
        n_times : int
            Number of random catalogs, each of the size of its data catalog.
        z_column : str
            Redshift column; default: the one of the redshift cut (``redshift_assigned``).
        seed : int
            Random catalog ``k`` of a kind depends on ``(seed, kind, k)`` only.
        write_to : path
            Write the randoms instead of returning them: ``name.hdf5`` gives ``name_00.hdf5``,
            ``name_01.hdf5``, ... (``name_<kind>_00.hdf5`` for a list of kinds), a directory
            gives ``random_00.hdf5``, ... Read one with :meth:`Catalog.read`.
        first : int
            Index of the first random catalog, to add more to existing ones.
        footprint : PatchFootprint
            Default: the footprint of the lightcone the catalog was made from.

        Returns
        -------
        A list of ``n_times`` catalogs with the columns ra, dec and the redshift, or the list of
        files when ``write_to`` is given. For a list of kinds: a dict of such lists.

        A cut on ``ra`` or ``dec`` made with ``cuts=`` is applied to the randoms as well.
        """
        from .randoms import make_random

        if footprint is None:
            if not self.settings.get("base"):
                raise ValueError("the catalog does not know its base catalog: pass footprint=")
            footprint = BaseCatalog(self.settings["base"], self.settings.get("base_tag", "romanmock_base")).footprint
        return make_random(self, footprint, which, n_times, z_column, seed, write_to, first, verbose)

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
        meta = {"settings": self.settings, "info": self.info, "line_names": self.line_names,
                "columns": self.columns, "area_deg2": self.area_deg2, "romanmock_version": __version__,
                "written": _dt.datetime.now().isoformat(timespec="seconds")}
        blob = json.dumps(meta, default=_json_default)
        suffix = path.suffix.lower()
        if suffix in (".hdf5", ".h5"):
            import h5py

            with h5py.File(path, "w") as f:
                f.attrs["romanmock_catalog"] = blob
                f.attrs["n_galaxies"] = len(self)
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
        if area_deg2 is not None:
            self.area_deg2 = area_deg2

    # ---- what is in it -------------------------------------------------------------
    @property
    def n_galaxies(self) -> int:
        return sum(v["n_galaxies"] for v in self.slices.values())

    @property
    def flux_limit(self) -> float:
        """Primary flux limit of the base catalog: the deepest limit step 2 can use."""
        return float(self.sensitivity.flux_limit(np.array([np.mean(self.sensitivity.wave_range)]))[0])

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
        confusion=None,
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
        confusion : dict, ConfusionMatrix or None
            ``{true line: {assigned line: probability, "fail": probability}}`` for the
            single-line galaxies; lines not listed are identified correctly. ``None`` = the
            package default. A full config (``{"type": ...}``) or a model object also works.
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
        spec = self._spec(flux_limit, secondary_flux_limit, confusion, z_range, z_column, lines, membership, outcomes, n_det_min,
                          cuts, downsample, columns, wave_range, detection, sigma_z, seed)
        populations = spec["populations"]
        if area_deg2 is not None:
            self.area_deg2 = area_deg2
        area = self.area_deg2
        if n_workers is None:
            n_workers = suggest_n_workers(max_workers=32, per_worker_gb=2.5)     # memory-limited: ~12 on a login node
        tasks = [(spec, str(self.dir / v["output"]), step) for step, v in self.slices.items()]
        if verbose:
            print(f"[romanmock] {len(tasks)} slices of {self.dir.name}, {n_workers} worker(s)", flush=True)
        results = _run(tasks, n_workers)

        data = {c: np.concatenate([r["columns"][c] for r in results]) for c in spec["columns"]}
        info = {"n_base": int(sum(r["n_base"] for r in results)),
                "n_detected": int(sum(r["n_detected"] for r in results)),
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
                                          counts, spec["outcome_codes"], spec["membership"])
        settings = {k: spec[k] for k in ("flux_limit", "secondary_flux_limit", "wave_range", "z_range", "z_column", "lines", "membership",
                                         "outcomes", "n_det_min", "cuts", "sigma_z", "seed", "detection")}
        settings.update(confusion=spec["confusion_config"], downsample=spec["downsample_label"], base=str(self.dir), base_tag=self.tag)
        info["area_deg2"], info["area_source"] = area, self.area_source
        cat = Catalog(data, settings, info, self.line_set.names, area)
        if verbose:
            how = {"user": "given", "footprint": "estimated from the sky patches of the mock",
                   "healpix": "estimated from the galaxies"}[self.area_source]
            print(f"[romanmock] sky area {area:.2f} deg2 ({how})", flush=True)
            print(f"[romanmock] {info['n_detected']:,} of {info['n_base']:,} base galaxies detected; "
                  f"{info['n_galaxies']:,} in the catalog", flush=True)
            print(cat.line_report(), flush=True)
        if cat.nbytes > 0.5e9 * memory_budget_gb():
            warnings.warn(f"the catalog takes {cat.nbytes / 1e9:.1f} GB of memory", stacklevel=2)
        if write_to is not None:
            cat.write(write_to)
        return cat

    # ---- settings -> a picklable description for the workers -------------------------
    def _spec(self, flux_limit, secondary, confusion, z_range, z_column, lines, membership, outcomes, n_det_min, cuts, downsample,
              columns, wave_range, detection, sigma_z, seed) -> dict:
        names = self.line_set.names
        flux_limit = self.flux_limit if flux_limit is None else float(flux_limit)
        if flux_limit < self.flux_limit * (1 - 1e-9):
            raise ValueError(f"flux_limit {flux_limit:.3g} is deeper than the base catalog ({self.flux_limit:.3g}); "
                             "rebuild the base catalog deeper")
        wave_range = self.wave_range if wave_range is None else (float(wave_range[0]), float(wave_range[1]))
        if wave_range[0] < self.wave_range[0] - 1e-6 or wave_range[1] > self.wave_range[1] + 1e-6:
            raise ValueError(f"wave_range {wave_range} extends beyond the bandpass of the base catalog {self.wave_range}")
        if isinstance(secondary, str):
            if secondary != "default":
                raise ValueError("secondary_flux_limit must be a number, None or 'default'")
            secondary = min(DEFAULT_SECONDARY_LINE_FLUX_LIMIT, flux_limit)
        sens = ConstantFluxLimit(flux_limit, wave_range, secondary_flux_limit=secondary)    # validates the pair

        if confusion is None:
            model = default_confusion(self.line_set)
        elif isinstance(confusion, Mapping) and "type" not in confusion:
            unknown = sorted(set(confusion) - set(names))
            if unknown:
                raise KeyError(f"confusion rows for unknown lines {unknown}; the lines are {names}")
            model = ConfusionMatrix.from_dict(self.line_set, {k: dict(v) for k, v in confusion.items()}, default_p_fail=0.0)
        else:
            model = build_confusion(confusion, self.line_set)
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
            "sensitivity": sens.to_config(), "flux_limit": flux_limit,
            "secondary_flux_limit": sens.secondary_flux_limit_value if sens.two_tier else None,
            "wave_range": list(wave_range), "detection": dict(detection or {"type": "threshold"}),
            "confusion": model, "confusion_config": confusion_config, "sigma_z": float(sigma_z), "seed": int(seed),
            "z_range": None if z_range is None else list(z_range), "z_column": z_column,
            "lines": None if lines is None else list(lines),
            "line_bits": None if lines is None else [names.index(x) for x in lines], "membership": membership,
            "outcomes": None if outcomes is None else [IdOutcome(c).name for c in codes], "outcome_codes": codes,
            "n_det_min": n_det_min, "cuts": {k: list(v) for k, v in cuts.items()},
            "columns": list(columns), "populations": populations, "downsample_label": label,
        }

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


def make_catalog(base_dir, **settings) -> Catalog:
    """``BaseCatalog(base_dir).make_catalog(**settings)``."""
    return BaseCatalog(base_dir).make_catalog(**settings)


def make_random(catalog, **settings):
    """``catalog.make_random(**settings)``; ``catalog`` is a :class:`Catalog` or a file written by one."""
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
    derived = set(DERIVED_COLUMNS)
    wanted = [c for c in dict.fromkeys([*spec["columns"], *spec["cuts"], spec["z_column"]]) if c not in derived]
    read = list(dict.fromkeys([*flux_columns, *comp_cols.values(), "redshift_rsd", *wanted]))
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
            z_rsd = np.asarray(data["redshift_rsd"], dtype=np.float64)
            flux = np.stack([np.asarray(data[c], dtype=np.float64) for c in flux_columns], axis=1)
            comp = {lum: np.asarray(data[name], dtype=np.float64) for lum, name in comp_cols.items()} or None
            det = detect_lines(flux, z_rsd, line_set, model, rng, component_flux=comp)
            za = assign_redshifts(det, z_rsd, line_set, spec["confusion"], rng, multi_line)
            cols = dict(data)
            cols.update(redshift_assigned=za.z_assigned, id_outcome=za.outcome, line_bitmask=det.bitmask, n_det=det.n_det,
                        line_true=za.line_true, line_assigned=za.line_assigned)
            detected = det.bitmask > 0
            part = {"n": stop - start, "n_detected": int(detected.sum())}
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


def _run(tasks, n_workers: int) -> list[dict]:
    if n_workers <= 1 or len(tasks) <= 1:
        return [_catalog_slice(*t) for t in tasks]
    import concurrent.futures as cf
    import multiprocessing as mp

    with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(_catalog_slice, *t) for t in tasks]
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
    wavelength bins of a binned model); ``None`` for a model that is not a matrix."""
    if isinstance(model, ConfusionMatrix):
        return model.matrix[:, :-1].copy()
    matrices = getattr(model, "matrices", None)
    if matrices and all(isinstance(m, ConfusionMatrix) for m in matrices):
        return np.max([m.matrix[:, :-1] for m in matrices], axis=0)
    return None


def line_report(names, rest_waves, wave_range, z_range, z_column, z_cover, assigned_bits=None, probabilities=None,
                counts=None, outcome_codes=None, membership: str = "detected") -> dict:
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
            "misidentified_kept": bool(kept), "membership": membership}


def format_line_report(report: dict | None) -> str:
    """The text of :func:`line_report`."""
    if not report:
        return "no line report (the catalog was not made by make_catalog)"
    lo, hi = (None, None) if report["z_range"] is None else report["z_range"]
    cut = f"{report['z_column']} " + ("unrestricted" if lo is None and hi is None else
                                      f">= {lo:g}" if hi is None else f"<= {hi:g}" if lo is None else f"in {lo:g} - {hi:g}")
    band = f"{report['wave_range'][0]:.0f} - {report['wave_range'][1]:.0f} A"
    counted = bool(report["lines"]) and "n_identified" in report["lines"][0]

    def span(v, fmt):
        return f"{v[0]:{fmt}} - {v[1]:{fmt}}"

    out = [f"[romanmock] lines inside {band} for {cut} (lightcone: z_rsd {span(report['z_cover'], '.2f')})"]
    out.append(f"  {'line':<11s}{'rest [A]':>9s}  {'redshift':^13s}  {'observed [A]':^15s}"
               + (f"{'detected':>13s}{'identified':>13s}{'brightest':>13s}{'mistaken':>12s}{'interlopers':>13s}" if counted else ""))
    for r in report["lines"]:
        out.append(f"  {r['line']:<11s}{r['rest_wave']:>9.1f}  {span(r['z'], '.3f'):^13s}  {span(r['wave_obs'], '.0f'):^15s}"
                   + (f"{r['n_detected']:>13,}{r['n_identified']:>13,}{r['n_brightest']:>13,}{r['n_mistaken']:>12,}"
                      f"{r['n_interlopers']:>13,}" if counted else "")
                   + ("" if r["selected"] else "   not in `lines`"))
    if not report["lines"]:
        out.append("  none")
    elif counted:
        out += ["  galaxies of the catalog, a galaxy with several lines counted in each of them:",
                "    detected     the line is detected",
                "    identified   ... and identified as this line: the redshift of the galaxy is right (every detected line",
                "                 of a multi-line galaxy, the line of a correctly identified single-line galaxy)",
                "    brightest    ... and it is the brightest detected line of the galaxy (`line_assigned`)",
                "    mistaken     the line is the only one detected and was taken for another line",
                "    interlopers  galaxies in which another line was taken for this line"]

    rows = report["interlopers"]
    likely = [r for r in rows if r["probability"] is None or r["probability"] > 0 or r.get("n", 0) > 0]
    out.append("[romanmock] line interlopers: single-line galaxies given a wrong line and so a wrong redshift")
    if likely:
        out.append(f"  {'true line':<11s}{'taken for':<11s}{'true z_rsd':^13s}  {'assigned z':^13s}  {'observed [A]':^15s}{'P':>7s}"
                   + (f"{'in catalog':>13s}{'contamination':>15s}" if counted else ""))
    for r in likely:
        p = "model" if r["probability"] is None else f"{r['probability']:.3f}"
        text = (f"  {r['line_true']:<11s}{r['line_assigned']:<11s}{span(r['z_true'], '.3f'):^13s}  "
                f"{span(r['z_assigned'], '.3f'):^13s}  {span(r['wave_obs'], '.0f'):^15s}{p:>7s}")
        if counted:
            share = f"{100 * r['n'] / r['n_line']:.2f}%" if r["n_line"] else "-"
            text += f"{r['n']:>13,}{share:>15s}"
        out.append(text)
    if counted:
        total = sum(r["n"] for r in rows)
        own = "identified" if report.get("membership", "detected") == "detected" else "brightest"
        out.append(f"  (contamination: this interloper as a share of the sample of the line of `taken for`, which is its "
                   f"`{own}` galaxies plus its interlopers)")
        out.append(f"  actual interlopers in the catalog: {total:,}"
                   + ("" if report["misidentified_kept"] else " (`outcomes` excludes the misidentified galaxies)"))
    idle = {}
    targets = {r["line_assigned"] for r in likely} | ({r["line"] for r in report["lines"] if r["selected"]}
                                                      if not all(r["selected"] for r in report["lines"]) else set())
    for r in rows:
        if r not in likely and r["line_assigned"] in targets:       # lines nothing is taken for are left out
            idle.setdefault(r["line_assigned"], []).append(f"{r['line_true']} (z {span(r['z_true'], '.2f')})")
    if idle:
        out.append("  possible by wavelength, but with probability 0 in the confusion matrix:")
        out += [f"    taken for {name}: " + ", ".join(v) for name, v in idle.items()]
    if not rows:
        out.append("  none possible: no other line falls at these wavelengths")
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

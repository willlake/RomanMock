"""Read and analyse the Galacticus lightcone mocks of the Roman GRS-PIT.

A Galacticus mock is a directory such as
``/global/cfs/cdirs/m4943/Galacticus/mockCatalogs/GP-MCMC-Calibration1_allUNIT_16sqDegMock_XMM-LSS``
whose ``galaxyCatalogs/`` holds a few hundred HDF5 files ``romanUNIT-d<a>_<b>_16sqDeg.hdf5``: random
subsamples of the merger trees, every one covering the whole 4 x 4 deg lightcone, so the mock is the
union of all of them. The galaxy properties are the datasets of ``/Lightcone/Output1/nodeData``
(dust-free) and ``/Lightcone/Output1/dustAttenuatedNodeData`` (same names, dust applied); the emission
lines come as one dataset per component, ``luminosityEmissionLine{Disk,Spheroid,AGN}:<line>`` in
erg/s, and the luminosity of a line is the sum of the three.

:class:`GalacticusMock` turns that into the dict-of-arrays interface of the rest of the package:

* ``mock.read(columns, z_range=...)``: the combined arrays of all files (or ``mock.iter_files`` to
  stream file by file). Columns are the package's names (``redshift_true``, ``redshift_rsd``, ``ra``,
  ``dec``, ``gal_id``, ``is_central``), Galacticus line names (``balmerAlpha6565``: disk + spheroid
  + AGN of the chosen dust treatment), explicit component datasets (``luminosityEmissionLineDisk:...``)
  or any other dataset of ``nodeData``.
* ``mock.read_fluxes(line_set)``: observed fluxes of the features of a :class:`~romanmock.lines.LineSet`
  (default ``galacticus_transitions``), erg/s/cm^2 with ``D_L`` at the cosmological redshift.
* ``mock.flux_distribution(...)``: the :class:`~romanmock.fluxdist.FluxDistribution` of the whole mock
  (quantiles and fractions above a threshold per redshift bin), one file per worker.
* ``mock.single_lines(...)``: the single line (observed wavelength, flux, feature) of every single-line
  galaxy, with any other columns of those galaxies.
* ``mock.line_fractions(...)``: the :class:`~romanmock.confusion.LineFractionTable` of the single-line
  galaxies (line mix per bin of observed wavelength and flux rank), as ``BaseCatalog.line_fractions``.

The cosmology is read from the file (``Parameters/cosmologyParameters``: flat LCDM, H0 = 67.74,
Omega_m = 0.3089, no radiation, as in the mock's Readme; it reproduces the stored comoving positions
to 2e-5). The sky area is ``angularSize**2`` of ``Parameters/geometryLightcone`` (16 deg^2).
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import h5py
import numpy as np

from .confusion import LineFractionTable
from .detection import detect_lines
from .fluxdist import DEFAULT_LOGF_EDGES, DEFAULT_Z_EDGES, FluxDistribution, flux_histograms
from .fluxes import component_fluxes, feature_fluxes, flux_column_names
from .lines import GALACTICUS_LINE_WAVELENGTHS, LineSet, get_line_set
from .redshift import single_lines
from .roman import DEFAULT_LINE_FLUX_LIMIT, GRISM_WAVE_RANGE_ANGSTROM
from .sensitivity import (ConstantFluxLimit, RomanGrismSensitivity, build_detection, build_sensitivity, needs_size,
                          nominal_flux_limit, nominal_secondary_limit)

CATALOG_SUBDIR = "galaxyCatalogs"
FILE_GLOB = "*.hdf5"
OUTPUT_GROUP = "Lightcone/Output1"
NODE_GROUP = "nodeData"
DUST_GROUP = "dustAttenuatedNodeData"
COMPONENTS = ("Disk", "Spheroid", "AGN")
LINE_PREFIX = "luminosityEmissionLine"
DEFAULT_LINE_SET = "galacticus_transitions"

#: Package column name -> Galacticus dataset (in ``nodeData``).
COLUMN_ALIASES: dict[str, str] = {
    "redshift_true": "lightconeRedshiftCosmological",
    "redshift_rsd": "lightconeRedshiftObserved",
    "ra": "rightAscension",
    "dec": "declination",
    "gal_id": "nodeIndex",
    "is_central": "nodeIsIsolated",
}

#: The emission lines of the Galacticus mocks (the ``<line>`` part of the dataset names).
GALACTICUS_LINES: tuple[str, ...] = tuple(GALACTICUS_LINE_WAVELENGTHS)

_CHUNK_RE = re.compile(r"-d(\d+)_(\d+)")

FRACTION_STREAM = 40000  #: random stream of the line-fraction scan, (seed, FRACTION_STREAM, chunk), as BaseCatalog.line_fractions


def air_to_vacuum(wave_air):
    """Air -> vacuum wavelength [A] (IAU standard, Morton 1991), the conversion behind
    :data:`romanmock.lines.GALACTICUS_LINE_WAVELENGTHS`."""
    w = np.asarray(wave_air, dtype=np.float64)
    s2 = (1e4 / w) ** 2
    n = 1.0 + 6.4328e-5 + 2.94981e-2 / (146.0 - s2) + 2.5540e-4 / (41.0 - s2)
    return w * n


def chunk_index(path) -> int:
    """First tree index of a file ``romanUNIT-d<a>_<b>_...hdf5`` (its position in the mock)."""
    m = _CHUNK_RE.search(Path(path).name)
    if not m:
        raise ValueError(f"cannot infer the chunk index from {path}")
    return int(m.group(1))


def list_files(path) -> list[Path]:
    """The catalog files of a Galacticus mock directory (or of its ``galaxyCatalogs`` directory)."""
    path = Path(path)
    cat_dir = path / CATALOG_SUBDIR if (path / CATALOG_SUBDIR).is_dir() else path
    files = sorted(cat_dir.glob(FILE_GLOB), key=chunk_index)
    if not files:
        raise FileNotFoundError(f"no {FILE_GLOB} files in {cat_dir}")
    return files


@dataclass(frozen=True)
class GalacticusFileInfo:
    path: Path
    chunk: int
    n_gal: int


def file_info(path) -> GalacticusFileInfo:
    """Chunk index and row count of one file (metadata only, no column read)."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        n = f[f"{OUTPUT_GROUP}/{NODE_GROUP}/{COLUMN_ALIASES['redshift_true']}"].shape[0]
    return GalacticusFileInfo(path, chunk_index(path), int(n))


def _header(path) -> dict:
    """Cosmology, geometry and dust metadata of a file."""
    with h5py.File(path, "r") as f:
        cosmo = {k: (v.item() if hasattr(v, "item") else v) for k, v in f["Parameters/cosmologyParameters"].attrs.items()}
        geom = {k: (v.item() if np.ndim(v) == 0 and hasattr(v, "item") else v) for k, v in f["Parameters/geometryLightcone"].attrs.items()}
        dust = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f[f"{OUTPUT_GROUP}/{DUST_GROUP}"].attrs.items()}
        air = {}
        node = f[f"{OUTPUT_GROUP}/{NODE_GROUP}"]
        for name in node:
            if name.startswith(f"{LINE_PREFIX}{COMPONENTS[0]}:"):
                air[name.split(":", 1)[1]] = float(node[name].attrs["wavelength"])
    return {"cosmology": cosmo, "geometry": geom, "dust": dust, "air_wavelengths": air}


def cosmology_from_header(header: Mapping):
    """``astropy`` flat LCDM cosmology of the mock (no radiation, as in its Readme)."""
    from astropy.cosmology import FlatLambdaCDM

    c = header["cosmology"]
    return FlatLambdaCDM(H0=float(c["HubbleConstant"]), Om0=float(c["OmegaMatter"]), Ob0=float(c.get("OmegaBaryon", 0.0)) or None)


def resolve_column(column: str, dust: bool = True, lines: Iterable[str] = GALACTICUS_LINES) -> list[str]:
    """Datasets (relative to ``Lightcone/Output1``) whose sum is ``column``.

    Package aliases map to ``nodeData``; a line name maps to its three components in the dust-attenuated
    (``dust=True``) or dust-free group; an explicit component or magnitude dataset to that group; anything
    else to ``nodeData/<column>`` (or is taken as a path when it contains ``/``).
    """
    group = DUST_GROUP if dust else NODE_GROUP
    if column in COLUMN_ALIASES:
        return [f"{NODE_GROUP}/{COLUMN_ALIASES[column]}"]
    if column in lines or column in GALACTICUS_LINES:
        return [f"{group}/{LINE_PREFIX}{comp}:{column}" for comp in COMPONENTS]
    if column.startswith(LINE_PREFIX) or column.startswith("apparentMagnitude"):
        return [f"{group}/{column}"]
    if "/" in column:
        return [column]
    return [f"{NODE_GROUP}/{column}"]


def read_file(path, columns: Sequence[str], z_range: tuple | None = None, z_column: str = "redshift_true",
              dust: bool = True) -> dict[str, np.ndarray]:
    """The requested columns of one file as a dict of arrays; ``z_range = (lo, hi)`` keeps ``lo <= z < hi``."""
    columns = list(dict.fromkeys(columns))
    with h5py.File(path, "r") as f:
        out_group = f[OUTPUT_GROUP]
        sel = None
        if z_range is not None:
            z = out_group[resolve_column(z_column, dust)[0]][...]
            lo, hi = z_range
            sel = np.ones(z.size, dtype=bool)
            if lo is not None:
                sel &= z >= lo
            if hi is not None:
                sel &= z < hi
            if sel.all():
                sel = None
        data = {}
        for col in columns:
            parts = resolve_column(col, dust)
            arr = None
            for name in parts:
                if name not in out_group:
                    raise KeyError(f"{Path(path).name}: no dataset {OUTPUT_GROUP}/{name} (column {col!r})")
                a = out_group[name][...]
                if len(parts) > 1:
                    a = a.astype(np.float64, copy=False)
                arr = a if arr is None else arr + a
            data[col] = arr if sel is None else arr[sel]
    return data


class GalacticusMock:
    """A Galacticus lightcone mock: the union of the files of ``galaxyCatalogs/``.

    Parameters
    ----------
    path : str or Path
        The mock directory (containing ``galaxyCatalogs/``) or the ``galaxyCatalogs`` directory.
    dust : bool
        Read the dust-attenuated line luminosities (default) or the dust-free ones.
    area_deg2 : float, optional
        Sky area; default ``angularSize**2`` of the lightcone geometry (16 deg^2 for the 4 x 4 deg mocks).
    files : sequence of paths, optional
        Restrict to these files (absolute, or relative to the catalog directory).
    """

    def __init__(self, path, dust: bool = True, area_deg2: float | None = None, files: Sequence[str | Path] | None = None):
        self.path = Path(path)
        all_files = list_files(self.path)
        self.catalog_dir = all_files[0].parent
        if files:
            self.files = [Path(f) if Path(f).is_absolute() else self.catalog_dir / f for f in files]
        else:
            self.files = all_files
        self.dust = bool(dust)
        self._header = _header(self.files[0])
        self.cosmology = cosmology_from_header(self._header)
        size = float(self._header["geometry"].get("angularSize", np.nan))
        self.area_deg2 = float(area_deg2) if area_deg2 is not None else size * size
        self.area_source = "user" if area_deg2 is not None else "geometry"
        self._infos: list[GalacticusFileInfo] | None = None

    # ---- description ---------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.path.name if self.path.name != CATALOG_SUBDIR else self.path.parent.name

    @property
    def lines(self) -> tuple[str, ...]:
        """Emission lines present in the files."""
        return tuple(self._header["air_wavelengths"])

    @property
    def dust_model(self) -> dict:
        return dict(self._header["dust"])

    def rest_wavelengths(self, vacuum: bool = True) -> dict[str, float]:
        """Rest wavelength [A] of every line, from the file attributes (air) or converted to vacuum."""
        air = self._header["air_wavelengths"]
        return {k: float(air_to_vacuum(w)) if vacuum else w for k, w in air.items()}

    @property
    def infos(self) -> list[GalacticusFileInfo]:
        """Chunk index and row count of every file (metadata only; 40 s for 400 files on CFS, cached)."""
        if self._infos is None:
            self._infos = [file_info(p) for p in self.files]
        return self._infos

    @property
    def n_gal(self) -> int:
        return int(sum(i.n_gal for i in self.infos))

    def __repr__(self) -> str:
        return (f"GalacticusMock({self.name!r}, {len(self.files)} files, {'dust-attenuated' if self.dust else 'dust-free'} lines, "
                f"{self.area_deg2:g} deg^2, H0={self.cosmology.H0.value:g}, Om0={self.cosmology.Om0:g})")

    def describe(self) -> str:
        lines = [repr(self), f"  files: {self.catalog_dir}", f"  galaxies: {self.n_gal:,}",
                 f"  lines: {', '.join(self.lines)}", f"  dust: {self.dust_model.get('emission_line_dust', self.dust_model)}"]
        return "\n".join(lines)

    # ---- reading -------------------------------------------------------------------------
    def resolve(self, column: str, dust: bool | None = None) -> list[str]:
        """Datasets (relative to ``Lightcone/Output1``) whose sum is the column."""
        return resolve_column(column, self.dust if dust is None else dust, self.lines)

    def read_file(self, path, columns: Sequence[str], z_range: tuple | None = None, z_column: str = "redshift_true",
                  dust: bool | None = None) -> dict[str, np.ndarray]:
        """The requested columns of one file as a dict of arrays, optionally cut in redshift."""
        return read_file(path, columns, z_range, z_column, self.dust if dust is None else dust)

    def iter_files(self, columns: Sequence[str], z_range: tuple | None = None, z_column: str = "redshift_true",
                   files: Sequence[str | Path] | None = None, dust: bool | None = None) -> Iterator[tuple[Path, dict[str, np.ndarray]]]:
        """Yield ``(path, data)`` file by file (a few MB each): the way to scan the mock without holding it."""
        for p in self._paths(files):
            yield p, self.read_file(p, columns, z_range, z_column, dust)

    def read(self, columns: Sequence[str], z_range: tuple | None = None, z_column: str = "redshift_true",
             files: Sequence[str | Path] | None = None, dust: bool | None = None, n_workers: int = 1,
             verbose: bool = False) -> dict[str, np.ndarray]:
        """The combined arrays of all files (in file order).

        ``z_range = (lo, hi)`` keeps ``lo <= z < hi`` in ``z_column`` (``None`` for an open end). The dust
        treatment is that of the mock unless ``dust`` is given. ``n_workers > 1`` reads several files in
        parallel processes (I/O bound: 4 to 8 workers on a login node).
        """
        columns = list(dict.fromkeys(columns))
        paths = self._paths(files)
        parts = self._map(_read_worker, paths, (self._spec(dust), columns, z_range, z_column), n_workers, verbose, "read")
        return {c: np.concatenate([p[c] for p in parts]) if parts else np.empty(0) for c in columns}

    def read_fluxes(self, line_set=DEFAULT_LINE_SET, columns: Sequence[str] = ("redshift_true", "redshift_rsd"),
                    z_range: tuple | None = None, z_column: str = "redshift_true", files: Sequence[str | Path] | None = None,
                    dust: bool | None = None, n_workers: int = 1, verbose: bool = False) -> dict[str, np.ndarray]:
        """Observed fluxes ``<feature>_flux`` [erg/s/cm^2] of the features of ``line_set`` plus ``columns``.

        ``F = L / (4 pi D_L^2)`` with ``D_L`` at the cosmological redshift and ``L`` the sum of the components
        (disk + spheroid + AGN) of every component line of the feature.
        """
        line_set = get_line_set(line_set)
        paths = self._paths(files)
        parts = self._map(_flux_worker, paths, (self._spec(dust), line_set.to_config(), list(columns), z_range, z_column),
                          n_workers, verbose, "fluxes")
        names = list(columns) + flux_column_names(line_set)
        return {c: np.concatenate([p[c] for p in parts]) if parts else np.empty(0) for c in names}

    def flux_distribution(self, z_edges=DEFAULT_Z_EDGES, logf_edges=DEFAULT_LOGF_EDGES, line_set=DEFAULT_LINE_SET,
                          z_column: str = "redshift_true", files: Sequence[str | Path] | None = None, dust: bool | None = None,
                          n_workers: int = 1, verbose: bool = True) -> FluxDistribution:
        """Histograms of ``log10`` flux of every feature per redshift bin, for every galaxy, no cut.

        Same object as ``romanmock flux-distribution`` for the diffsky mock (:class:`FluxDistribution`:
        ``quantile``, ``fraction_above``, ``number_above``, ``save``/``load``); ``steps`` holds the chunk
        indices of the files.
        """
        line_set = get_line_set(line_set)
        z_edges, logf_edges = np.asarray(z_edges, dtype=np.float64), np.asarray(logf_edges, dtype=np.float64)
        result = FluxDistribution.empty(z_edges, logf_edges, line_set, z_column)
        paths = self._paths(files)
        args = (self._spec(dust), line_set.to_config(), z_edges.tolist(), logf_edges.tolist(), z_column)
        for chunk, n_gal, parts in self._map(_hist_worker, paths, args, n_workers, verbose, "flux-distribution"):
            result.add(*parts, chunk)
        result.steps.sort()
        return result

    def single_lines(self, flux_limit: float = DEFAULT_LINE_FLUX_LIMIT, secondary_flux_limit: float | None | str = "default",
                     wave_range: tuple[float, float] = GRISM_WAVE_RANGE_ANGSTROM, snr: float | None = None,
                     secondary_snr: float | None | str = "default", size: float | None = None, exposure_time: float | None = None,
                     background: float | None = None, sensitivity=None, detection: Mapping | None = None,
                     line_set=DEFAULT_LINE_SET, columns: Sequence[str] = (), seed: int = 12345,
                     cuts: Mapping[str, tuple] | None = None, files: Sequence[str | Path] | None = None, dust: bool | None = None,
                     n_workers: int = 1, verbose: bool = True) -> dict[str, np.ndarray]:
        """The single line of every single-line galaxy of the mock, the input of :meth:`line_fractions`.

        The scan and the detection are those of :meth:`line_fractions` (same arguments). Returns a dict of
        arrays, one entry per single-line galaxy in file order: ``wave`` (observed wavelength [A]), ``flux``
        (measured flux [erg/s/cm^2]), ``line`` (index of the feature in ``line_set``) and every column of
        ``columns`` (any column of :meth:`read`, e.g. ``redshift_rsd`` or a gas metal mass) of those galaxies.
        """
        sens = self._sensitivity(flux_limit, secondary_flux_limit, wave_range, snr, secondary_snr, size, exposure_time,
                                 background, sensitivity)
        out, _ = self._scan_single_lines(sens, detection, line_set, columns, seed, cuts, files, dust, n_workers, verbose)
        if verbose:
            print(f"[galacticus] {out['line'].size:,} single-line galaxies", flush=True)
        return out

    def line_fractions(self, flux_limit: float = DEFAULT_LINE_FLUX_LIMIT, secondary_flux_limit: float | None | str = "default",
                       wave_range: tuple[float, float] = GRISM_WAVE_RANGE_ANGSTROM, snr: float | None = None,
                       secondary_snr: float | None | str = "default", size: float | None = None, exposure_time: float | None = None,
                       background: float | None = None, sensitivity=None, detection: Mapping | None = None,
                       line_set=DEFAULT_LINE_SET, wave_bin: float = 250.0, wave_edges=None, galaxies_per_bin: int = 500,
                       n_flux_bins: int | None = None, flux_quantiles=None, min_count: int = 100, mode: str = "draw",
                       p_fail: float = 0.0, seed: int = 12345, cuts: Mapping[str, tuple] | None = None,
                       files: Sequence[str | Path] | None = None, dust: bool | None = None, n_workers: int = 1, write_to=None,
                       verbose: bool = True) -> LineFractionTable:
        """The line mix of the single-line galaxies of the mock per bin of observed wavelength and flux
        rank: the :class:`~romanmock.confusion.LineFractionTable` of
        :meth:`romanmock.catalog.BaseCatalog.line_fractions`, built from the Galacticus files.

        Every file is scanned once: the fluxes of the features of ``line_set`` (default
        ``galacticus_transitions``, all 16 features, so [N II], [S II], Hγ, ... count as other lines),
        the detection at the observed wavelength from ``redshift_rsd`` (the two-tier flux limits
        ``flux_limit`` / ``secondary_flux_limit`` inside ``wave_range``, the S/N route ``snr``,
        ``secondary_snr``, ``size`` in arcsec, ``exposure_time``, ``background``, or a ``sensitivity``
        model), and the single line of every single-line galaxy is kept. The binning (``wave_bin`` or
        ``wave_edges``; ``galaxies_per_bin``, ``n_flux_bins`` or ``flux_quantiles``), ``min_count``,
        ``mode`` and ``p_fail`` are those of :meth:`LineFractionTable.from_single_lines`. The mock has no
        angular sizes in the package's sense, so the S/N route takes a fixed ``size``. ``cuts =
        {column: (lo, hi)}`` keeps only the galaxies with ``lo <= column <= hi`` (``None`` for an open
        end; any column of :meth:`read`), as ``make_catalog(cuts=...)``. ``n_workers`` processes read the
        files (4 to 8 on a login node); ``write_to`` saves the table (``.npz``).
        """
        sens = self._sensitivity(flux_limit, secondary_flux_limit, wave_range, snr, secondary_snr, size, exposure_time,
                                 background, sensitivity)
        wave_range = tuple(sens.wave_range)
        if wave_edges is None:
            n_bins = max(1, int(round((wave_range[1] - wave_range[0]) / float(wave_bin))))
            wave_edges = np.linspace(wave_range[0], wave_range[1], n_bins + 1)
        wave_edges = np.asarray(wave_edges, dtype=np.float64)
        line_set = get_line_set(line_set)
        single, info = self._scan_single_lines(sens, detection, line_set, (), seed, cuts, files, dust, n_workers, verbose)
        wave, flux, line = single["wave"], single["flux"], single["line"]
        settings = {"mock": str(self.path), "dust": self.dust if dust is None else bool(dust), "line_set": line_set.name,
                    "sensitivity": sens.to_config(), "flux_limit": nominal_flux_limit(sens),
                    "secondary_flux_limit": nominal_secondary_limit(sens), "wave_range": list(wave_range),
                    "detection": info["detection"], "wave_bin": float(np.median(np.diff(wave_edges))), "n_base": info["n"],
                    "cuts": {k: list(v) for k, v in info["cuts"].items()}, "n_cut": info["n_cut"],
                    "built": _dt.datetime.now().isoformat(timespec="seconds")}
        table = LineFractionTable.from_single_lines(line_set, wave, flux, line, wave_edges, galaxies_per_bin, n_flux_bins,
                                                    flux_quantiles, min_count, mode, p_fail, settings)
        if verbose:
            print(table.summary(), flush=True)
        if write_to is not None:
            path = table.save(write_to)
            if verbose:
                print(f"[galacticus] line fractions written to {path}", flush=True)
        return table

    # ---- helpers -------------------------------------------------------------------------
    @staticmethod
    def _sensitivity(flux_limit, secondary_flux_limit, wave_range, snr, secondary_snr, size, exposure_time, background,
                     sensitivity):
        """The sensitivity model of :meth:`line_fractions` / :meth:`single_lines` (fixed size only)."""
        snr_given = {k: v for k, v in dict(snr=snr, size=size, exposure_time=exposure_time, background=background).items()
                     if v is not None}
        if not (isinstance(secondary_snr, str) and secondary_snr == "default"):
            snr_given["secondary_snr"] = secondary_snr
        if sensitivity is not None:
            sens = build_sensitivity(sensitivity)
        elif snr_given:
            sens = RomanGrismSensitivity(wave_range=wave_range, **snr_given)
        else:
            sens = ConstantFluxLimit(flux_limit, wave_range, secondary_flux_limit=secondary_flux_limit)
        if needs_size(sens):
            raise ValueError("the Galacticus reader gives no angular size per galaxy; use a fixed size=<arcsec>")
        return sens

    def _scan_single_lines(self, sens, detection, line_set, columns, seed, cuts, files, dust, n_workers, verbose):
        """Scan the files for single-line galaxies: (dict of arrays, scan info)."""
        cuts = {str(k): (None if v[0] is None else float(v[0]), None if v[1] is None else float(v[1])) for k, v in (cuts or {}).items()}
        line_set = get_line_set(line_set)
        columns = list(dict.fromkeys(columns))
        paths = self._paths(files)
        detection = dict(detection or {"type": "threshold"})
        if verbose:
            print(f"[galacticus] scanning {len(paths)} files of {self.name} for single-line galaxies "
                  f"({sens.describe()}), {n_workers} worker(s)", flush=True)
        args = (self._spec(dust), line_set.to_config(), sens.to_config(), detection, int(seed), cuts, columns)
        results = self._map(_single_lines_worker, paths, args, n_workers, False, "single-lines")
        out = {k: np.concatenate([r[k] for r in results]) for k in ("wave", "flux", "line", *columns)}
        info = {"n": int(sum(r["n"] for r in results)), "n_cut": int(sum(r["n_cut"] for r in results)), "cuts": cuts,
                "detection": detection}
        return out, info

    def _paths(self, files) -> list[Path]:
        if not files:
            return list(self.files)
        return [Path(f) if Path(f).is_absolute() else self.catalog_dir / f for f in files]

    def _spec(self, dust) -> dict:
        """What a worker needs (picklable)."""
        return {"dust": self.dust if dust is None else bool(dust),
                "cosmology": {k: float(v) for k, v in self._header["cosmology"].items()}}

    @staticmethod
    def _map(fn, paths, args, n_workers, verbose, what):
        if n_workers <= 1:
            out = []
            for i, p in enumerate(paths):
                out.append(fn(str(p), *args))
                if verbose:
                    print(f"[galacticus {what}] {i + 1}/{len(paths)} {Path(p).name}", flush=True)
            return out
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(fn, str(p), *args) for p in paths]
            out = []
            for i, fut in enumerate(futures):                    # file order is kept
                out.append(fut.result())
                if verbose:
                    print(f"[galacticus {what}] {i + 1}/{len(paths)} {Path(paths[i]).name}", flush=True)
            return out


# ---------------------------------------------------------------------------------
# workers (module level so that they pickle for the process pool)
# ---------------------------------------------------------------------------------
def _cosmology(spec: dict):
    return cosmology_from_header({"cosmology": spec["cosmology"]})


def _read_worker(path: str, spec: dict, columns, z_range, z_column):
    return read_file(path, columns, z_range, z_column, spec["dust"])


def _flux_worker(path: str, spec: dict, line_cfg, columns, z_range, z_column):
    line_set = LineSet.from_config(line_cfg)
    cols = list(dict.fromkeys([*columns, "redshift_true", *line_set.required_columns]))
    data = read_file(path, cols, z_range, z_column, spec["dust"])
    flux = feature_fluxes(data, line_set, data["redshift_true"], _cosmology(spec))
    out = {c: data[c] for c in columns}
    out.update({name: flux[:, i] for i, name in enumerate(flux_column_names(line_set))})
    return out


def _single_lines_worker(path: str, spec: dict, line_cfg, sens_cfg, detection_cfg, seed, cuts, columns=()):
    """The single line (observed wavelength, flux, feature index) of every single-line galaxy of one file
    that passes the cuts, with the requested columns of those galaxies."""
    line_set = LineSet.from_config(line_cfg)
    cols = list(dict.fromkeys(["redshift_true", "redshift_rsd", *line_set.required_columns, *cuts, *columns]))
    data = read_file(path, cols, dust=spec["dust"])
    n = len(data["redshift_true"])
    keep = np.ones(n, dtype=bool)
    for col, (lo, hi) in cuts.items():
        if lo is not None:
            keep &= data[col] >= lo
        if hi is not None:
            keep &= data[col] <= hi
    if not keep.all():
        data = {c: v[keep] for c, v in data.items()}
    cosmo = _cosmology(spec)
    flux = feature_fluxes(data, line_set, data["redshift_true"], cosmo)
    comp = component_fluxes(data, line_set, data["redshift_true"], cosmo) if line_set.has_separated_components else None
    model = build_detection(detection_cfg, build_sensitivity(sens_cfg))
    rng = np.random.default_rng([int(seed), FRACTION_STREAM, chunk_index(path)])
    det = detect_lines(flux, np.asarray(data["redshift_rsd"], dtype=np.float64), line_set, model, rng, component_flux=comp)
    wave, f, i = single_lines(det)
    rows = np.flatnonzero(det.n_det == 1)                       # the rows of single_lines
    out = {"n": n, "n_cut": n - int(keep.sum()), "wave": wave, "flux": f, "line": i.astype(np.int8)}
    out.update({c: np.asarray(data[c])[rows] for c in columns})
    return out


def _hist_worker(path: str, spec: dict, line_cfg, z_edges, logf_edges, z_column):
    line_set = LineSet.from_config(line_cfg)
    cols = list(dict.fromkeys([z_column, "redshift_true", *line_set.required_columns]))
    data = read_file(path, cols, dust=spec["dust"])
    parts = flux_histograms(data, line_set, _cosmology(spec), z_edges, logf_edges, z_column)
    return chunk_index(path), len(data["redshift_true"]), parts


def flux_distribution(path, z_edges=DEFAULT_Z_EDGES, logf_edges=DEFAULT_LOGF_EDGES, line_set=DEFAULT_LINE_SET,
                      z_column: str = "redshift_true", dust: bool = True, files: Sequence[str | Path] | None = None,
                      n_workers: int = 1, verbose: bool = True) -> FluxDistribution:
    """Flux histograms of a Galacticus mock directory: ``GalacticusMock(path, dust).flux_distribution(...)``."""
    return GalacticusMock(path, dust=dust).flux_distribution(z_edges, logf_edges, line_set, z_column, files, None, n_workers, verbose)

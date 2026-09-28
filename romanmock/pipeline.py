"""End-to-end pipeline: diffsky slices -> parent emission-line catalog -> analysis samples.

Stage A (:meth:`RomanMockPipeline.run`) processes one lightcone slice (file) at a time
so memory stays bounded and slices can be distributed over jobs. Each slice gets a
deterministic random stream seeded by ``(seed, step)`` so results do not depend on
which subset of files a job processes. Outputs are OpenCosmo files
(``lc_cores-<step>.<tag>.hdf5``) that ``opencosmo.open`` re-assembles into a Lightcone.

Stage B (:meth:`RomanMockPipeline.build_samples`) reads the parent catalog and writes
one file per analysis sample (footprint, z range, tracer, quality, downsampling).

Two-step workflow. Reading the input mock is the expensive part of stage A and does not
depend on the choices one wants to vary (flux limits above a base depth, confusion model,
seed). :meth:`RomanMockPipeline.build_base` therefore writes a *base flux catalog* once
(galaxies with a line above the base limit, with the flux of every line), and stage A
with ``config.source = "base"`` (``romanmock assign``) recomputes detection, bitmask and
redshift assignment from it. Its output has the format of the ordinary parent catalog, so
stage B and the diagnostics apply unchanged.
"""

from __future__ import annotations

import datetime as _dt
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import opencosmo as oc

from ._version import __version__ as _pkg_version
from .config import RomanMockConfig, SampleConfig
from .confusion import build_confusion
from .detection import detect_lines
from .fluxes import FLUX_UNIT, component_fluxes, feature_fluxes, flux_column_names
from .io import (
    sample_files,
    file_info,
    iter_slices,
    load_aux_data,
    open_catalog,
    output_filename,
    read_manifest,
    select_files,
    stamp_output,
    write_dataset,
    write_manifest,
    write_run_metadata,
)
from .lines import get_line_set
from .redshift import IdOutcome, PerfectMultiLine, assign_redshifts
from .selection import (
    SampleDefinition,
    TargetDNDZ,
    build_footprint,
    dndz_from_counts,
    keep_probability,
    sky_area_deg2,
)
from .sensitivity import ConstantFluxLimit, ThresholdDetection, build_detection, build_sensitivity

#: bitmask at the limits of the base catalog. It must not be called ``line_bitmask``: OpenCosmo writes the stored
#: column instead of a new one of the same name, so step 2 would silently keep the base bitmask.
BASE_BITMASK_COLUMN = "line_bitmask_base"
IDENTIFICATION_COLUMNS = ("line_bitmask", "n_det", "line_true", "line_assigned")
RESULT_COLUMNS = ("redshift_rsd", "redshift_assigned", "id_outcome", *IDENTIFICATION_COLUMNS)
#: names base.columns may not use: written by step 1 or by step 2
BASE_DERIVED_COLUMNS = (BASE_BITMASK_COLUMN, *RESULT_COLUMNS)


class RomanMockPipeline:
    """Build the models from a :class:`RomanMockConfig` and run the stages."""

    def __init__(self, config: RomanMockConfig, aux_data: dict | None = None):
        self.config = config
        if config.source not in ("catalog", "base"):
            raise ValueError("config.source must be 'catalog' or 'base'")
        line_set = get_line_set(config.line_set)
        self.base_manifest = None
        if config.source == "base":
            # the base catalog fixes the line set (names, components and the rest wavelengths
            # used when it was built); the input mock is not opened at all
            self.base_manifest = self._read_base_manifest()
            base_lines = get_line_set(self.base_manifest["base"]["line_set"])
            if base_lines.names != line_set.names or \
                    [f.components for f in base_lines] != [f.components for f in line_set]:
                raise ValueError(f"line set of the configuration {line_set.names} differs from the one of the base "
                                 f"catalog {base_lines.names}")
            line_set = type(line_set)(list(base_lines), name=line_set.name)
            aux_data = None
        elif aux_data is None and config.catalog.rest_waves_from_catalog and config.catalog.path:
            aux_data = load_aux_data(config.catalog.path)
        if config.source == "catalog" and aux_data is not None and config.catalog.rest_waves_from_catalog:
            line_set = line_set.with_rest_waves_from(aux_data["ssp_data"])
        self.aux_data = aux_data
        self.line_set = line_set
        self.sensitivity = build_sensitivity(config.sensitivity)
        self.detection = build_detection(config.detection, self.sensitivity)
        self.confusion = build_confusion(config.confusion, self.line_set)
        self.multi_line = PerfectMultiLine(config.redshift.sigma_z)
        if config.source == "base":
            self._check_base()

    @classmethod
    def from_yaml(cls, path, **kwargs) -> "RomanMockPipeline":
        return cls(RomanMockConfig.from_yaml(path), **kwargs)

    # ------------------------------------------------------------------------------
    # column bookkeeping
    # ------------------------------------------------------------------------------
    @property
    def input_columns(self) -> list[str]:
        cfg = self.config
        cols = list(self.line_set.required_columns)
        for c in [cfg.flux.redshift_column, cfg.redshift.rsd_column, *cfg.redshift.context_columns, *cfg.output.passthrough_columns]:
            if c not in cols:
                cols.append(c)
        return cols

    @property
    def flux_columns(self) -> list[str]:
        return flux_column_names(self.line_set, self.config.flux.suffix)

    def column_descriptions(self) -> dict[str, str]:
        d = {}
        for name, feat in zip(self.flux_columns, self.line_set):
            d[name] = (f"Observed flux of {feat.label or feat.name} (sum of {'+'.join(feat.components)}), "
                       f"dust attenuation included; D_L from column {self.config.flux.redshift_column}")
        for feat in self.line_set:
            d[f"{feat.name}_wave_obs"] = f"Observed wavelength of {feat.name} [Angstrom] = rest_wave (1 + {self.config.redshift.rsd_column})"
            d[f"{feat.name}_snr"] = f"Signal-to-noise ratio of {feat.name} from the detection model"
            d[f"{feat.name}_flux_measured"] = f"Measured (noisy) flux of {feat.name} from the detection model"
        d["line_bitmask"] = "Detected-line bitmask; bit i = feature i of the line set (see header/run metadata)"
        d["n_det"] = "Number of detected features = popcount(line_bitmask)"
        d["line_true"] = "Index of the brightest detected feature (-1 if none)"
        d["line_assigned"] = "Feature index assigned by the redshift identification (-1 = none / failed)"
        d["id_outcome"] = "Identification outcome: " + IdOutcome.describe()
        d["redshift_assigned"] = "Assigned (measured) redshift; NaN when no reliable redshift"
        d["redshift_rsd"] = f"True redshift including the line-of-sight peculiar velocity (copy of {self.config.redshift.rsd_column})"
        return d

    # ------------------------------------------------------------------------------
    # core computation on arrays (I/O free, unit-testable)
    # ------------------------------------------------------------------------------
    def compute_fluxes(self, data: Mapping[str, np.ndarray], cosmology) -> tuple[np.ndarray, dict | None]:
        """Feature fluxes ``(n, n_features)`` and, for line sets with separated components,
        the per-transition fluxes ``{luminosity column: (n,)}``."""
        cfg = self.config
        z_dl = np.asarray(data[cfg.flux.redshift_column], dtype=np.float64)
        flux = feature_fluxes(data, self.line_set, z_dl, cosmology, backend=cfg.flux.backend)
        comp = component_fluxes(data, self.line_set, z_dl, cosmology, backend=cfg.flux.backend) \
            if self.line_set.has_separated_components else None
        return flux, comp

    def process_arrays(self, data: Mapping[str, np.ndarray], cosmology, rng: np.random.Generator) -> dict[str, np.ndarray]:
        cfg = self.config
        flux, comp = self.compute_fluxes(data, cosmology)
        context = {c: np.asarray(data[c]) for c in cfg.redshift.context_columns}
        return self.process_fluxes(flux, data[cfg.redshift.rsd_column], rng, component_flux=comp, context=context)

    def process_fluxes(self, flux: np.ndarray, z_rsd, rng: np.random.Generator, component_flux: dict | None = None,
                       context: Mapping[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """Detection, bitmask and redshift assignment from observed fluxes (the part of
        stage A that does not need the input mock)."""
        cfg = self.config
        z_rsd = np.asarray(z_rsd, dtype=np.float64)
        comp = component_flux
        det = detect_lines(flux, z_rsd, self.line_set, self.detection, rng, component_flux=comp)
        za = assign_redshifts(det, z_rsd, self.line_set, self.confusion, rng, self.multi_line, extra_context=dict(context or {}))

        out: dict[str, np.ndarray] = {}
        if cfg.output.store_fluxes:
            for i, name in enumerate(self.flux_columns):
                out[name] = flux[:, i]
        for i, feat in enumerate(self.line_set):
            if cfg.output.store_wave_obs:
                out[f"{feat.name}_wave_obs"] = det.wave_obs[:, i]
            if cfg.output.store_snr:
                out[f"{feat.name}_snr"] = det.snr[:, i]
            if cfg.output.store_measured_flux:
                out[f"{feat.name}_flux_measured"] = det.flux_measured[:, i]
        out["redshift_rsd"] = z_rsd
        out["redshift_assigned"] = za.z_assigned
        out["id_outcome"] = za.outcome
        out["line_bitmask"] = det.bitmask          # always computed: needed to drop undetected rows
        out["n_det"] = det.n_det
        out["line_true"] = za.line_true
        out["line_assigned"] = za.line_assigned
        return out

    # ------------------------------------------------------------------------------
    # one OpenCosmo dataset / lightcone
    # ------------------------------------------------------------------------------
    def process_dataset(self, dataset, rng: np.random.Generator):
        """Return ``dataset`` restricted to the passthrough columns plus the new result
        columns, filtered to detected galaxies unless ``output.keep_undetected``."""
        cfg = self.config
        missing = [c for c in self.input_columns if c not in dataset.columns]
        if missing:
            raise KeyError(f"catalog lacks required columns: {missing}")
        data = dataset.select(*self.input_columns).get_data("numpy", wrap_single=True)
        result = self.process_arrays(data, dataset.cosmology, rng)
        # Row selection is done with take_rows *before* adding columns: OpenCosmo's
        # filter() drops the descriptions of freshly added columns (v1.3.11).
        base = dataset.select(*cfg.output.passthrough_columns)
        if not cfg.output.keep_undetected:
            keep = np.where(result["line_bitmask"] > 0)[0]
            base = base.take_rows(keep)
            result = {k: v[keep] for k, v in result.items()}
        if not cfg.output.store_identification:
            result = {k: v for k, v in result.items() if k not in IDENTIFICATION_COLUMNS}
        descriptions = self.column_descriptions()
        new_cols = {}
        for name, arr in result.items():
            if name in self.flux_columns or name.endswith("_flux_measured"):
                new_cols[name] = arr * FLUX_UNIT
            else:
                new_cols[name] = arr
        return base.with_new_columns(descriptions={k: descriptions[k] for k in new_cols}, **new_cols)

    # ------------------------------------------------------------------------------
    # stage A: parent emission-line catalog
    # ------------------------------------------------------------------------------
    def run(
        self,
        files: Sequence[str | Path] | None = None,
        steps: Sequence[int] | None = None,
        z_min: float | None = None,
        z_max: float | None = None,
        verbose: bool = True,
        n_workers: int = 1,
    ) -> list[Path]:
        """Process lightcone slices and write one output file per slice.

        ``n_workers > 1`` processes slices in parallel with a process pool (one slice
        per task, ~2-5 GB peak memory per worker for the largest slices). Results are
        identical to a serial run because each slice has its own ``(seed, step)`` stream.

        With ``config.source == "base"`` the slices of the base flux catalog are processed
        instead of the input mock (step 2 of the two-step workflow): same outputs, under
        2 GB per worker, and ``files`` is ignored.
        """
        cfg = self.config
        if cfg.source == "base":
            return self._run_from_base(steps, z_min, z_max, verbose, n_workers)
        if not cfg.catalog.path:
            raise ValueError("config.catalog.path is not set")
        if not cfg.output.dir:
            raise ValueError("config.output.dir is not set")
        paths = select_files(
            cfg.catalog.path,
            files=files or cfg.catalog.files,
            steps=steps if steps is not None else cfg.catalog.steps,
            z_min=z_min if z_min is not None else cfg.catalog.z_min,
            z_max=z_max if z_max is not None else cfg.catalog.z_max,
        )
        out_dir = Path(cfg.output.dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._prepare_manifest(out_dir, verbose)

        todo, outputs = [], []
        for path in paths:
            out_path = output_filename(path, out_dir, cfg.output.tag)
            marker = out_path.with_suffix(".empty")
            info = file_info(path)
            entry = manifest["slices"].get(int(info.step))
            reusable = (
                entry is not None and not cfg.output.overwrite
                and entry.get("input") == path.name and entry.get("input_size") == path.stat().st_size
                and ((entry.get("n_galaxies", 0) > 0 and out_path.exists()) or (entry.get("n_galaxies") == 0 and marker.exists()))
            )
            if reusable:
                if verbose:
                    print(f"[romanmock] reuse {out_path.name}: completed with the same configuration")
                if out_path.exists():
                    outputs.append(out_path)
                continue
            if (out_path.exists() or marker.exists()) and not cfg.output.overwrite:
                raise FileExistsError(
                    f"{out_path.name} exists but is not recorded as completed with this configuration "
                    f"(manifest {out_dir / (cfg.output.tag + '_manifest.yaml')}); set output.overwrite "
                    "to redo it or remove it.")
            todo.append(path)

        metadata = self.metadata()
        metadata["config_hash"] = manifest["config_hash"]
        metadata["input_files"] = [str(p) for p in paths]
        write_run_metadata(out_dir / f"{cfg.output.tag}_metadata.yaml", metadata)

        def record(path, out_path, n_out):
            info = file_info(path)
            manifest["slices"][int(info.step)] = {
                "input": path.name, "input_size": path.stat().st_size, "output": Path(out_path).name,
                "n_galaxies": int(n_out), "completed": _dt.datetime.now().isoformat(timespec="seconds")}
            write_manifest(out_dir, cfg.output.tag, manifest)

        if n_workers <= 1 or len(todo) <= 1:
            for path in todo:
                out_path, n_out = self.process_file(path, verbose=verbose)
                record(path, out_path, n_out)
                if n_out > 0:
                    outputs.append(out_path)
            return sorted(outputs)

        # Parallel: workers rebuild the pipeline from a serialisable description with the
        # line set already resolved (no auxiliary-data loading in the workers).
        import concurrent.futures as cf
        import multiprocessing as mp

        spec = self._worker_spec()
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = {pool.submit(_run_one_file, spec, str(path)): path for path in todo}
            for fut in cf.as_completed(futures):
                msg, out_path, n_out = fut.result()  # re-raises worker exceptions
                record(futures[fut], out_path, n_out)
                if n_out > 0:
                    outputs.append(Path(out_path))
                if verbose:
                    print(msg, flush=True)
        return sorted(outputs)

    # ------------------------------------------------------------------------------
    # provenance
    # ------------------------------------------------------------------------------
    def scientific_config(self) -> dict:
        """The part of the configuration that determines the content of a slice output."""
        cfg = self.config
        source = {"source": "base", "base_dir": str(cfg.base.dir), "base_tag": cfg.base.tag,
                  "base_config_hash": self.base_manifest["config_hash"]} if cfg.source == "base" else {}
        return {
            **source,
            "line_set": self.line_set.to_config(),
            "flux": _asdict(cfg.flux),
            "sensitivity": _to_config(self.sensitivity),
            "detection": _to_config(self.detection),
            "confusion": _to_config(self.confusion),
            "redshift": _asdict(cfg.redshift),
            "seed": cfg.seed,
            "open_kwargs": dict(cfg.catalog.open_kwargs),
            "passthrough_columns": list(cfg.output.passthrough_columns),
            "keep_undetected": cfg.output.keep_undetected,
            "store": {k: getattr(cfg.output, k) for k in ("store_identification", "store_fluxes", "store_wave_obs", "store_snr", "store_measured_flux")},
        }

    def config_hash(self) -> str:
        import hashlib
        import json

        blob = json.dumps(self.scientific_config(), sort_keys=True, default=_json_default).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def _prepare_manifest(self, out_dir: Path, verbose: bool, tag: str | None = None, config_hash: str | None = None,
                          overwrite: bool | None = None) -> dict:
        """Load or create the manifest; refuse to mix outputs of different configurations."""
        cfg = self.config
        h = config_hash if config_hash is not None else self.config_hash()
        tag = tag if tag is not None else cfg.output.tag
        overwrite = cfg.output.overwrite if overwrite is None else overwrite
        manifest = read_manifest(out_dir, tag)
        if manifest is None:
            return {"config_hash": h, "romanmock_version": _pkg_version, "slices": {}}
        if manifest.get("config_hash") == h:
            manifest.setdefault("slices", {})
            manifest["slices"] = {int(k): v for k, v in manifest["slices"].items()}
            return manifest
        if not overwrite:
            raise RuntimeError(
                f"outputs in {out_dir} were produced with configuration hash {manifest.get('config_hash')} "
                f"but the current configuration has hash {h}; use a different output dir/tag, or set "
                "overwrite to discard the existing outputs.")
        if verbose:
            print(f"[romanmock] configuration changed ({manifest.get('config_hash')} -> {h}); "
                  f"discarding {len(manifest.get('slices', {}))} recorded slice outputs", flush=True)
        for entry in manifest.get("slices", {}).values():
            out = out_dir / entry["output"]
            out.unlink(missing_ok=True)
            out.with_suffix(".empty").unlink(missing_ok=True)
        return {"config_hash": h, "romanmock_version": _pkg_version, "slices": {}}

    def expected_steps(self) -> list[int]:
        """Steps of the slices selected by ``config.catalog`` (the intended parent catalog)."""
        from .io import step_from_filename

        cfg = self.config
        if cfg.source == "base":
            return sorted(self.base_slices())
        paths = select_files(cfg.catalog.path, files=cfg.catalog.files, steps=cfg.catalog.steps,
                             z_min=cfg.catalog.z_min, z_max=cfg.catalog.z_max)
        return sorted(step_from_filename(p) for p in paths)

    def process_file(self, path, verbose: bool = True) -> tuple[Path, int]:
        """Process one slice file and write its output (serial building block).

        Returns ``(output_path, n_galaxies)``. A slice with no detected galaxy writes a
        ``.empty`` marker instead of a catalog file (OpenCosmo cannot re-open an empty one).
        """
        cfg = self.config
        path = Path(path)
        info = file_info(path)
        out_path = output_filename(path, Path(cfg.output.dir), cfg.output.tag)
        if verbose:
            print(f"[romanmock] step {info.step:4d}  z=[{info.z_min:.3f},{info.z_max:.3f}]  n={info.n_gal:,}  -> {out_path.name}", flush=True)
        catalog = open_catalog([path], **cfg.catalog.open_kwargs)
        rng = np.random.default_rng([cfg.seed, info.step])
        result = self.process_dataset(catalog, rng)
        n_out = len(result)
        marker = out_path.with_suffix(".empty")
        if n_out == 0:
            out_path.unlink(missing_ok=True)
            marker.write_text(f"no galaxies selected from {path.name}\n")
            if verbose:
                print(f"[romanmock]     no galaxies selected; wrote marker {marker.name}", flush=True)
            return out_path, 0
        marker.unlink(missing_ok=True)
        write_dataset(out_path, result, overwrite=True, compress_index=cfg.output.compress_index)
        stamp_output(out_path, {"romanmock_config_hash": self.config_hash(), "romanmock_version": _pkg_version,
                                "romanmock_step": int(info.step), "romanmock_input": path.name})
        if verbose:
            print(f"[romanmock]     wrote {n_out:,} galaxies ({100.0 * n_out / max(info.n_gal, 1):.2f}% of slice)", flush=True)
        return out_path, n_out

    def _worker_spec(self) -> dict:
        import dataclasses

        cfg = dataclasses.replace(self.config, line_set=self.line_set.to_config())
        cfg.catalog = dataclasses.replace(cfg.catalog, rest_waves_from_catalog=False)
        return cfg.to_dict()

    # ------------------------------------------------------------------------------
    # two-step workflow, step 1: the base flux catalog
    # ------------------------------------------------------------------------------
    @property
    def component_flux_columns(self) -> dict[str, str]:
        """``{luminosity column: flux column}`` of the transitions of separated multi-component
        features; they are stored in the base catalog because each has its own bandpass test."""
        if not self.line_set.has_separated_components:
            return {}
        return {c: f"{c}{self.config.flux.suffix}" for feat in self.line_set if not feat.is_blend for c in feat.components}

    def base_description(self) -> dict:
        """What determines the content of the base catalog (hashed into its manifest)."""
        cfg = self.config
        return {
            "line_set": self.line_set.to_config(),
            "flux": _asdict(cfg.flux),
            "rsd_column": cfg.redshift.rsd_column,
            "sensitivity": _to_config(self.sensitivity),
            "columns": list(cfg.base.columns),
            "store_bitmask": cfg.base.store_bitmask,
            "bitmask_column": BASE_BITMASK_COLUMN,
            "flux_dtype": str(np.dtype(cfg.base.flux_dtype)),
            "open_kwargs": dict(cfg.catalog.open_kwargs),
        }

    def base_hash(self) -> str:
        import hashlib
        import json

        blob = json.dumps(self.base_description(), sort_keys=True, default=_json_default).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def base_column_descriptions(self) -> dict[str, str]:
        d = self.column_descriptions()
        lim = _to_config(self.sensitivity)
        d[BASE_BITMASK_COLUMN] = ("Line bitmask at the limits of the base catalog "
                                  f"({lim}); bit i = feature i of the line set. Step 2 computes its own line_bitmask")
        for lum, name in self.component_flux_columns.items():
            d[name] = f"Observed flux of the transition {lum}, dust attenuation included; D_L from column {self.config.flux.redshift_column}"
        return d

    def base_dataset(self, dataset):
        """Rows of ``dataset`` with at least one line above the primary flux limit inside the
        bandpass, with ``base.columns``, ``redshift_rsd``, the flux of every line and the
        bitmask at the base limits."""
        cfg = self.config
        clash = sorted(set(cfg.base.columns) & {*BASE_DERIVED_COLUMNS, *self.flux_columns, *self.component_flux_columns.values()})
        if clash:
            raise ValueError(f"base.columns must name input-catalog columns; {clash} are written by the pipeline itself")
        cols = list(dict.fromkeys([*self.line_set.required_columns, cfg.flux.redshift_column, cfg.redshift.rsd_column,
                                   *cfg.base.columns]))
        missing = [c for c in cols if c not in dataset.columns]
        if missing:
            raise KeyError(f"catalog lacks required columns: {missing}")
        data = dataset.select(*cols).get_data("numpy", wrap_single=True)
        flux, comp = self.compute_fluxes(data, dataset.cosmology)
        z_rsd = np.asarray(data[cfg.redshift.rsd_column], dtype=np.float64)
        # the base selection is always the deterministic rule on true fluxes
        det = detect_lines(flux, z_rsd, self.line_set, ThresholdDetection(self.sensitivity), component_flux=comp)
        keep = np.where(det.bitmask > 0)[0]
        dtype = np.dtype(cfg.base.flux_dtype)
        new = {"redshift_rsd": z_rsd[keep]}
        for i, name in enumerate(self.flux_columns):
            new[name] = flux[keep, i].astype(dtype) * FLUX_UNIT
        for lum, name in self.component_flux_columns.items():
            new[name] = np.asarray(comp[lum])[keep].astype(dtype) * FLUX_UNIT
        if cfg.base.store_bitmask:
            new[BASE_BITMASK_COLUMN] = det.bitmask[keep]
        descriptions = self.base_column_descriptions()
        base = dataset.select(*cfg.base.columns).take_rows(keep)
        return base.with_new_columns(descriptions={k: descriptions[k] for k in new}, **new)

    def process_base_file(self, path, verbose: bool = True) -> tuple[Path, int]:
        """Step 1 for one slice of the input mock; returns ``(output_path, n_galaxies)``."""
        cfg = self.config
        path = Path(path)
        info = file_info(path)
        out_path = output_filename(path, Path(cfg.base.dir), cfg.base.tag)
        if verbose:
            print(f"[romanmock] step {info.step:4d}  z=[{info.z_min:.3f},{info.z_max:.3f}]  n={info.n_gal:,}  -> {out_path.name}", flush=True)
        result = self.base_dataset(open_catalog([path], **cfg.catalog.open_kwargs))
        return self._write_slice(out_path, result, self.base_hash(), info.step, path.name, info.n_gal, verbose,
                                 compress_index=cfg.base.compress_index)

    def _write_slice(self, out_path: Path, result, config_hash: str, step: int, input_name: str, n_in: int,
                     verbose: bool, compress_index: bool = True) -> tuple[Path, int]:
        n_out = len(result)
        marker = out_path.with_suffix(".empty")
        if n_out == 0:
            out_path.unlink(missing_ok=True)
            marker.write_text(f"no galaxies selected from {input_name}\n")
            if verbose:
                print(f"[romanmock]     no galaxies selected; wrote marker {marker.name}", flush=True)
            return out_path, 0
        marker.unlink(missing_ok=True)
        write_dataset(out_path, result, overwrite=True, compress_index=compress_index)
        stamp_output(out_path, {"romanmock_config_hash": config_hash, "romanmock_version": _pkg_version,
                                "romanmock_step": int(step), "romanmock_input": input_name})
        if verbose:
            print(f"[romanmock]     wrote {n_out:,} galaxies ({100.0 * n_out / max(n_in, 1):.2f}% of slice)", flush=True)
        return out_path, n_out

    def build_base(
        self,
        files: Sequence[str | Path] | None = None,
        steps: Sequence[int] | None = None,
        z_min: float | None = None,
        z_max: float | None = None,
        verbose: bool = True,
        n_workers: int = 1,
    ) -> list[Path]:
        """Step 1: write the base flux catalog, one file per slice, to ``config.base.dir``.

        Slices already completed with the same base description are reused. Peak memory is
        that of stage A (~4.4 GB per worker on the largest slice)."""
        cfg = self.config
        if cfg.source != "catalog":
            raise ValueError("build_base reads the input mock: config.source must be 'catalog'")
        if not cfg.catalog.path:
            raise ValueError("config.catalog.path is not set")
        if not cfg.base.dir:
            raise ValueError("config.base.dir is not set")
        paths = select_files(
            cfg.catalog.path,
            files=files or cfg.catalog.files,
            steps=steps if steps is not None else cfg.catalog.steps,
            z_min=z_min if z_min is not None else cfg.catalog.z_min,
            z_max=z_max if z_max is not None else cfg.catalog.z_max,
        )
        out_dir = Path(cfg.base.dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._prepare_manifest(out_dir, verbose, tag=cfg.base.tag, config_hash=self.base_hash(),
                                          overwrite=cfg.base.overwrite)
        manifest["kind"] = "base"
        manifest["base"] = self.base_description()

        todo, outputs = [], []
        for path in paths:
            out_path = output_filename(path, out_dir, cfg.base.tag)
            marker = out_path.with_suffix(".empty")
            info = file_info(path)
            entry = manifest["slices"].get(int(info.step))
            reusable = (
                entry is not None and not cfg.base.overwrite
                and entry.get("input") == path.name and entry.get("input_size") == path.stat().st_size
                and ((entry.get("n_galaxies", 0) > 0 and out_path.exists()) or (entry.get("n_galaxies") == 0 and marker.exists()))
            )
            if reusable:
                if verbose:
                    print(f"[romanmock] reuse {out_path.name}: completed with the same configuration")
                if out_path.exists():
                    outputs.append(out_path)
                continue
            if (out_path.exists() or marker.exists()) and not cfg.base.overwrite:
                raise FileExistsError(
                    f"{out_path.name} exists but is not recorded as completed with this configuration "
                    f"(manifest {out_dir / (cfg.base.tag + '_manifest.yaml')}); set base.overwrite to redo it or remove it.")
            todo.append(path)

        metadata = self.metadata()
        metadata["kind"] = "base"
        metadata["config_hash"] = manifest["config_hash"]
        metadata["base"] = self.base_description()
        metadata["columns"] = [*cfg.base.columns, "redshift_rsd", *self.flux_columns, *self.component_flux_columns.values(),
                               *([BASE_BITMASK_COLUMN] if cfg.base.store_bitmask else [])]
        metadata["input_files"] = [str(p) for p in paths]
        write_run_metadata(out_dir / f"{cfg.base.tag}_metadata.yaml", metadata)

        def record(path, out_path, n_out):
            info = file_info(path)
            manifest["slices"][int(info.step)] = {
                "input": path.name, "input_size": path.stat().st_size, "output": Path(out_path).name,
                "n_galaxies": int(n_out), "n_input": int(info.n_gal), "z_min": float(info.z_min), "z_max": float(info.z_max),
                "completed": _dt.datetime.now().isoformat(timespec="seconds")}
            write_manifest(out_dir, cfg.base.tag, manifest)

        write_manifest(out_dir, cfg.base.tag, manifest)
        if n_workers <= 1 or len(todo) <= 1:
            for path in todo:
                out_path, n_out = self.process_base_file(path, verbose=verbose)
                record(path, out_path, n_out)
                if n_out > 0:
                    outputs.append(out_path)
            return sorted(outputs)

        import concurrent.futures as cf
        import multiprocessing as mp

        spec = self._worker_spec()
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = {pool.submit(_base_one_file, spec, str(path)): path for path in todo}
            for fut in cf.as_completed(futures):
                msg, out_path, n_out = fut.result()
                record(futures[fut], out_path, n_out)
                if n_out > 0:
                    outputs.append(Path(out_path))
                if verbose:
                    print(msg, flush=True)
        return sorted(outputs)

    # ------------------------------------------------------------------------------
    # two-step workflow, step 2: detection and redshifts from the base catalog
    # ------------------------------------------------------------------------------
    def _read_base_manifest(self) -> dict:
        cfg = self.config
        if not cfg.base.dir:
            raise ValueError("config.base.dir is not set")
        manifest = read_manifest(cfg.base.dir, cfg.base.tag)
        if manifest is None or manifest.get("kind") != "base":
            raise FileNotFoundError(f"no base catalog manifest {cfg.base.tag}_manifest.yaml in {cfg.base.dir}; "
                                    "run `romanmock base` first")
        manifest["slices"] = {int(k): v for k, v in (manifest.get("slices") or {}).items()}
        return manifest

    def _check_base(self):
        """The requested selection must be contained in the base catalog: nowhere deeper than
        the base primary limit and inside the base bandpass. The secondary limit is free,
        because the base catalog stores the flux of every line."""
        cfg = self.config
        if Path(cfg.base.dir).resolve() == Path(cfg.output.dir or ".").resolve() and cfg.base.tag == cfg.output.tag:
            raise ValueError("output.dir/output.tag must differ from base.dir/base.tag")
        base = self.base_manifest["base"]
        base_sens = build_sensitivity(base["sensitivity"])
        lo, hi = self.sensitivity.wave_range
        wave = np.linspace(lo, hi, 4001)[1:-1]
        req, ref = self.sensitivity.flux_limit(wave), base_sens.flux_limit(wave)
        inside = np.isfinite(req)
        if np.any(inside & ~np.isfinite(ref)):
            raise ValueError(f"requested bandpass {self.sensitivity.wave_range} extends beyond the bandpass of the "
                             f"base catalog {base_sens.wave_range}; rebuild the base catalog")
        if np.any(req[inside] < ref[inside] * (1.0 - 1e-9)):
            k = np.argmin(req[inside] / ref[inside])
            raise ValueError(f"requested flux limit {req[inside][k]:.3g} is deeper than the base catalog limit "
                             f"{ref[inside][k]:.3g} (at {wave[inside][k]:.0f} A); rebuild the base catalog deeper")
        if not isinstance(self.detection, ThresholdDetection) and np.any(req[inside] < ref[inside] * 1.5):
            warnings.warn("a noisy detection model scatters galaxies across the flux limit, but the base catalog has "
                          "none below its own limit; use a base catalog clearly deeper than the requested limit",
                          stacklevel=3)
        missing = [c for c in [*cfg.output.passthrough_columns, *cfg.redshift.context_columns]
                   if c not in [*base["columns"], "redshift_rsd"]]
        if missing:
            raise KeyError(f"columns {missing} are not in the base catalog (it has {base['columns']}); add them to "
                           "base.columns and rebuild it, or join on gal_id")

    def base_slices(self, steps: Sequence[int] | None = None, z_min: float | None = None, z_max: float | None = None) -> dict[int, dict]:
        """Base-catalog slices selected by ``config.catalog`` (``steps``, ``z_min``, ``z_max``)."""
        cfg = self.config
        steps = steps if steps is not None else cfg.catalog.steps
        z_min = z_min if z_min is not None else cfg.catalog.z_min
        z_max = z_max if z_max is not None else cfg.catalog.z_max
        out = {}
        for step, entry in sorted(self.base_manifest["slices"].items()):
            if steps is not None and step not in set(int(s) for s in steps):
                continue
            if z_min is not None and entry["z_max"] < z_min:
                continue
            if z_max is not None and entry["z_min"] > z_max:
                continue
            out[step] = entry
        if not out:
            raise FileNotFoundError("no base catalog slices match the requested selection")
        return out

    def assign_dataset(self, base, rng: np.random.Generator):
        """Step 2 for one base-catalog dataset: parent-catalog rows (same columns as stage A)."""
        cfg = self.config
        comp_cols = self.component_flux_columns
        need = list(dict.fromkeys([*self.flux_columns, *comp_cols.values(), "redshift_rsd", *cfg.redshift.context_columns]))
        missing = [c for c in need if c not in base.columns]
        if missing:
            raise KeyError(f"base catalog lacks required columns: {missing}")
        clash = sorted(set(RESULT_COLUMNS) & set(base.columns) - {"redshift_rsd"})
        if clash:
            raise RuntimeError(
                f"the base catalog stores {clash}, which step 2 has to write itself (OpenCosmo would keep the stored "
                "values); it was built by an older version: rebuild it with `romanmock base` (base.overwrite: true), "
                "or use romanmock.BaseCatalog.make_catalog, which is not affected.")
        data = base.select(*need).get_data("numpy", wrap_single=True)
        flux = np.stack([np.asarray(data[c], dtype=np.float64) for c in self.flux_columns], axis=1)
        comp = {lum: np.asarray(data[name], dtype=np.float64) for lum, name in comp_cols.items()} or None
        context = {c: np.asarray(data[c]) for c in cfg.redshift.context_columns}
        result = self.process_fluxes(flux, data["redshift_rsd"], rng, component_flux=comp, context=context)
        # columns that already exist in the base catalog are copied, not recreated
        copied = ["redshift_rsd", *(self.flux_columns if cfg.output.store_fluxes else [])]
        for c in ["redshift_rsd", *self.flux_columns]:
            result.pop(c, None)
        out = base.select(*dict.fromkeys([*cfg.output.passthrough_columns, *copied]))
        if not cfg.output.keep_undetected:
            keep = np.where(result["line_bitmask"] > 0)[0]
            out = out.take_rows(keep)
            result = {k: v[keep] for k, v in result.items()}
        if not cfg.output.store_identification:
            result = {k: v for k, v in result.items() if k not in IDENTIFICATION_COLUMNS}
        descriptions = self.column_descriptions()
        new_cols = {k: (v * FLUX_UNIT if k.endswith("_flux_measured") else v) for k, v in result.items()}
        return out.with_new_columns(descriptions={k: descriptions[k] for k in new_cols}, **new_cols)

    def process_base_slice(self, step: int, verbose: bool = True) -> tuple[Path, int]:
        """Step 2 for one slice of the base catalog; returns ``(output_path, n_galaxies)``."""
        cfg = self.config
        entry = self.base_manifest["slices"][int(step)]
        out_path = output_filename(entry["input"], Path(cfg.output.dir), cfg.output.tag)
        if verbose:
            print(f"[romanmock] step {step:4d}  z=[{entry['z_min']:.3f},{entry['z_max']:.3f}]  "
                  f"base n={entry['n_galaxies']:,}  -> {out_path.name}", flush=True)
        if entry["n_galaxies"] == 0:
            out_path.unlink(missing_ok=True)
            out_path.with_suffix(".empty").write_text(f"no galaxies in the base catalog slice of {entry['input']}\n")
            return out_path, 0
        base = oc.open(Path(cfg.base.dir) / entry["output"])
        rng = np.random.default_rng([cfg.seed, int(step)])
        result = self.assign_dataset(base, rng)
        return self._write_slice(out_path, result, self.config_hash(), step, entry["output"], entry["n_galaxies"], verbose,
                                 compress_index=cfg.output.compress_index)

    def _run_from_base(self, steps, z_min, z_max, verbose: bool, n_workers: int) -> list[Path]:
        cfg = self.config
        if not cfg.output.dir:
            raise ValueError("config.output.dir is not set")
        slices = self.base_slices(steps, z_min, z_max)
        out_dir = Path(cfg.output.dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._prepare_manifest(out_dir, verbose)
        manifest["base_config_hash"] = self.base_manifest["config_hash"]

        todo, outputs = [], []
        for step, entry in slices.items():
            out_path = output_filename(entry["input"], out_dir, cfg.output.tag)
            marker = out_path.with_suffix(".empty")
            done = manifest["slices"].get(int(step))
            reusable = (
                done is not None and not cfg.output.overwrite
                and done.get("input") == entry["output"] and done.get("base_completed") == entry.get("completed")
                and ((done.get("n_galaxies", 0) > 0 and out_path.exists()) or (done.get("n_galaxies") == 0 and marker.exists()))
            )
            if reusable:
                if verbose:
                    print(f"[romanmock] reuse {out_path.name}: completed with the same configuration")
                if out_path.exists():
                    outputs.append(out_path)
                continue
            if (out_path.exists() or marker.exists()) and not cfg.output.overwrite:
                raise FileExistsError(
                    f"{out_path.name} exists but is not recorded as completed with this configuration "
                    f"(manifest {out_dir / (cfg.output.tag + '_manifest.yaml')}); set output.overwrite "
                    "to redo it or remove it.")
            todo.append(step)

        metadata = self.metadata()
        metadata["config_hash"] = manifest["config_hash"]
        metadata["base"] = {"dir": str(cfg.base.dir), "tag": cfg.base.tag, "config_hash": self.base_manifest["config_hash"],
                            "description": self.base_manifest["base"]}
        write_run_metadata(out_dir / f"{cfg.output.tag}_metadata.yaml", metadata)

        def record(step, out_path, n_out):
            entry = slices[step]
            manifest["slices"][int(step)] = {
                "input": entry["output"], "base_completed": entry.get("completed"), "output": Path(out_path).name,
                "n_galaxies": int(n_out), "completed": _dt.datetime.now().isoformat(timespec="seconds")}
            write_manifest(out_dir, cfg.output.tag, manifest)

        if n_workers <= 1 or len(todo) <= 1:
            for step in todo:
                out_path, n_out = self.process_base_slice(step, verbose=verbose)
                record(step, out_path, n_out)
                if n_out > 0:
                    outputs.append(out_path)
            return sorted(outputs)

        import concurrent.futures as cf
        import multiprocessing as mp

        spec = self._worker_spec()
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = {pool.submit(_assign_one_slice, spec, int(step)): step for step in todo}
            for fut in cf.as_completed(futures):
                msg, out_path, n_out = fut.result()
                record(futures[fut], out_path, n_out)
                if n_out > 0:
                    outputs.append(Path(out_path))
                if verbose:
                    print(msg, flush=True)
        return sorted(outputs)

    # ------------------------------------------------------------------------------
    # stage B: analysis samples
    # ------------------------------------------------------------------------------
    def parent_files(self, parent_dir=None, allow_partial: bool = False, check_config: bool = True) -> list[Path]:
        """Slice outputs recorded as completed in the manifest.

        Raises if the manifest is missing, was produced with a different configuration
        (unless ``check_config=False``), or does not cover every slice selected by
        ``config.catalog`` (unless ``allow_partial=True``).
        """
        d = Path(parent_dir or self.config.output.dir)
        tag = self.config.output.tag
        manifest = read_manifest(d, tag)
        if manifest is None:
            raise FileNotFoundError(f"no manifest {tag}_manifest.yaml in {d}; run stage A first")
        if check_config and manifest.get("config_hash") != self.config_hash():
            raise RuntimeError(
                f"parent catalog in {d} has configuration hash {manifest.get('config_hash')}, current config has "
                f"{self.config_hash()}; use the matching configuration or pass check_config=False.")
        slices = {int(k): v for k, v in manifest.get("slices", {}).items()}
        missing = [s for s in self.expected_steps() if s not in slices]
        if missing and not allow_partial:
            raise RuntimeError(f"parent catalog is incomplete: {len(missing)} of the selected slices are missing "
                               f"(steps {missing[:8]}{'...' if len(missing) > 8 else ''}); rerun stage A or pass allow_partial=True.")
        files = sorted(d / v["output"] for v in slices.values() if v.get("n_galaxies", 0) > 0)
        lost = [f for f in files if not f.exists()]
        if lost:
            raise FileNotFoundError(f"manifest lists outputs that do not exist: {[f.name for f in lost]}")
        if not files:
            raise FileNotFoundError(f"no non-empty parent catalog files in {d}")
        return files

    def open_parent(self, parent_dir=None, allow_partial: bool = False, check_config: bool = True):
        return oc.open(*self.parent_files(parent_dir, allow_partial=allow_partial, check_config=check_config))

    def build_samples(self, parent_dir=None, samples: Sequence[SampleConfig] | None = None, verbose: bool = True,
                      allow_partial: bool = False, n_workers: int = 1) -> list[Path]:
        """Stage B: analysis samples, built slice by slice (see :mod:`romanmock.samples`).

        Each sample is written to ``<parent>/samples/<name>/`` as one OpenCosmo file per
        slice (open it with :func:`romanmock.io.open_sample`), or to the single file
        ``<parent>/samples/<name>.hdf5`` with ``output.sample_layout = "single"``. The
        slice layout needs about 1.3 GB per worker whatever the size of the catalog; the
        single file is assembled by OpenCosmo and needs ~0.3 GB per slice of the sample.
        Returns the paths of the non-empty samples.
        """
        import shutil

        from . import samples as _samples

        cfg = self.config
        samples = list(samples if samples is not None else cfg.samples)
        if not samples:
            raise ValueError("no samples defined in config.samples")
        if cfg.output.sample_layout not in ("slices", "single"):
            raise ValueError("output.sample_layout must be 'slices' or 'single'")
        names = [s.name for s in samples]
        if len(set(names)) != len(names):
            raise ValueError(f"sample names must be unique; got {names}")
        files = self.parent_files(parent_dir, allow_partial=allow_partial)
        parent = oc.open(*files)
        out_dir = Path(parent_dir or cfg.output.dir) / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- plan: what every worker needs to know ------------------------------------
        need = ["redshift_assigned", "id_outcome", "line_assigned", "line_bitmask", "n_det", "ra", "dec"]
        for scfg in samples:
            need += list(scfg.extra_cuts.keys())
        tcfg = cfg.population_thinning
        if tcfg is not None:
            need += [tcfg.by, tcfg.z_column]
        need = list(dict.fromkeys(need))
        missing = [c for c in need if c not in parent.columns]
        if missing:
            raise KeyError(f"parent catalog lacks {missing}; stage A must be run with output.store_identification=true")
        columns = None
        if cfg.output.sample_columns:
            columns = [c for c in cfg.output.sample_columns if c in parent.columns]
            absent = sorted(set(cfg.output.sample_columns) - set(columns))
            if absent:
                warnings.warn(f"sample columns not in the parent catalog are skipped: {absent}")

        single = cfg.output.sample_layout == "single"
        targets_out, plan_samples = [], []
        for scfg in samples:
            final = out_dir / (f"{scfg.name}.hdf5" if single else scfg.name)
            other = out_dir / (scfg.name if single else f"{scfg.name}.hdf5")
            work = out_dir / f".{scfg.name}_slices" if single else final
            for path in (final, other):
                exists = path.is_file() or (path.is_dir() and any(path.glob("*.hdf5")))
                if exists and not cfg.output.overwrite:
                    raise FileExistsError(f"sample output {path} exists; set output.overwrite to replace it")
            targets_out.append((final, other, work))
            plan_samples.append({
                "name": scfg.name, "dir": str(work), "columns": columns, "edges": None, "p_keep": None,
                "definition": {"name": scfg.name, "z_min": scfg.z_min, "z_max": scfg.z_max, "outcomes": list(scfg.outcomes),
                               "assigned_lines": scfg.assigned_lines, "membership": scfg.membership,
                               "n_det_min": scfg.n_det_min, "footprint": scfg.footprint, "extra_cuts": dict(scfg.extra_cuts)},
            })
        plan = {"seed": int(cfg.seed), "line_set": self.line_set.to_config(), "columns": need, "thinning": None,
                "samples": plan_samples, "compress_index": cfg.output.compress_index}

        def run(tasks):
            return _samples.run_pass(plan, files, tasks, n_workers)

        # ---- pass 1: population thinning ------------------------------------------------
        thinning_info, pop_targets = {}, {}
        if tcfg is not None:
            if tcfg.outside_range not in ("keep", "drop"):
                raise ValueError("outside_range must be 'keep' or 'drop'")
            area_pop = tcfg.area_deg2 if tcfg.area_deg2 is not None else sky_area_deg2(parent, tcfg.nside)
            pops = {}
            for name, spec in tcfg.targets.items():
                target = self._target(spec, default_line=name)
                pop_targets[name] = target
                edges = getattr(target, "edges", None)
                if edges is None and hasattr(target, "z_range"):
                    edges = _regular_edges(*target.z_range, tcfg.bin_width)
                pops[name] = {"bit": int(self.line_set.bit(name)), "p_keep": None,
                              "edges": None if edges is None else np.asarray(edges, dtype=np.float64)}
            plan["thinning"] = {"by": tcfg.by, "z_column": tcfg.z_column, "outside_range": tcfg.outside_range,
                                "populations": pops}
            unknown = [name for name, pop in pops.items() if pop["edges"] is None]
            if unknown:
                ranges = run({"pop_range": unknown})
                for name in unknown:
                    lo, hi = _merge_ranges(r["pop_range"][name] for r in ranges)
                    pops[name]["edges"] = None if lo > hi else _regular_edges(lo, hi, tcfg.bin_width)
            active = [name for name, pop in pops.items() if pop["edges"] is not None]
            results = run({"pop_hist": active})
            for name in active:
                pop = pops[name]
                n_before = sum(r["pop_n"][name] for r in results)
                if n_before == 0:
                    continue
                edges = pop["edges"]
                zc = 0.5 * (edges[1:] + edges[:-1])
                counts = np.sum([r["pop_hist"][name] for r in results], axis=0)
                mock = dndz_from_counts(counts, edges, area_pop, tcfg.smooth_bins, tcfg.poly_degree)
                tgt = np.asarray(pop_targets[name](zc), dtype=np.float64)
                empty = (mock == 0) & (tgt > 0)
                if np.any(empty):   # not a shortage of the mock: nobody has this line as `by` there
                    warnings.warn(
                        f"{name}: {empty.sum()} of {empty.size} target bins (z {edges[:-1][empty].min():.2f}-"
                        f"{edges[1:][empty].max():.2f}) contain no galaxies with {tcfg.by} == {name!r}; "
                        "the target cannot be applied there (another line is the brightest one).",
                        RuntimeWarning, stacklevel=2)
                pop["p_keep"] = keep_probability(mock, tgt, zc, warn_empty=False)
                thinning_info[name] = {"z_edges": edges, "z_center": zc, "p_keep": pop["p_keep"], "mock": mock,
                                       "target": tgt, "n_before": int(n_before), "n_after": None, "area_deg2": float(area_pop)}

        # ---- pass 2: downsampling of individual samples ----------------------------------
        sample_info = {}
        down = [k for k, scfg in enumerate(samples) if scfg.downsample is not None]
        if down:
            need_range = []
            for k in down:
                scfg, dcfg = samples[k], samples[k].downsample
                if dcfg.z_edges is not None:
                    plan_samples[k]["edges"] = np.asarray(dcfg.z_edges, dtype=float)
                elif scfg.z_min is not None and scfg.z_max is not None:
                    plan_samples[k]["edges"] = np.linspace(scfg.z_min, scfg.z_max, dcfg.nbins + 1)
                else:
                    need_range.append(k)
            if need_range:
                results = run({"sample_range": need_range})
                for k in need_range:
                    scfg, dcfg = samples[k], samples[k].downsample
                    lo, hi = _merge_ranges(r["sample_range"][k] for r in results)
                    if lo > hi:          # nothing selected: the sample will be empty
                        continue
                    lo = scfg.z_min if scfg.z_min is not None else lo
                    hi = scfg.z_max if scfg.z_max is not None else hi
                    plan_samples[k]["edges"] = np.linspace(lo, hi, dcfg.nbins + 1)
            active = [k for k in down if plan_samples[k]["edges"] is not None]
            results = run({"sample_hist": active}) if active else []
            for k in active:
                scfg, dcfg = samples[k], samples[k].downsample
                edges = plan_samples[k]["edges"]
                counts = np.sum([r["sample_hist"][k] for r in results], axis=0)
                if counts.sum() == 0:
                    continue
                target = self._target(dcfg.target if dcfg.target is not None else dcfg.target_file,
                                      default_line=(scfg.assigned_lines or [None])[0])
                if dcfg.area_deg2 is not None:
                    area = dcfg.area_deg2
                else:  # area of the lightcone coverage inside the sample footprint
                    area = sky_area_deg2(parent, dcfg.nside, footprint=build_footprint(scfg.footprint))
                zc = 0.5 * (edges[1:] + edges[:-1])
                mock = dndz_from_counts(counts, edges, area, dcfg.smooth_bins, dcfg.poly_degree)
                plan_samples[k]["p_keep"] = keep_probability(mock, np.asarray(target(zc), dtype=np.float64), zc)
                sample_info[k] = {"area_deg2": float(area)}
                if verbose:
                    p = plan_samples[k]["p_keep"]
                    print(f"[romanmock] sample {scfg.name!r}: area {area:.2f} deg^2, p_keep range {p.min():.3f}-{p.max():.3f}")

        # ---- pass 3: selection and writing ---------------------------------------------------
        for final, other, work in targets_out:
            for path in {final, other, work}:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            work.mkdir(parents=True, exist_ok=True)
        results = run({"write": True})
        n_parent = sum(r["n"] for r in results)
        for name, info in thinning_info.items():
            info["n_after"] = int(sum(r["pop_after"][name] for r in results))
            if verbose:
                print(f"[romanmock] population thinning {name!r}: {info['n_before']:,} -> {info['n_after']:,} "
                      f"(p_keep {info['p_keep'].min():.3f}-{info['p_keep'].max():.3f}, area {info['area_deg2']:.0f} deg^2)",
                      flush=True)

        outputs = []
        for k, (scfg, (final, other, work)) in enumerate(zip(samples, targets_out)):
            n_keep = int(sum(r["written"][k] for r in results))
            if verbose:
                print(f"[romanmock] sample {scfg.name!r}: {n_keep:,} of {n_parent:,} parent galaxies", flush=True)
            meta = {"sample": _asdict(scfg), "n_galaxies": n_keep, "parent_config_hash": self.config_hash(),
                    "layout": cfg.output.sample_layout, "created": _dt.datetime.now().isoformat(timespec="seconds")}
            meta_path = out_dir / f"{scfg.name}_metadata.yaml"
            if n_keep == 0:
                # do not leave a stale non-empty catalog looking like the current result (review B2)
                shutil.rmtree(work, ignore_errors=True)
                meta["empty"] = True
                write_run_metadata(meta_path, meta)
                warnings.warn(f"sample {scfg.name!r} is empty; previous output removed, metadata records the empty result")
                continue
            slice_files = sample_files(work)
            if single:
                write_dataset(final, oc.open(*slice_files), overwrite=True, compress_index=cfg.output.compress_index)
                shutil.rmtree(work)
                meta["path"] = final.name
            else:
                meta["path"] = final.name
                meta["files"] = [f.name for f in slice_files]
            if plan_samples[k]["p_keep"] is not None:
                meta["p_keep"] = np.asarray(plan_samples[k]["p_keep"]).tolist()
                meta["z_edges"] = np.asarray(plan_samples[k]["edges"]).tolist()
                meta["area_deg2"] = sample_info[k]["area_deg2"]
            if thinning_info:
                meta["population_thinning"] = {
                    name: {"z_edges": v["z_edges"].tolist(), "p_keep": v["p_keep"].tolist(),
                           "n_before": v["n_before"], "n_after": v["n_after"]}
                    for name, v in thinning_info.items()}
            write_run_metadata(meta_path, meta)
            outputs.append(final)
        return outputs

    def _target(self, spec, default_line: str | None = None) -> TargetDNDZ:
        """Resolve a target spec (dict, file path, or TargetDNDZ) into a TargetDNDZ."""
        if isinstance(spec, TargetDNDZ):
            return spec
        if spec is None:
            raise ValueError("a downsampling target is required (reference, file, or z/dndz)")
        if isinstance(spec, (str, Path)):
            return TargetDNDZ.from_file(spec)
        spec = dict(spec)
        if "reference" in spec or not ({"file", "z"} & set(spec)):
            from .reference_dndz import reference_target

            name = spec.pop("reference", "wang2022")
            line = spec.pop("line", default_line)
            if line is None:
                raise ValueError("reference target needs a 'line' (or a sample with assigned_lines)")
            if "flux_limit" not in spec:
                if not isinstance(self.sensitivity, ConstantFluxLimit):
                    raise ValueError("give 'flux_limit' explicitly for a non-constant sensitivity model")
                spec["flux_limit"] = self.sensitivity.flux_limit_value
            return reference_target(name, line=line, **spec)
        if "file" in spec:
            return TargetDNDZ.from_file(spec["file"])
        return TargetDNDZ(spec["z"], spec["dndz"])

    # ------------------------------------------------------------------------------
    def metadata(self) -> dict:
        import opencosmo

        versions = {"romanmock": _pkg_version, "opencosmo": opencosmo.__version__, "numpy": np.__version__}
        try:
            import diffsky

            versions["diffsky"] = diffsky.__version__
        except Exception:  # pragma: no cover
            pass
        return {
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "versions": versions,
            "config": self.config.to_dict(),
            "line_set": {"name": self.line_set.name, "features": self.line_set.to_config(),
                         "bit_ordering": self.line_set.describe()},
            "id_outcome_codes": {o.name: int(o) for o in IdOutcome},
            "sensitivity": _to_config(self.sensitivity),
            "detection": _to_config(self.detection),
            "confusion": _to_config(self.confusion),
        }


def _run_one_file(spec: dict, path: str) -> str:
    """Process pool worker: rebuild the pipeline from ``spec`` and process one slice."""
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    pipe = RomanMockPipeline(RomanMockConfig.from_dict(spec), aux_data=None)
    info = file_info(path)
    out, n_out = pipe.process_file(path, verbose=False)
    msg = (f"[romanmock] step {info.step:4d}  z=[{info.z_min:.3f},{info.z_max:.3f}]  "
           f"n={info.n_gal:,} -> {out.name}: {n_out:,} galaxies ({100.0 * n_out / max(info.n_gal, 1):.2f}%)")
    return msg, str(out), n_out


def _base_one_file(spec: dict, path: str):
    """Process pool worker of :meth:`RomanMockPipeline.build_base`."""
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    pipe = RomanMockPipeline(RomanMockConfig.from_dict(spec), aux_data=None)
    info = file_info(path)
    out, n_out = pipe.process_base_file(path, verbose=False)
    msg = (f"[romanmock] step {info.step:4d}  z=[{info.z_min:.3f},{info.z_max:.3f}]  "
           f"n={info.n_gal:,} -> {out.name}: {n_out:,} galaxies ({100.0 * n_out / max(info.n_gal, 1):.2f}%)")
    return msg, str(out), n_out


def _assign_one_slice(spec: dict, step: int):
    """Process pool worker of stage A on the base catalog (``config.source == "base"``)."""
    pipe = RomanMockPipeline(RomanMockConfig.from_dict(spec), aux_data=None)
    n_base = pipe.base_manifest["slices"][int(step)]["n_galaxies"]
    out, n_out = pipe.process_base_slice(step, verbose=False)
    msg = (f"[romanmock] step {step:4d}  base n={n_base:,} -> {out.name}: {n_out:,} galaxies "
           f"({100.0 * n_out / max(n_base, 1):.2f}%)")
    return msg, str(out), n_out


def _regular_edges(lo: float, hi: float, bin_width: float) -> np.ndarray:
    n = max(int(np.ceil((hi - lo) / bin_width - 1e-9)), 1)
    return lo + bin_width * np.arange(n + 1)


def _merge_ranges(ranges) -> tuple[float, float]:
    ranges = list(ranges)
    return min(r[0] for r in ranges), max(r[1] for r in ranges)


def _to_config(obj):
    return obj.to_config() if hasattr(obj, "to_config") else repr(obj)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    return str(o)


def _asdict(obj):
    import dataclasses

    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else dict(obj)



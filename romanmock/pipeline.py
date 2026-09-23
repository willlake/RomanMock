"""End-to-end pipeline: diffsky slices -> parent emission-line catalog -> analysis samples.

Stage A (:meth:`RomanMockPipeline.run`) processes one lightcone slice (file) at a time
so memory stays bounded and slices can be distributed over jobs. Each slice gets a
deterministic random stream seeded by ``(seed, step)`` so results do not depend on
which subset of files a job processes. Outputs are OpenCosmo files
(``lc_cores-<step>.<tag>.hdf5``) that ``opencosmo.open`` re-assembles into a Lightcone.

Stage B (:meth:`RomanMockPipeline.build_samples`) reads the parent catalog and writes
one file per analysis sample (footprint, z range, tracer, quality, downsampling).
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
from .fluxes import FLUX_UNIT, feature_fluxes, flux_column_names
from .io import (
    file_info,
    iter_slices,
    load_aux_data,
    open_catalog,
    output_filename,
    select_files,
    write_dataset,
    write_run_metadata,
)
from .lines import get_line_set
from .redshift import IdOutcome, PerfectMultiLine, assign_redshifts
from .selection import (
    SampleDefinition,
    TargetDNDZ,
    build_footprint,
    downsample_to_dndz,
    mock_dndz,
    sample_mask,
    sky_area_deg2,
)
from .sensitivity import build_detection, build_sensitivity

RESULT_COLUMNS = ("line_bitmask", "n_det", "line_true", "line_assigned", "id_outcome", "z_obs")


class RomanMockPipeline:
    """Build the models from a :class:`RomanMockConfig` and run the stages."""

    def __init__(self, config: RomanMockConfig, aux_data: dict | None = None):
        self.config = config
        line_set = get_line_set(config.line_set)
        if aux_data is None and config.catalog.rest_waves_from_catalog and config.catalog.path:
            aux_data = load_aux_data(config.catalog.path)
        if aux_data is not None and config.catalog.rest_waves_from_catalog:
            line_set = line_set.with_rest_waves_from(aux_data["ssp_data"])
        self.aux_data = aux_data
        self.line_set = line_set
        self.sensitivity = build_sensitivity(config.sensitivity)
        self.detection = build_detection(config.detection, self.sensitivity)
        self.confusion = build_confusion(config.confusion, self.line_set)
        self.multi_line = PerfectMultiLine(config.redshift.sigma_z)

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
        d["z_obs"] = "Assigned (measured) redshift; NaN when no reliable redshift"
        return d

    # ------------------------------------------------------------------------------
    # core computation on arrays (I/O free, unit-testable)
    # ------------------------------------------------------------------------------
    def process_arrays(self, data: Mapping[str, np.ndarray], cosmology, rng: np.random.Generator) -> dict[str, np.ndarray]:
        cfg = self.config
        z_dl = np.asarray(data[cfg.flux.redshift_column], dtype=np.float64)
        z_rsd = np.asarray(data[cfg.redshift.rsd_column], dtype=np.float64)
        flux = feature_fluxes(data, self.line_set, z_dl, cosmology, backend=cfg.flux.backend)
        det = detect_lines(flux, z_rsd, self.line_set, self.detection, rng)
        context = {c: np.asarray(data[c]) for c in cfg.redshift.context_columns}
        za = assign_redshifts(det, z_rsd, self.line_set, self.confusion, rng, self.multi_line, extra_context=context)

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
        out["line_bitmask"] = det.bitmask
        out["n_det"] = det.n_det
        out["line_true"] = za.line_true
        out["line_assigned"] = za.line_assigned
        out["id_outcome"] = za.outcome
        out["z_obs"] = za.z_obs
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
        """
        cfg = self.config
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
        write_run_metadata(out_dir / f"{cfg.output.tag}_metadata.yaml", self.metadata())

        todo, outputs = [], []
        for path in paths:
            out_path = output_filename(path, out_dir, cfg.output.tag)
            marker = out_path.with_suffix(".empty")
            if (out_path.exists() or marker.exists()) and not cfg.output.overwrite:
                if verbose:
                    print(f"[romanmock] skip {out_path.name}: exists (set output.overwrite to redo)")
                if out_path.exists():
                    outputs.append(out_path)
            else:
                todo.append(path)

        if n_workers <= 1 or len(todo) <= 1:
            for path in todo:
                out_path, n_out = self.process_file(path, verbose=verbose)
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
                if n_out > 0:
                    outputs.append(Path(out_path))
                if verbose:
                    print(msg, flush=True)
        return sorted(outputs)

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
        write_dataset(out_path, result, overwrite=cfg.output.overwrite)
        if verbose:
            print(f"[romanmock]     wrote {n_out:,} galaxies ({100.0 * n_out / max(info.n_gal, 1):.2f}% of slice)", flush=True)
        return out_path, n_out

    def _worker_spec(self) -> dict:
        import dataclasses

        cfg = dataclasses.replace(self.config, line_set=self.line_set.to_config())
        cfg.catalog = dataclasses.replace(cfg.catalog, rest_waves_from_catalog=False)
        return cfg.to_dict()

    # ------------------------------------------------------------------------------
    # stage B: analysis samples
    # ------------------------------------------------------------------------------
    def parent_files(self, parent_dir=None) -> list[Path]:
        d = Path(parent_dir or self.config.output.dir)
        files = sorted(d.glob(f"lc_cores-*.{self.config.output.tag}.hdf5"))
        if not files:
            raise FileNotFoundError(f"no parent catalog files lc_cores-*.{self.config.output.tag}.hdf5 in {d}")
        return files

    def open_parent(self, parent_dir=None):
        return oc.open(*self.parent_files(parent_dir))

    def build_samples(self, parent_dir=None, samples: Sequence[SampleConfig] | None = None, verbose: bool = True) -> list[Path]:
        cfg = self.config
        samples = list(samples if samples is not None else cfg.samples)
        if not samples:
            raise ValueError("no samples defined in config.samples")
        parent = self.open_parent(parent_dir)
        out_dir = Path(parent_dir or cfg.output.dir) / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for k, scfg in enumerate(samples):
            sdef = SampleDefinition(
                name=scfg.name, z_min=scfg.z_min, z_max=scfg.z_max, outcomes=scfg.outcomes,
                assigned_lines=scfg.assigned_lines, n_det_min=scfg.n_det_min,
                footprint=build_footprint(scfg.footprint), extra_cuts=scfg.extra_cuts,
            )
            need = ["z_obs", "id_outcome", "line_assigned", "n_det", "ra", "dec", *scfg.extra_cuts.keys()]
            data = parent.select(*dict.fromkeys(need)).get_data("numpy", wrap_single=True)
            keep = sample_mask(data, self.line_set, sdef)
            p_keep = None
            if scfg.downsample is not None:
                dcfg = scfg.downsample
                target = _target_from_cfg(dcfg)
                if dcfg.area_deg2 is not None:
                    area = dcfg.area_deg2
                else:  # area of the lightcone coverage inside the sample footprint
                    area = sky_area_deg2(parent, dcfg.nside, footprint=sdef.footprint)
                z_sel = data["z_obs"][keep]
                if dcfg.z_edges is not None:
                    z_edges = np.asarray(dcfg.z_edges, dtype=float)
                else:
                    lo = scfg.z_min if scfg.z_min is not None else np.nanmin(z_sel)
                    hi = scfg.z_max if scfg.z_max is not None else np.nanmax(z_sel)
                    z_edges = np.linspace(lo, hi, dcfg.nbins + 1)
                rng = np.random.default_rng([cfg.seed, 10_000 + k])
                mock = mock_dndz(z_sel, z_edges, area, dcfg.smooth_bins)
                sub_keep, p_keep = downsample_to_dndz(z_sel, target, area, z_edges, rng, mock_dndz_values=mock)
                idx = np.where(keep)[0]
                keep = np.zeros_like(keep)
                keep[idx[sub_keep]] = True
                if verbose:
                    print(f"[romanmock] sample {scfg.name!r}: area {area:.2f} deg^2, p_keep range "
                          f"{p_keep.min():.3f}-{p_keep.max():.3f}")
            n_keep = int(keep.sum())
            if verbose:
                print(f"[romanmock] sample {scfg.name!r}: {n_keep:,} of {keep.size:,} parent galaxies", flush=True)
            if n_keep == 0:
                warnings.warn(f"sample {scfg.name!r} is empty; nothing written")
                continue
            out = parent.take_rows(np.where(keep)[0])
            out_path = out_dir / f"{scfg.name}.hdf5"
            write_dataset(out_path, out, overwrite=cfg.output.overwrite)
            meta = {"sample": _asdict(scfg), "n_galaxies": n_keep}
            if p_keep is not None:
                meta["p_keep"] = p_keep.tolist()
                meta["z_edges"] = z_edges.tolist()
            write_run_metadata(out_dir / f"{scfg.name}_metadata.yaml", meta)
            outputs.append(out_path)
        return outputs

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


def _to_config(obj):
    return obj.to_config() if hasattr(obj, "to_config") else repr(obj)


def _asdict(obj):
    import dataclasses

    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else dict(obj)


def _target_from_cfg(dcfg) -> TargetDNDZ:
    if dcfg.target_file:
        return TargetDNDZ.from_file(dcfg.target_file)
    if dcfg.target:
        return TargetDNDZ(dcfg.target["z"], dcfg.target["dndz"])
    raise ValueError("downsample needs target_file or target {z, dndz}")

"""Command line interface: ``romanmock <command> ...``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _jax_platform(args):
    # diffsky imports JAX; on nodes without a GPU this avoids noisy CUDA probing.
    if getattr(args, "jax_platform", None):
        os.environ.setdefault("JAX_PLATFORMS", args.jax_platform)


def cmd_info(args):
    import numpy as np

    from .io import file_info, list_catalog_files, mock_name, open_catalog
    from .lines import DIFFSKY_LINE_WAVELENGTHS

    files = list_catalog_files(args.catalog_dir)
    print(f"mock: {mock_name(files[0])}   ({len(files)} slices)")
    tot = 0
    for f in files:
        info = file_info(f)
        tot += info.n_gal
        print(f"  step {info.step:4d}  z=[{info.z_min:.4f},{info.z_max:.4f}]  n={info.n_gal:>12,}  {f.name}")
    print(f"total galaxies: {tot:,}")
    cat = open_catalog([files[-1]])
    print("cosmology:", cat.cosmology)
    present = [c for c in DIFFSKY_LINE_WAVELENGTHS if c in cat.columns]
    print("line luminosity columns:", present, "unit:", cat.units.get(present[0]) if present else None)


def cmd_lines(args):
    from .lines import LINE_SET_PRESETS, get_line_set

    names = [args.preset] if args.preset else list(LINE_SET_PRESETS)
    for n in names:
        print(get_line_set(n).describe())
        print()


def cmd_example_config(args):
    from .config import RomanMockConfig, SampleConfig, DownsampleConfig

    from .config import default_confusion_rows

    cfg = RomanMockConfig()
    cfg.confusion = {"type": "matrix", "default_p_fail": 0.0, "rows": default_confusion_rows(cfg.line_set)}
    cfg.catalog.path = args.catalog or "/path/to/diffsky/mock"
    cfg.output.dir = args.output or "/path/to/output"
    cfg.samples = [SampleConfig(name="halpha_z1p0_1p9", z_min=1.0, z_max=1.9, assigned_lines=["Ha"],
                                downsample=DownsampleConfig(target_file="target_dndz.txt", nbins=45))]
    sys.stdout.write(cfg.to_yaml())


def cmd_run(args):
    _jax_platform(args)
    from .pipeline import RomanMockPipeline

    pipe = RomanMockPipeline.from_yaml(args.config)
    if args.print_config:
        print(pipe.config.to_yaml())
        print(pipe.line_set.describe())
    if args.workers == "auto":
        from .io import memory_budget_gb, suggest_n_workers

        n_workers = suggest_n_workers()
        print(f"[romanmock] memory budget {memory_budget_gb():.0f} GB -> {n_workers} worker(s)")
    else:
        n_workers = int(args.workers)
    outputs = pipe.run(files=args.files, steps=args.steps, z_min=args.z_min, z_max=args.z_max, verbose=not args.quiet,
                       n_workers=n_workers)
    print(f"[romanmock] wrote {len(outputs)} file(s) to {pipe.config.output.dir}")


def _n_workers(args, per_worker_gb=None):
    if args.workers != "auto":
        return int(args.workers)
    from .io import memory_budget_gb, suggest_n_workers

    n = suggest_n_workers() if per_worker_gb is None else suggest_n_workers(per_worker_gb=per_worker_gb)
    print(f"[romanmock] memory budget {memory_budget_gb():.0f} GB -> {n} worker(s)")
    return n


def cmd_base(args):
    """Two-step workflow, step 1: the base flux catalog."""
    _jax_platform(args)
    from .config import RomanMockConfig
    from .pipeline import RomanMockPipeline

    cfg = RomanMockConfig.from_yaml(args.config)
    cfg.source = "catalog"
    if args.base_dir:
        cfg.base.dir = args.base_dir
    pipe = RomanMockPipeline(cfg)
    if args.print_config:
        print(pipe.config.to_yaml())
        print(pipe.line_set.describe())
    outputs = pipe.build_base(files=args.files, steps=args.steps, z_min=args.z_min, z_max=args.z_max,
                              verbose=not args.quiet, n_workers=_n_workers(args))
    print(f"[romanmock] base catalog: {len(outputs)} file(s) in {cfg.base.dir} "
          f"(limits {pipe.sensitivity}; columns {cfg.base.columns} + redshift_rsd + line fluxes)")


def cmd_assign(args):
    """Two-step workflow, step 2: detection and redshift assignment from the base catalog."""
    _jax_platform(args)
    from .config import RomanMockConfig
    from .pipeline import RomanMockPipeline

    cfg = RomanMockConfig.from_yaml(args.config)
    cfg.source = "base"
    if args.base_dir:
        cfg.base.dir = args.base_dir
    if args.output_dir:
        cfg.output.dir = args.output_dir
    if args.flux_limit is not None or args.secondary_flux_limit is not None:
        if cfg.sensitivity.get("type", "constant") != "constant":
            raise SystemExit("--flux-limit/--secondary-flux-limit apply to the constant sensitivity model; "
                             "edit the sensitivity block of the configuration instead")
        if args.flux_limit is not None:
            cfg.sensitivity["flux_limit"] = args.flux_limit
        if args.secondary_flux_limit is not None:
            cfg.sensitivity["secondary_flux_limit"] = None if args.secondary_flux_limit <= 0 else args.secondary_flux_limit
    if args.seed is not None:
        cfg.seed = args.seed
    pipe = RomanMockPipeline(cfg)
    if args.print_config:
        print(pipe.config.to_yaml())
    outputs = pipe.run(steps=args.steps, z_min=args.z_min, z_max=args.z_max, verbose=not args.quiet,
                       n_workers=_n_workers(args, per_worker_gb=2.0))
    effective = Path(cfg.output.dir) / "config.yaml"
    cfg.to_yaml(effective)
    print(f"[romanmock] wrote {len(outputs)} file(s) to {cfg.output.dir} (limits {pipe.sensitivity})")
    print(f"[romanmock] effective configuration: {effective}  (use it for `romanmock select` / `romanmock summary`)")


def cmd_catalog(args):
    """Step 2 in one call: a catalog with the given settings from the base catalog, written to a file."""
    _jax_platform(args)
    import yaml

    from .catalog import BaseCatalog

    base = BaseCatalog(args.base_dir, area_deg2=args.area)
    confusion = None
    if args.confusion == "empirical":                  # the line mix of the mock, built on the fly
        confusion = {"type": "empirical"}
    elif args.confusion and args.confusion.endswith(".npz"):   # a saved line-fraction table
        confusion = {"type": "empirical", "file": args.confusion}
    elif args.confusion:
        with open(args.confusion) as f:
            confusion = yaml.safe_load(f)
        confusion = confusion.get("confusion", confusion) if isinstance(confusion, dict) else confusion
        if isinstance(confusion, dict) and "type" not in confusion and "rows" in confusion:
            confusion = confusion["rows"]
    secondary = "default" if args.secondary_flux_limit is None else (None if args.secondary_flux_limit <= 0 else args.secondary_flux_limit)
    downsample = None
    if args.downsample:
        downsample = {"reference": args.downsample, "dust_av": args.dust_av}
    columns = args.columns if args.columns != ["all"] else "all"
    cat = base.make_catalog(
        flux_limit=args.flux_limit, secondary_flux_limit=secondary, confusion=confusion, z_range=args.z_range,
        z_column=args.z_column, lines=args.lines, membership=args.membership, n_det_min=args.n_det_min, downsample=downsample, columns=columns,
        seed=args.seed, n_workers=None if args.workers == "auto" else int(args.workers), verbose=not args.quiet)
    print(cat.summary())
    print(f"[romanmock] written: {cat.write(args.output)}")


def cmd_line_fractions(args):
    """The line mix of the single-line galaxies of the base catalog per observed wavelength and flux rank."""
    _jax_platform(args)
    from .catalog import BaseCatalog

    base = BaseCatalog(args.base_dir)
    secondary = "default" if args.secondary_flux_limit is None else (None if args.secondary_flux_limit <= 0 else args.secondary_flux_limit)
    base.line_fractions(flux_limit=args.flux_limit, secondary_flux_limit=secondary, wave_bin=args.wave_bin,
                        galaxies_per_bin=args.galaxies_per_bin, n_flux_bins=args.n_flux_bins, flux_quantiles=args.flux_quantiles,
                        min_count=args.min_count, mode=args.mode, p_fail=args.p_fail, seed=args.seed,
                        n_workers=None if args.workers == "auto" else int(args.workers), write_to=args.output, verbose=not args.quiet)
    print(f"[romanmock] line fractions written to {args.output}")


def cmd_census(args):
    """dN/dz census of the input mock: emitters above flux limits, single-line and multi-line."""
    _jax_platform(args)
    import numpy as np

    from .census import line_census

    secondary = [None if s <= 0 else s for s in (args.secondary_flux_limits or [0.0])]
    limits = [(p, s) for p in args.flux_limits for s in secondary]
    lo, hi = args.z_range
    edges = np.round(lo + args.dz * np.arange(int(round((hi - lo) / args.dz)) + 1), 6)
    census = line_census(args.catalog, limits, edges, line_set=args.line_set, n_workers=_n_workers(args, 5.0),
                         verbose=not args.quiet)
    path = census.save(args.output)
    print(f"[romanmock] census of {len(census.steps)} slices, limits (primary, secondary) {census.limits} -> {path}")


def cmd_flux_distribution(args):
    """Histograms of the line fluxes of the whole mock per redshift bin (no flux cut)."""
    _jax_platform(args)
    import numpy as np

    from .fluxdist import flux_distribution

    lo, hi = args.z_range
    edges = np.round(lo + args.dz * np.arange(int(round((hi - lo) / args.dz)) + 1), 6)
    result = flux_distribution(args.catalog, edges, line_set=args.line_set, n_workers=_n_workers(args, 5.0),
                               verbose=not args.quiet)
    path = result.save(args.output)
    print(f"[romanmock] flux distribution of {len(result.steps)} slices, {int(result.n_gal.sum()):,} galaxies -> {path}")


def cmd_random(args):
    """Random catalogs for a catalog written by `romanmock catalog` / `Catalog.write`."""
    from .catalog import Catalog

    cat = Catalog.read(args.catalog)
    which = args.which[0] if len(args.which) == 1 else args.which
    output = args.output or str(Path(args.catalog).with_name(Path(args.catalog).stem + "_random.hdf5"))
    cat.make_random(which=which, n_times=args.n_times, z_column=args.z_column, seed=args.seed, write_to=output,
                    first=args.first, verbose=not args.quiet)


def cmd_compress_index(args):
    """Convert existing output files in place: spatial index gzip-compressed, data untouched."""
    from .io import compress_index_in_place

    files = []
    for p in map(Path, args.paths):
        if p.is_dir():
            files += sorted(q for q in p.rglob("*.hdf5") if not q.name.startswith("."))
        elif p.exists():
            files.append(p)
        else:
            raise SystemExit(f"{p} does not exist")
    if not files:
        raise SystemExit("no .hdf5 files found")
    n_workers = max(1, int(args.workers))
    if n_workers > 1:
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
            results = list(pool.map(compress_index_in_place, map(str, files)))
    else:
        results = [compress_index_in_place(f) for f in files]
    before = sum(r[1] for r in results)
    after = sum(r[2] for r in results)
    done = sum(r[3] for r in results)
    if not args.quiet:
        for path, b, a, conv in results:
            print(f"  {'converted' if conv else 'unchanged'}  {b / 1e6:9.1f} MB -> {a / 1e6:9.1f} MB  {path}")
    print(f"[romanmock] {done} of {len(results)} file(s) converted: {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB")


def cmd_select(args):
    _jax_platform(args)
    from .pipeline import RomanMockPipeline

    pipe = RomanMockPipeline.from_yaml(args.config)
    outputs = pipe.build_samples(parent_dir=args.parent_dir, verbose=not args.quiet, allow_partial=args.allow_partial,
                                 n_workers=_n_workers(args, per_worker_gb=2.0))
    for p in outputs:
        print(f"[romanmock] sample written: {p}" + ("/  (one file per slice; romanmock.io.open_sample)" if p.is_dir() else ""))


def cmd_summary(args):
    _jax_platform(args)
    import numpy as np
    import opencosmo as oc

    from .diagnostics import empirical_confusion, outcome_summary
    from .lines import get_line_set
    from .pipeline import RomanMockPipeline

    pipe = RomanMockPipeline.from_yaml(args.config)
    parent = pipe.open_parent(args.parent_dir, allow_partial=args.allow_partial)
    data = parent.select("id_outcome", "n_det", "line_true", "line_assigned").get_data("numpy", wrap_single=True)
    print(f"parent catalog: {len(parent):,} galaxies, z_range {parent.z_range}")
    for k, v in outcome_summary(data).items():
        print(f"  {k:<15s} {v:>12,}")
    m = empirical_confusion(data, pipe.line_set)
    names = pipe.line_set.names + ["fail"]
    print("empirical confusion among single-line galaxies (rows: true, cols: assigned):")
    print("            " + " ".join(f"{n[:9]:>9s}" for n in names))
    for i, n in enumerate(pipe.line_set.names):
        print(f"  {n[:9]:<9s} " + " ".join(f"{x:9.4f}" for x in m[i]))


def main(argv=None):
    p = argparse.ArgumentParser(prog="romanmock", description="Roman-like emission-line mock from a diffsky lightcone")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("info", help="list the slices, redshift ranges and line columns of a diffsky mock")
    s.add_argument("catalog_dir")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("lines", help="print the bit ordering of the line-set presets")
    s.add_argument("preset", nargs="?")
    s.set_defaults(func=cmd_lines)

    s = sub.add_parser("example-config", help="print an example YAML configuration")
    s.add_argument("--catalog")
    s.add_argument("--output")
    s.set_defaults(func=cmd_example_config)

    s = sub.add_parser("base", help="two-step workflow, step 1: base flux catalog (galaxies with a line above the "
                                    "primary limit; ra, dec, redshifts, all line fluxes, bitmask)")
    s.add_argument("config", help="YAML configuration file")
    s.add_argument("--base-dir", help="directory of the base catalog (default: base.dir)")
    s.add_argument("--files", nargs="*", help="subset of slice files to process")
    s.add_argument("--steps", nargs="*", type=int, help="subset of step numbers to process")
    s.add_argument("--z-min", type=float)
    s.add_argument("--z-max", type=float)
    s.add_argument("--print-config", action="store_true")
    s.add_argument("--workers", default="1", help="number of parallel slice processes, or 'auto' (~5 GB each)")
    s.add_argument("--jax-platform", default="cpu", help="JAX_PLATFORMS value for diffsky imports (default cpu)")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_base)

    s = sub.add_parser("catalog", help="step 2 in one call: build a catalog from the base catalog with the given flux "
                                       "limits, confusion matrix, lines and redshift range, and write it to a file")
    s.add_argument("base_dir", help="directory of the base catalog (written by `romanmock base`)")
    s.add_argument("-o", "--output", required=True, help="output file (.hdf5, .fits or .npz); replaced if it exists")
    s.add_argument("--flux-limit", type=float, help="primary flux limit [erg/s/cm^2] (default: the base catalog's)")
    s.add_argument("--secondary-flux-limit", type=float, help="limit for the other lines (default 4e-17; 0 = no second tier)")
    s.add_argument("--confusion", help="YAML file with the rows {true line: {assigned line: probability}}; or a line-fraction "
                                       "table (.npz, from `romanmock line-fractions`); or `empirical` to build that table first")
    s.add_argument("--lines", nargs="+", help="keep galaxies in which one of these lines is identified, e.g. Ha")
    s.add_argument("--membership", default="detected", choices=["detected", "assigned"],
                   help="multi-line galaxies: every detected line counts (detected), or only the brightest (assigned)")
    s.add_argument("--z-range", nargs=2, type=float, metavar=("ZMIN", "ZMAX"))
    s.add_argument("--z-column", default="redshift_assigned", choices=["redshift_assigned", "redshift_rsd", "redshift_true"])
    s.add_argument("--n-det-min", type=int)
    s.add_argument("--downsample", help="reference dN/dz to thin the emitter populations to, e.g. wang2022 (default: none)")
    s.add_argument("--dust-av", type=float, default=1.65, help="dust model of the reference (1.65 or 1.92)")
    s.add_argument("--columns", nargs="+", help="columns to keep, or 'all'")
    s.add_argument("--area", type=float, help="sky area of the lightcone [deg^2] (default: estimated)")
    s.add_argument("--seed", type=int, default=12345)
    s.add_argument("--workers", default="auto")
    s.add_argument("--jax-platform", default="cpu", help="JAX_PLATFORMS value (default cpu)")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_catalog)

    s = sub.add_parser("line-fractions", help="the line mix of the single-line galaxies of the base catalog per bin of "
                                              "observed wavelength and flux rank: the table of the empirical line identification "
                                              "(`romanmock catalog --confusion table.npz`, YAML `confusion: {type: empirical, file: ...}`)")
    s.add_argument("base_dir", help="directory of the base catalog (written by `romanmock base`)")
    s.add_argument("-o", "--output", required=True, help="output file (.npz), read with LineFractionTable.load")
    s.add_argument("--flux-limit", type=float, help="primary flux limit of the catalog the table is for (default: the base catalog's)")
    s.add_argument("--secondary-flux-limit", type=float, help="limit for the other lines (default 4e-17; 0 = no second tier)")
    s.add_argument("--wave-bin", type=float, default=250.0, help="width of the observed-wavelength bins [A] (default 250)")
    s.add_argument("--galaxies-per-bin", type=int, default=500, help="each wavelength bin is cut into as many equal-count flux-rank "
                                                                      "bins as hold at least this many galaxies (default 500)")
    s.add_argument("--n-flux-bins", type=int, help="instead: this many equal-count flux-rank bins in every wavelength bin")
    s.add_argument("--flux-quantiles", nargs="+", type=float, help="instead: the quantiles of the flux bins, from 0 to 1")
    s.add_argument("--min-count", type=int, default=100, help="lines that are the single line of fewer galaxies are left out")
    s.add_argument("--mode", default="draw", choices=["draw", "max"], help="draw a line with the shares of the cell, or take the most common")
    s.add_argument("--p-fail", type=float, default=0.0, help="uniform probability of no redshift")
    s.add_argument("--seed", type=int, default=12345)
    s.add_argument("--workers", default="auto")
    s.add_argument("--jax-platform", default="cpu", help="JAX_PLATFORMS value (default cpu)")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_line_fractions)

    s = sub.add_parser("census", help="dN/dz of the emitters of the input mock above flux limits, split into "
                                      "single-line and multi-line; no confusion, no downsampling")
    s.add_argument("catalog", help="directory of the diffsky mock")
    s.add_argument("-o", "--output", required=True, help="output file (.npz), read with LineCensus.load")
    s.add_argument("--flux-limits", nargs="+", type=float, default=[1.0e-16], help="primary limits [erg/s/cm^2]")
    s.add_argument("--secondary-flux-limits", nargs="+", type=float,
                   help="secondary limits, each combined with every primary limit; 0 = single threshold")
    s.add_argument("--z-range", nargs=2, type=float, default=[0.5, 3.0], metavar=("ZMIN", "ZMAX"))
    s.add_argument("--dz", type=float, default=0.1, help="width of the redshift bins")
    s.add_argument("--line-set", default="roman_transitions")
    s.add_argument("--workers", default="auto", help="number of parallel slice processes, or 'auto' (~5 GB each)")
    s.add_argument("--jax-platform", default="cpu", help="JAX_PLATFORMS value (default cpu)")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_census)

    s = sub.add_parser("flux-distribution", help="histograms of the line fluxes of every galaxy of the input mock per "
                                                 "redshift bin, with no flux cut (quantiles, fractions above a limit)")
    s.add_argument("catalog", help="directory of the diffsky mock")
    s.add_argument("-o", "--output", required=True, help="output file (.npz), read with FluxDistribution.load")
    s.add_argument("--z-range", nargs=2, type=float, default=[0.0, 4.0], metavar=("ZMIN", "ZMAX"))
    s.add_argument("--dz", type=float, default=0.1, help="width of the redshift bins")
    s.add_argument("--line-set", default="roman_transitions")
    s.add_argument("--workers", default="auto", help="number of parallel slice processes, or 'auto' (~5 GB each)")
    s.add_argument("--jax-platform", default="cpu", help="JAX_PLATFORMS value (default cpu)")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_flux_distribution)

    s = sub.add_parser("random", help="random catalogs for a catalog: uniform inside the footprint, redshifts drawn "
                                      "from the data; N files of the size of the data")
    s.add_argument("catalog", help="catalog file (.hdf5 or .npz) written by `romanmock catalog`")
    s.add_argument("-o", "--output", help="name.hdf5 gives name_00.hdf5, name_01.hdf5, ... (default: <catalog>_random.hdf5)")
    s.add_argument("-n", "--n-times", type=int, default=1, help="number of random catalogs, each of the size of the data")
    s.add_argument("--which", nargs="+", default=["all"], choices=["all", "galaxies", "interlopers"],
                   help="data catalog: all, galaxies (correct redshift), interlopers (wrong redshift)")
    s.add_argument("--z-column", help="redshift column (default: the one of the redshift cut)")
    s.add_argument("--first", type=int, default=0, help="index of the first random catalog")
    s.add_argument("--seed", type=int, default=12345)
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_random)

    s = sub.add_parser("compress-index", help="convert existing output files in place: store the spatial index "
                                              "compressed (268 MB -> ~1 MB per file), data unchanged")
    s.add_argument("paths", nargs="+", help="output files or directories (searched recursively for *.hdf5)")
    s.add_argument("--workers", default="1", help="number of parallel processes (under 1 GB each)")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_compress_index)

    s = sub.add_parser("assign", help="two-step workflow, step 2: apply flux limits to the base catalog, build the "
                                      "bitmask and assign redshifts (writes a parent catalog)")
    s.add_argument("config", help="YAML configuration file")
    s.add_argument("--base-dir", help="directory of the base catalog (default: base.dir)")
    s.add_argument("--output-dir", help="directory of the outputs (default: output.dir)")
    s.add_argument("--flux-limit", type=float, help="primary flux limit [erg/s/cm^2]; not deeper than the base catalog")
    s.add_argument("--secondary-flux-limit", type=float,
                   help="limit for the other lines of a galaxy with a primary line; 0 switches the second tier off")
    s.add_argument("--seed", type=int)
    s.add_argument("--steps", nargs="*", type=int, help="subset of step numbers to process")
    s.add_argument("--z-min", type=float)
    s.add_argument("--z-max", type=float)
    s.add_argument("--print-config", action="store_true")
    s.add_argument("--workers", default="1", help="number of parallel slice processes, or 'auto' (~2 GB each)")
    s.add_argument("--jax-platform", default="cpu", help="JAX_PLATFORMS value for diffsky imports (default cpu)")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_assign)

    for name, fn, help_ in [("run", cmd_run, "stage A: fluxes, detection, bitmask, redshift assignment"),
                            ("select", cmd_select, "stage B: build analysis samples from the parent catalog"),
                            ("summary", cmd_summary, "outcome counts and empirical confusion of a parent catalog")]:
        s = sub.add_parser(name, help=help_)
        s.add_argument("config", help="YAML configuration file")
        s.add_argument("--jax-platform", default="cpu", help="JAX_PLATFORMS value for diffsky imports (default cpu)")
        s.add_argument("--quiet", action="store_true")
        if name == "run":
            s.add_argument("--files", nargs="*", help="subset of slice files to process")
            s.add_argument("--steps", nargs="*", type=int, help="subset of step numbers to process")
            s.add_argument("--z-min", type=float)
            s.add_argument("--z-max", type=float)
            s.add_argument("--print-config", action="store_true")
            s.add_argument("--workers", default="1", help="number of parallel slice processes, or 'auto' to fit the memory budget (~5 GB each)")
        else:
            s.add_argument("--parent-dir", help="directory of the stage-A outputs (default: output.dir)")
            s.add_argument("--allow-partial", action="store_true", help="accept a parent catalog that lacks some of the selected slices")
            if name == "select":
                s.add_argument("--workers", default="1", help="number of parallel slice processes, or 'auto' (~2 GB each)")
        s.set_defaults(func=fn)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()

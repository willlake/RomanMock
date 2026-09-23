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

    cfg = RomanMockConfig()
    cfg.catalog.path = args.catalog or "/path/to/diffsky/mock"
    cfg.output.dir = args.output or "/path/to/output"
    cfg.confusion = {"type": "matrix", "rows": {"Ha": {"OIII_5007": 0.05, "fail": 0.05}}, "default_p_fail": 0.02}
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
    outputs = pipe.run(files=args.files, steps=args.steps, z_min=args.z_min, z_max=args.z_max, verbose=not args.quiet)
    print(f"[romanmock] wrote {len(outputs)} file(s) to {pipe.config.output.dir}")


def cmd_select(args):
    _jax_platform(args)
    from .pipeline import RomanMockPipeline

    pipe = RomanMockPipeline.from_yaml(args.config)
    outputs = pipe.build_samples(parent_dir=args.parent_dir, verbose=not args.quiet)
    for p in outputs:
        print(f"[romanmock] sample written: {p}")


def cmd_summary(args):
    _jax_platform(args)
    import numpy as np
    import opencosmo as oc

    from .diagnostics import empirical_confusion, outcome_summary
    from .lines import get_line_set
    from .pipeline import RomanMockPipeline

    pipe = RomanMockPipeline.from_yaml(args.config)
    parent = pipe.open_parent(args.parent_dir)
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
        else:
            s.add_argument("--parent-dir", help="directory of the stage-A outputs (default: output.dir)")
        s.set_defaults(func=fn)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()

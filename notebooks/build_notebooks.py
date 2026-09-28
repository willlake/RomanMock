"""Generate (and optionally execute) the tutorial notebooks.

    python notebooks/build_notebooks.py            # write .ipynb files
    python notebooks/build_notebooks.py --execute  # also run them and store outputs

Notebooks are generated from code so they stay in sync with the package; edit this
file rather than the .ipynb files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent

SETUP = '''import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # diffsky imports JAX; avoid CUDA probing on login nodes
import warnings
warnings.filterwarnings("ignore", message="Version checking")

import numpy as np
import matplotlib.pyplot as plt
import opencosmo as oc

from romanmock import plotting
plotting.use_style()

CATALOG = "/global/cfs/cdirs/m4943/Simulations/sdss_feniks_hizels_260710_09_07_2026"
WORK = os.path.join(os.environ.get("PSCRATCH", "/tmp"), "romanmock_demo")   # outputs of the tutorials
os.makedirs(WORK, exist_ok=True)'''


def md(s):
    return new_markdown_cell(s.strip())


def code(s):
    return new_code_cell(s.strip())


# =====================================================================================
# 01 - explore the diffsky catalog with OpenCosmo
# =====================================================================================
NB01 = [
    md('''
# 01 — Exploring the diffsky mock with OpenCosmo

The input of `romanmock` is Andrew Hearin's diffsky lightcone. This notebook shows how the
catalog is organised, what the emission-line columns contain, and how to turn the stored line
luminosities (erg/s) into observed fluxes (erg/s/cm²) with `romanmock.add_line_fluxes`.
'''),
    code(SETUP),
    md('''
## Slices of the lightcone

The mock is split into thin redshift shells, one file per HACC step. `romanmock.io` reads the
headers only (cheap) to list them.
'''),
    code('''
from romanmock.io import list_catalog_files, file_info, open_catalog, load_aux_data

files = list_catalog_files(CATALOG)
print(f"{len(files)} slices")
infos = [file_info(f) for f in files]
for info in infos[::6]:
    print(f"step {info.step:4d}  z = [{info.z_min:.3f}, {info.z_max:.3f}]  n = {info.n_gal:>12,}")
print("total galaxies:", f"{sum(i.n_gal for i in infos):,}")'''),
    md('''
## Opening a slice

`open_catalog` wraps `opencosmo.open` with the flags diffsky's own loader uses. The result is an
`opencosmo.Lightcone`; nothing is read from disk until `get_data` is called.
'''),
    code('''
cat = open_catalog([f for f in files if f.name.startswith("lc_cores-219.")])
print(cat.z_range, len(cat), "galaxies")
print(cat.cosmology)'''),
    code('''
line_cols = ["Ba_alpha_6563", "Ba_beta_4861", "O_III_4363", "O_III_4959", "O_III_5007",
             "O_II_3726", "O_II_3729", "S_III_6312", "S_III_9069", "S_III_9532"]
for c in line_cols + ["redshift_obs", "redshift_true", "vpec", "av"]:
    print(f"{c:<16s} {str(cat.units[c]):<10s} {cat.descriptions[c]}")'''),
    md('''
The diffsky auxiliary files carry the SSP metadata, including the vacuum rest wavelength of each
line. `romanmock` uses these wavelengths (its built-in table is identical).
'''),
    code('''
aux = load_aux_data(CATALOG)
wave = aux["ssp_data"].ssp_emline_wave
for name, w in zip(wave._fields, wave):
    print(f"{name:<14s} {w:9.3f} A")'''),
    md('''
## Line luminosities

Read a contiguous block of 300k galaxies. The luminosities include dust attenuation and are
continuum subtracted.
'''),
    code('''
sub = cat.take(300_000, at="start")
lum = sub.select(*line_cols, "redshift_obs", "logsm_obs").get_data("numpy", wrap_single=True)

fig, ax = plt.subplots()
series = [("Ha", lum["Ba_alpha_6563"]), ("OIII_5007", lum["O_III_5007"]), ("OII_3727", lum["O_II_3726"] + lum["O_II_3729"])]
bins = np.linspace(38, 44.5, 66)
for name, L in series:
    ax.hist(np.log10(L), bins=bins, histtype="step", color=plotting.feature_color(name), label=name)
ax.set_xlabel("log10 line luminosity [erg/s]"); ax.set_ylabel("galaxies")
ax.set_title(f"Dust-attenuated line luminosities, step 219 (z ≈ {cat.z_range[0]:.2f}–{cat.z_range[1]:.2f})")
ax.legend(); plt.show()'''),
    md('''
## From luminosity to flux

`add_line_fluxes` sums the components of each feature (for example the [O II] doublet), converts
with `F = L / (4π D_L²)` using the cosmology of the file header, and returns a dataset with one
`<feature>_flux` column per feature, in erg/s/cm². The luminosity distance is evaluated at the
cosmological redshift (`redshift_true`).
'''),
    code('''
from romanmock import add_line_fluxes, get_line_set

line_set = get_line_set("roman_transitions")
print(line_set.describe())
with_flux = add_line_fluxes(sub, line_set)
flux = with_flux.select(*[f"{f.name}_flux" for f in line_set]).get_data("numpy", wrap_single=True)
print(with_flux.units["Ha_flux"], "|", with_flux.descriptions["OII_3727_flux"])'''),
    code('''
from romanmock.roman import DEFAULT_LINE_FLUX_LIMIT, DEFAULT_SECONDARY_LINE_FLUX_LIMIT

fig, ax = plt.subplots()
bins = np.linspace(-19, -14.5, 70)
for name in ["Ha", "OIII_5007", "OII_3727"]:
    ax.hist(np.log10(flux[f"{name}_flux"]), bins=bins, histtype="step", color=plotting.feature_color(name), label=name)
ymax = ax.get_ylim()[1]
for lim, ls, y, text in [(DEFAULT_LINE_FLUX_LIMIT, "--", 0.95, "default limit"),
                         (DEFAULT_SECONDARY_LINE_FLUX_LIMIT, ":", 0.85, "secondary limit, galaxies with a line above the default")]:
    ax.axvline(np.log10(lim), color=plotting.TEXT_SECONDARY, lw=1, ls=ls)
    ax.text(np.log10(lim) + 0.05, ymax * y, f"{lim:.1e} erg/s/cm² ({text})", color=plotting.TEXT_SECONDARY, fontsize=8, va="top")
ax.set_xlabel("log10 observed line flux [erg/s/cm²]"); ax.set_ylabel("galaxies")
ax.set_title("Observed line fluxes at z ≈ 1.2"); ax.legend(); plt.show()'''),
    md('''
## Which lines fall in the Roman grism?

The grism covers 1.00–1.93 μm. Plotting `λ_obs = λ₀ (1 + z)` for every feature shows which
redshift ranges each line is observable in, and where two lines can be confused
(for example [O III] 5007 at z ≈ 2.2 and Hα at z ≈ 1.44 both land at 1.6 μm).
'''),
    code('''
from romanmock.roman import GRISM_WAVE_RANGE_ANGSTROM

z = np.linspace(0, 4, 400)
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.axhspan(*[w / 1e4 for w in GRISM_WAVE_RANGE_ANGSTROM], color=plotting.GRID, zorder=0)
ax.text(0.05, GRISM_WAVE_RANGE_ANGSTROM[1] / 1e4 - 0.05, "Roman grism 1.00–1.93 μm", color=plotting.TEXT_SECONDARY, fontsize=8, va="top")
ymax = 3.5
labels = []
for feat in line_set:
    if feat.name in ("OIII_4363", "SIII_6312"):
        continue
    lam = feat.rest_wave * (1 + z) / 1e4
    ax.plot(z, lam, color=plotting.feature_color(feat.name), label=feat.name)
    if lam[-1] <= ymax:                       # direct labels only where the line reaches the right edge
        labels.append([z[-1] + 0.05, lam[-1], feat.name])
labels.sort(key=lambda l: l[1])               # nudge labels apart vertically
for k in range(1, len(labels)):
    if labels[k][0] > 4 and labels[k - 1][0] > 4 and labels[k][1] - labels[k - 1][1] < 0.13:
        labels[k][1] = labels[k - 1][1] + 0.13
for x, y, name in labels:
    ax.text(x, min(y, ymax - 0.05), name, color=plotting.feature_color(name), fontsize=8, va="center")
ax.set_xlim(0, 4.6); ax.set_ylim(0.3, ymax)
ax.set_xlabel("redshift"); ax.set_ylabel("observed wavelength [μm]")
ax.set_title("Observed wavelength of each feature"); ax.legend(loc="upper left", fontsize=8); plt.show()'''),
]

# =====================================================================================
# 02 - the pipeline step by step on arrays
# =====================================================================================
NB02 = [
    md('''
# 02 — The pipeline step by step

Every physical step of the plan is a function that works on plain numpy arrays, so it can be
inspected and tested without the catalog machinery. This notebook walks through

1. fluxes → 2. detection and bitmask → 3. confusion matrix and redshift assignment

on a mixture of two slices: step 219 (z ≈ 1.2, Hα and [O III] both in the grism) and step 148
(z ≈ 2.3, only [O III]/[O II]/Hβ in the grism), so single-line [O III] galaxies exist.
'''),
    code(SETUP),
    code('''
from romanmock.io import open_catalog
from romanmock.lines import get_line_set

line_set = get_line_set("roman_transitions")
cols = line_set.required_columns + ["redshift_true", "redshift_obs", "gal_id"]

parts = []
for step in (219, 148):
    cat = open_catalog([f"{CATALOG}/lc_cores-{step}.diffsky_gals.hdf5"])
    parts.append(cat.take(200_000, at="start").select(*cols).get_data("numpy", wrap_single=True))
    cosmology = cat.cosmology
data = {c: np.concatenate([p[c] for p in parts]) for c in cols}
z_rsd = data["redshift_obs"]
print(len(z_rsd), "galaxies; z_rsd range", z_rsd.min().round(3), z_rsd.max().round(3))'''),
    md('''
## 1. Fluxes

`feature_fluxes` sums the component columns of each feature and converts to erg/s/cm².
'''),
    code('''
from romanmock import feature_fluxes

flux = feature_fluxes(data, line_set, data["redshift_true"], cosmology)   # (n, n_features)
print(flux.shape)
for i, feat in enumerate(line_set):
    print(f"{feat.name:<10s} median flux {np.median(flux[:, i]):.2e} erg/s/cm²")'''),
    md('''
## 2. Detection and the bitmask

A line is detectable if `F > F_lim(λ_obs)` with `λ_obs = λ₀ (1 + z_rsd)` inside the bandpass.
`ConstantFluxLimit` is the plan's initial model, with two tiers: a galaxy needs one line above the
primary limit (default 1.0e-16 erg/s/cm²); once it has one, its other lines inside the bandpass count as
detected above the lower secondary limit (default 4.0e-17). A galaxy without a primary line has no
detected line at all. Bit *i* of the bitmask is feature *i* of the line set; `n_det = popcount(bitmask)`.
'''),
    code('''
from romanmock import ConstantFluxLimit, ThresholdDetection, detect_lines, unpack_bitmask

rng = np.random.default_rng(0)
from romanmock.roman import DEFAULT_LINE_FLUX_LIMIT, DEFAULT_SECONDARY_LINE_FLUX_LIMIT
sens = ConstantFluxLimit(flux_limit=DEFAULT_LINE_FLUX_LIMIT, wave_range=(10000.0, 19300.0),
                         secondary_flux_limit=DEFAULT_SECONDARY_LINE_FLUX_LIMIT)
print(sens)
model = ThresholdDetection(sens)
det = detect_lines(flux, z_rsd, line_set, model, rng)

print("bitmask dtype", det.bitmask.dtype, "| example bitmasks", det.bitmask[:8])
print("n_det       ", det.n_det[:8])
print("bit 0 (Ha) of the first 8:", unpack_bitmask(det.bitmask[:8], len(line_set))[:, 0].astype(int))'''),
    code('''
# effect of the second tier: same galaxies, more lines per galaxy
single_tier = detect_lines(flux, z_rsd, line_set, ThresholdDetection(
    ConstantFluxLimit(DEFAULT_LINE_FLUX_LIMIT, (10000.0, 19300.0), secondary_flux_limit=None)))
print("same detected galaxies:", np.array_equal(det.n_det > 0, single_tier.n_det > 0),
      "| lines above the primary limit identical:", np.array_equal(det.primary, single_tier.detected))
n = (det.n_det > 0).sum()
print(f"detected galaxies {n:,}; multi-line fraction {np.mean(single_tier.n_det[det.n_det > 0] > 1):.3f} "
      f"(primary limit only) -> {np.mean(det.n_det[det.n_det > 0] > 1):.3f} (with the secondary limit)")
for i, name in enumerate(line_set.names):
    a, b = single_tier.detected[:, i].sum(), det.detected[:, i].sum()
    if b:
        print(f"  {name:<10s} {a:>8,} -> {b:>8,} detected lines ({b - a:,} between the two limits)")'''),
    code('''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
counts = np.bincount(det.n_det, minlength=5)
plotting.bar(ax1, [str(i) for i in range(len(counts))], counts)
ax1.set_xlabel("number of detected features"); ax1.set_ylabel("galaxies"); ax1.set_title("N_det = popcount(bitmask)")

frac = det.detected.mean(axis=0)
plotting.bar(ax2, line_set.names, frac * 100, colors=[plotting.feature_color(n) for n in line_set.names])
ax2.set_ylabel("% of galaxies with the line detected"); ax2.set_title("Detection fraction per feature")
ax2.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.show()'''),
    md('''
## 3. Confusion matrix and redshift assignment

For `N_det ≥ 2` the redshift is taken as correct (`z_assigned = z_rsd`). For `N_det = 1` an identity
is drawn from `C_ij = P(assigned j | true i)`, with an explicit fail column, and
`z_assigned = λ_i (1 + z_rsd) / λ_j − 1`. The matrix below is an assumed scenario, not a calibration.
'''),
    code('''
from romanmock import ConfusionMatrix, assign_redshifts, IdOutcome

from romanmock.confusion import DEFAULT_CONFUSION_ROWS

print(DEFAULT_CONFUSION_ROWS)          # the package default: e.g. a true Ha line is called [O III] with probability 0.5 %
cm = ConfusionMatrix.from_dict(line_set, DEFAULT_CONFUSION_ROWS, default_p_fail=0.0)
print(cm)
za = assign_redshifts(det, z_rsd, line_set, cm, rng)
for o in IdOutcome:
    print(f"{o.value} {o.name:<15s} {int((za.outcome == o).sum()):>9,}")'''),
    code('''
sel = rng.random(len(z_rsd)) < 0.1     # thin for plotting
fig, ax = plt.subplots(figsize=(7.5, 5))
for name, size in [("SINGLE_CORRECT", 6), ("MULTI_LINE", 2), ("SINGLE_MISID", 4)]:
    m = sel & (za.outcome == IdOutcome[name])
    ax.scatter(z_rsd[m], za.z_assigned[m], s=size, color=plotting.OUTCOME_COLORS[name], alpha=0.6, lw=0,
               label=f"{name} ({int((za.outcome == IdOutcome[name]).sum()):,})", rasterized=True)
ax.plot([0, 3], [0, 3], color=plotting.TEXT_SECONDARY, lw=1, ls="--")
ax.set_xlabel("redshift_rsd (true, with peculiar velocity)"); ax.set_ylabel("redshift_assigned")
ax.set_title("Assigned versus true redshift (multi-line and single-correct both sit on the diagonal)")
ax.legend(markerscale=3, fontsize=8); plt.show()'''),
    md('''
The off-diagonal streaks are the interloper tracks, e.g. [O III] 5007 at z ≈ 2.3 assigned as Hα
lands at an assigned redshift (1+2.3)·5008/6565 − 1 ≈ 1.5.

The sampling reproduces the input matrix: compare it with the empirical confusion among
single-line galaxies.
'''),
    code('''
from romanmock.diagnostics import empirical_confusion

emp = empirical_confusion({"n_det": det.n_det, "line_true": za.line_true, "line_assigned": za.line_assigned}, line_set)
names = line_set.names + ["fail"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, mat, title in zip(axes, [cm.matrix, emp], ["input confusion matrix", "empirical (single-line galaxies)"]):
    im = ax.imshow(mat, cmap=plotting.sequential_cmap(), vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8); ax.set_yticks(range(len(line_set)), line_set.names, fontsize=8)
    ax.set_xlabel("assigned"); ax.set_ylabel("true"); ax.set_title(title); ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0.005:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if mat[i, j] > 0.5 else plotting.TEXT_PRIMARY)
plt.tight_layout(); plt.show()
print("rows with no single-line galaxies in this subsample are empty in the empirical matrix")'''),
    md('''
## The same thing through the pipeline object

`RomanMockPipeline.process_arrays` runs exactly these steps from a configuration and returns the
output columns of the catalog.
'''),
    code('''
from romanmock import RomanMockConfig, RomanMockPipeline
from romanmock.roman import DEFAULT_LINE_FLUX_LIMIT as FLUX_LIMIT, DEFAULT_SECONDARY_LINE_FLUX_LIMIT as SECONDARY_LIMIT

cfg = RomanMockConfig.from_dict({
    "line_set": "roman_transitions",
    "sensitivity": {"type": "constant", "flux_limit": FLUX_LIMIT, "secondary_flux_limit": SECONDARY_LIMIT,
                    "wave_range": [10000.0, 19300.0]},
    "confusion": {"type": "matrix", "default_p_fail": 0.0, "rows": cm.to_dict()["rows"]},   # same as the package default
    "output": {"store_snr": True},
})
pipe = RomanMockPipeline(cfg)
out = pipe.process_arrays(data, cosmology, np.random.default_rng(0))
print(list(out))
print("identical bitmasks:", np.array_equal(out["line_bitmask"], det.bitmask))'''),
]

# =====================================================================================
# 03 - running on slices and building samples
# =====================================================================================
NB03 = [
    md('''
# 03 — Running the pipeline on lightcone slices and building analysis samples

Stage A (`RomanMockPipeline.run`) processes whole slices and writes OpenCosmo files. Stage B
(`build_samples`) applies footprint, redshift, tracer and quality cuts and can downsample to a
target dN/dz. Outputs go to `WORK` (your `$PSCRATCH`). Three slices are processed here (≈ 1–2 min);
for the whole mock use the CLI with a job array (`scripts/run_slices.sh`).
'''),
    code(SETUP),
    md('''
## Stage A: configuration and run

The configuration is a dataclass that round-trips to YAML (`cfg.to_yaml()`), so the same
object drives the CLI. Each slice gets a random stream seeded by `(seed, step)`.
'''),
    code('''
from romanmock import RomanMockConfig, RomanMockPipeline
from romanmock.roman import DEFAULT_LINE_FLUX_LIMIT as FLUX_LIMIT, DEFAULT_SECONDARY_LINE_FLUX_LIMIT as SECONDARY_LIMIT

# one output directory per pair of limits, so runs with different thresholds never mix
PARENT_DIR = os.path.join(WORK, f"parent_F{FLUX_LIMIT:.1e}_S{SECONDARY_LIMIT:.1e}")
cfg = RomanMockConfig.from_dict({
    "catalog": {"path": CATALOG, "steps": [148, 219, 224]},   # the intended parent catalog: three slices
    "output": {"dir": PARENT_DIR, "overwrite": False, "store_fluxes": True,   # fluxes are optional; kept here to show them
               "passthrough_columns": ["gal_id", "ra", "dec", "redshift_true", "logsm_obs", "roman_F158"]},
    "line_set": "roman_transitions",
    "sensitivity": {"type": "constant", "flux_limit": FLUX_LIMIT, "secondary_flux_limit": SECONDARY_LIMIT,
                    "wave_range": [10000.0, 19300.0]},
    # confusion: omitted -> the package default scenario (romanmock.config.DEFAULT_CONFUSION_ROWS)
    "seed": 2026,
})
pipe = RomanMockPipeline(cfg)
outputs = pipe.run()      # slices already completed with this configuration are reused (see the manifest)
outputs'''),
    md('''
Every run writes a manifest of completed slices with a hash of the scientific configuration
(line set, sensitivity, detection, confusion, seed, ...). A rerun with the same configuration
reuses finished slices; a changed configuration refuses to mix with existing outputs unless
`output.overwrite` is set, and stage B only accepts a parent catalog whose manifest matches the
configuration and covers every selected slice.
'''),
    code('''
print(open(os.path.join(cfg.output.dir, "romanmock_manifest.yaml")).read())
print(open(os.path.join(cfg.output.dir, "romanmock_metadata.yaml")).read()[:600])'''),
    md('''
## The parent emission-line catalog

The per-slice outputs re-open as one `opencosmo.Lightcone`; header, cosmology and HEALPix index
are preserved, so all OpenCosmo queries (`filter`, `cone_search`, `with_redshift_range`, ...) work.
'''),
    code('''
parent = pipe.open_parent()
print(len(parent), "galaxies, z_range", parent.z_range)
print(parent.columns)
print(parent.units["Ha_flux"], "|", parent.descriptions["id_outcome"])'''),
    code('''
from romanmock.diagnostics import outcome_summary

cols = ["redshift_assigned", "redshift_rsd", "id_outcome", "n_det", "line_true", "line_assigned", "ra", "dec"]
pdata = parent.select(*cols).get_data("numpy", wrap_single=True)
summary = outcome_summary(pdata)
fig, ax = plt.subplots(figsize=(9, 4.2))
plotting.bar(ax, [k.replace("_", "\\n") for k in summary], list(summary.values()), colors=[plotting.OUTCOME_COLORS[k] for k in summary])
ax.set_ylabel("galaxies"); ax.set_title("Identification outcomes in the parent catalog"); plt.show()'''),
    md('''
## dN/dz of the assigned-Hα sample and its contamination

An observer's Hα sample is every galaxy *assigned* Hα (`line_assigned == bit("Ha")`) with a
redshift. The contamination fraction `f_{i→j}(B)` uses that sample as denominator
(plan comment 4): here [O III] 5007 emitters at z ≈ 2.3 misidentified as Hα appear at assigned redshift 1.5.

Only three slices were processed, so the bins around assigned redshift 1.5 contain interlopers and no
true Hα emitters (there is no slice at z ≈ 1.5 in this demo), which is why the interloper fraction
reaches 100 % there. In a full run those bins are dominated by genuine Hα galaxies and the
fraction is a few per cent, set by the confusion matrix and the relative abundances.
'''),
    code('''
from romanmock.diagnostics import contamination_fraction, completeness, dndz
from romanmock.selection import sky_area_deg2

area = sky_area_deg2(parent, nside=256)
z_edges = np.linspace(1.1, 1.7, 31)
cont = contamination_fraction(pdata, pipe.line_set, z_edges, assigned_line="Ha")
zc = 0.5 * (z_edges[1:] + z_edges[:-1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ax1.step(zc, cont["n_sample"] / (np.diff(z_edges) * area), where="mid", color=plotting.feature_color("Ha"))
ax1.set_xlabel("redshift_assigned"); ax1.set_ylabel("dN/dz per deg²"); ax1.set_title(f"Assigned-Hα sample, dN/dz (area {area:.0f} deg²)")
for name in ["OIII_5007", "OII_3727"]:
    i = pipe.line_set.bit(name)
    ax2.step(zc, cont["fraction"][i] * 100, where="mid", color=plotting.feature_color(name), label=f"true {name}")
ax2.set_xlabel("redshift_assigned"); ax2.set_ylabel("% of the assigned-Hα sample"); ax2.set_title("Interloper fraction f(i→Hα)"); ax2.legend()
plt.tight_layout(); plt.show()'''),
    code('''
comp = completeness(pdata, pipe.line_set, np.linspace(1.15, 1.26, 12), line="Ha")
print("Hα completeness per true-z bin:", np.round(comp["completeness"], 3))'''),
    md('''
## Stage B: analysis samples

A `SampleConfig` defines cuts; `downsample` thins the sample to a target dN/dz with
`p_keep = target / mock` (never above 1). The target here is a smooth curve at 40% of the mock
density, just to demonstrate the mechanics; the mock dN/dz used in the ratio is estimated inside
the footprint and smoothed over one bin.
'''),
    code('''
from romanmock.config import SampleConfig, DownsampleConfig
from romanmock.selection import build_footprint

box = {"type": "box", "ra_min": 0.0, "ra_max": 45.0, "dec_min": 55.0, "dec_max": 66.0}
footprint = build_footprint(box)
fp_area = sky_area_deg2(parent, 256, footprint=footprint)
sel = (pdata["line_assigned"] == pipe.line_set.bit("Ha")) & np.isfinite(pdata["redshift_assigned"]) & footprint(pdata["ra"], pdata["dec"])
edges = np.linspace(1.15, 1.26, 12)
mock = dndz(pdata["redshift_assigned"][sel], edges, fp_area)
zc = 0.5 * (edges[1:] + edges[:-1])
target = {"z": zc.tolist(), "dndz": (0.4 * np.convolve(mock, np.ones(3) / 3, mode="same")).tolist()}

samples = [
    SampleConfig(name="halpha_full", z_min=1.15, z_max=1.26, assigned_lines=["Ha"]),
    SampleConfig(name="halpha_box_downsampled", z_min=1.15, z_max=1.26, assigned_lines=["Ha"], footprint=box,
                 downsample=DownsampleConfig(target=target, z_edges=edges.tolist(), smooth_bins=1.0)),
]
cfg.output.overwrite = True
sample_files = pipe.build_samples(samples=samples)      # one directory per sample, one file per slice inside
sample_files'''),
    code('''
from romanmock.io import open_sample

ds_sample = open_sample(sample_files[1])                # the slice files of a sample, as one lightcone
zs = ds_sample.select("redshift_assigned").get_data("numpy")
fig, ax = plt.subplots()
ax.step(zc, mock, where="mid", color=plotting.CATEGORICAL[0], label="mock, assigned Hα in box")
ax.plot(zc, target["dndz"], color=plotting.CATEGORICAL[1], label="target")
ax.step(zc, dndz(zs, edges, fp_area), where="mid", color=plotting.CATEGORICAL[2], label="downsampled sample")
ax.set_xlabel("redshift_assigned"); ax.set_ylabel("dN/dz per deg²"); ax.set_title(f"Downsampling to a target dN/dz (footprint {fp_area:.0f} deg²)")
ax.legend(); plt.show()
print(open(os.path.join(PARENT_DIR, "samples", "halpha_box_downsampled_metadata.yaml")).read()[-400:])'''),
    md('''
The sample file keeps `ra`, `dec`, `redshift_true`, `redshift_rsd`, `redshift_assigned` and `id_outcome`
(configurable with `output.sample_columns`); the parent catalog also has `line_bitmask`, `line_true`, `line_assigned`,
so completeness, contamination and clustering can be checked against the truth, as the plan asks.
'''),
]

# =====================================================================================
# 04 - extending the models
# =====================================================================================
NB04 = [
    md('''
# 04 — Extending the models

The plan's comments ask for room to refine each stage. Every stage is a small protocol with a
registry, so a new model is a class plus one registration and is then selectable from YAML.
This notebook shows: a wavelength-dependent sensitivity, a noisy detection model, a confusion
model that depends on signal-to-noise, and a custom line set.
'''),
    code(SETUP),
    code('''
from romanmock.lines import get_line_set
from romanmock import feature_fluxes, detect_lines, ConstantFluxLimit, TabulatedFluxLimit, ThresholdDetection, GaussianNoiseDetection
from romanmock.io import open_catalog

line_set = get_line_set("roman_transitions")
cat = open_catalog([f"{CATALOG}/lc_cores-219.diffsky_gals.hdf5"])
data = cat.take(200_000, at="start").select(*line_set.required_columns, "redshift_true", "redshift_obs").get_data("numpy", wrap_single=True)
flux = feature_fluxes(data, line_set, data["redshift_true"], cat.cosmology)
z_rsd = data["redshift_obs"]'''),
    md('''
## Wavelength-dependent sensitivity

`TabulatedFluxLimit` interpolates a flux-limit curve (log-linear) and is infinite outside its
range. Any object with `wave_range` and `flux_limit(wave_obs)` works; register it in
`romanmock.sensitivity.SENSITIVITY_MODELS` to select it by name from YAML (`type: tabulated`,
with `file:` for a two-column text file). A model that also defines `secondary_flux_limit(wave_obs)`
gets the two-tier rule (lower limit for the other lines of a galaxy with one line above `flux_limit`);
`TabulatedFluxLimit` takes it as a ratio, `secondary_ratio=0.4` being the equivalent of 4e-17 at 1e-16.
'''),
    code('''
wave = np.linspace(10000, 19300, 50)
curve = 1e-16 * (1 + 2.0 * ((wave - 14650) / 4650) ** 2)      # toy: worse sensitivity towards the band edges
tab = TabulatedFluxLimit(wave, curve, secondary_ratio=0.4)      # same two tiers as the constant default (1e-16, 4e-17)

fig, ax = plt.subplots()
w = np.linspace(9500, 20000, 300)
ax.plot(w / 1e4, ConstantFluxLimit(1e-16).flux_limit(w) * 1e16, color=plotting.CATEGORICAL[0], label="constant")
ax.plot(w / 1e4, tab.flux_limit(w) * 1e16, color=plotting.CATEGORICAL[1], label="tabulated (toy)")
ax.set_ylim(0, 4); ax.set_xlabel("observed wavelength [μm]"); ax.set_ylabel("flux limit [1e-16 erg/s/cm²]")
ax.set_title("Sensitivity models"); ax.legend(); plt.show()

for name, model in [("constant", ConstantFluxLimit(1e-16)), ("tabulated", tab)]:
    det = detect_lines(flux, z_rsd, line_set, ThresholdDetection(model))
    print(f"{name:<10s} detected fraction (any line): {np.mean(det.n_det > 0):.3f}   multi-line among detected: "
          f"{np.mean(det.n_det[det.n_det > 0] > 1):.3f}")'''),
    md('''
## Detection scatter

`GaussianNoiseDetection` adds flux noise `σ = F_lim / snr_at_limit` and detects at
`S/N > snr_threshold`; the detection probability becomes a smooth function of `F / F_lim`.
'''),
    code('''
rng = np.random.default_rng(1)
ratio = np.logspace(-0.7, 0.7, 40)
n = 4000
z = np.full(n, 1.2)
probs = {}
for label, model in [("threshold", ThresholdDetection(ConstantFluxLimit(1e-16))),
                     ("gaussian noise, S/N>6.5", GaussianNoiseDetection(ConstantFluxLimit(1e-16), snr_at_limit=6.5, snr_threshold=6.5)),
                     ("gaussian noise, S/N>5", GaussianNoiseDetection(ConstantFluxLimit(1e-16), snr_at_limit=6.5, snr_threshold=5.0))]:
    p = []
    for r in ratio:
        f = np.zeros((n, len(line_set))); f[:, 0] = r * 1e-16
        p.append(detect_lines(f, z, line_set, model, rng).detected[:, 0].mean())
    probs[label] = np.array(p)

fig, ax = plt.subplots()
for (label, p), color in zip(probs.items(), plotting.CATEGORICAL):
    ax.plot(ratio, p, color=color, label=label)
ax.set_xscale("log"); ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"]); ax.minorticks_off()
ax.set_xlabel("F / F_lim"); ax.set_ylabel("detection probability")
ax.set_title("Detection models"); ax.legend(); plt.show()'''),
    md('''
## A confusion model that depends on signal-to-noise

`assign(true_index, context, rng)` receives `wave_obs`, `snr`, `flux`, `z_rsd` and any
`redshift.context_columns` from the catalog (photometry, ...). Below, the failure probability
falls with S/N and the remainder is distributed like a base matrix. Registering the class in
`CONFUSION_MODELS` makes `confusion: {type: snr_dependent, ...}` valid in YAML.
'''),
    code('''
from romanmock.confusion import ConfusionMatrix, CONFUSION_MODELS, FAIL, build_confusion
from romanmock import assign_redshifts, IdOutcome


class SNRDependentConfusion:
    """p_fail(S/N) = p0 * (snr_ref / snr)^k, applied on top of a base matrix without fail."""

    def __init__(self, line_set, base_rows, p0=0.5, snr_ref=6.5, k=2.0):
        self.base = ConfusionMatrix.from_dict(line_set, base_rows)   # base matrix, rows sum to 1
        self.names = self.base.names
        self.p0, self.snr_ref, self.k = p0, snr_ref, k

    def p_fail(self, snr):
        return np.clip(self.p0 * (self.snr_ref / np.maximum(snr, 1e-3)) ** self.k, 0, 1)

    def assign(self, true_index, context, rng):
        j = self.base.assign(true_index, context, rng)
        fail = rng.random(len(j)) < self.p_fail(np.asarray(context["snr"]))
        return np.where(fail, FAIL, j)


CONFUSION_MODELS["snr_dependent"] = SNRDependentConfusion
model = build_confusion({"type": "snr_dependent", "base_rows": {"Ha": {"OIII_5007": 0.05}}, "p0": 0.4}, line_set)

det = detect_lines(flux, z_rsd, line_set, ThresholdDetection(ConstantFluxLimit()), rng)   # package default limit
za = assign_redshifts(det, z_rsd, line_set, model, rng)
single = det.n_det == 1
snr = det.snr[np.arange(len(z_rsd)), np.maximum(za.line_true, 0)][single]
failed = (za.outcome == IdOutcome.SINGLE_FAILED)[single]
bins = np.logspace(np.log10(6.5), 2, 15)
idx = np.digitize(snr, bins)
frac = np.array([failed[idx == b].mean() if np.any(idx == b) else np.nan for b in range(1, len(bins))])
fig, ax = plt.subplots()
xc = np.sqrt(bins[1:] * bins[:-1])
ax.plot(xc, model.p_fail(xc), color=plotting.CATEGORICAL[1], label="model p_fail(S/N)")
ax.plot(xc, frac, "o", color=plotting.CATEGORICAL[0], label="realised fail fraction")
ax.set_xscale("log"); ax.set_xlabel("S/N of the single detected line"); ax.set_ylabel("fraction with no reliable z")
ax.set_title("S/N-dependent failure"); ax.legend(); plt.show()'''),
    md('''
## Custom line sets

Bit ordering is whatever the `LineSet` says. A custom set can be given as a list of dicts (also in
YAML under `line_set:`); blends are sums of columns. Existing presets should never be reordered.
'''),
    code('''
from romanmock.lines import LineSet, LineFeature

custom = LineSet.from_config([
    {"name": "Ha",   "components": ["Ba_alpha_6563"]},                                    # rest_wave from the built-in table
    {"name": "OIII", "components": ["O_III_4959", "O_III_5007"], "rest_wave": 5008.314},
    {"name": "OII",  "components": ["O_II_3726", "O_II_3729"], "rest_wave": 3728.6},
], name="three_features")
print(custom.describe())

from romanmock import RomanMockConfig
cfg = RomanMockConfig.from_dict({"line_set": custom.to_config(), "confusion": {"type": "identity", "p_fail": 0.05}})
print(cfg.to_yaml()[:600])'''),
]

# =====================================================================================
# 05 - full run over every slice (exclusive CPU node)
# =====================================================================================
NB05 = [
    md("""
# 05 — Full run over the whole lightcone

This notebook processes all 67 slices (324 M galaxies, z = 0–4) and then looks at the complete
Roman-like parent catalog. It is meant for an exclusive CPU node (Perlmutter: 128 physical
cores, 512 GB): slices are processed in parallel, one per worker process, with a peak of a few GB
per worker. Wall time is dominated by reading the 340 GB of input from CFS.

The same run from the command line is
`romanmock run <config.yaml> --workers 32` (the notebook writes that config file).
"""),
    code(SETUP),
    code("""
import time
from romanmock import RomanMockConfig, RomanMockPipeline
from romanmock.roman import DEFAULT_LINE_FLUX_LIMIT as FLUX_LIMIT, DEFAULT_SECONDARY_LINE_FLUX_LIMIT as SECONDARY_LIMIT

from romanmock.io import memory_budget_gb, suggest_n_workers

# one slice per process, ~4.5 GB peak each: size the pool by the memory this session may use
# (node memory, or the per-user cgroup cap on a NERSC login node)
N_WORKERS = suggest_n_workers(max_workers=32)
print(f"memory budget {memory_budget_gb():.0f} GB -> {N_WORKERS} worker(s)")
OUT_DIR = os.path.join(WORK, f"full_F{FLUX_LIMIT:.1e}_S{SECONDARY_LIMIT:.1e}")   # one directory per pair of limits

cfg = RomanMockConfig.from_dict({
    "catalog": {"path": CATALOG},                        # every slice, z = 0-4
    "output": {"dir": OUT_DIR, "overwrite": False,
               "passthrough_columns": ["ra", "dec", "redshift_true"]},          # plus redshift_rsd, redshift_assigned, id_outcome, identification columns
    "line_set": "roman_transitions",
    # primary limit for the first line of a galaxy, lower secondary limit for its other lines
    "sensitivity": {"type": "constant", "flux_limit": FLUX_LIMIT, "secondary_flux_limit": SECONDARY_LIMIT,
                    "wave_range": [10000.0, 19300.0]},
    "detection": {"type": "threshold"},
    # confusion: omitted -> the package default scenario (romanmock.config.DEFAULT_CONFUSION_ROWS):
    #   true Ha called [O III] 5007 with p = 0.005; true [O III] 5007, [S III] 9532 called Ha with p = 0.05;
    #   true Hb called [O III] 5007 with p = 0.05; no failures. An assumed scenario, not a calibration.
    "seed": 2026,
})
os.makedirs(OUT_DIR, exist_ok=True)
cfg.to_yaml(os.path.join(OUT_DIR, "config.yaml"))     # reusable with: romanmock run .../config.yaml --workers 32
pipe = RomanMockPipeline(cfg)
print(pipe.line_set.describe())
print(pipe.confusion)"""),
    code("""
t0 = time.time()
outputs = pipe.run(n_workers=N_WORKERS)               # existing outputs are skipped (output.overwrite = False)
print(f"{len(outputs)} slices, {(time.time() - t0) / 60:.1f} min")"""),
    md("""
## The parent catalog

All slices re-open as one `opencosmo.Lightcone`. Only the columns needed below are read.
"""),
    code("""
from romanmock.diagnostics import outcome_summary
from romanmock.selection import sky_area_deg2

parent = pipe.open_parent()
area = sky_area_deg2(parent, nside=256)
print(f"{len(parent):,} galaxies with at least one detected line, z_range {parent.z_range}, area {area:.0f} deg²")
cols = ["redshift_assigned", "redshift_rsd", "id_outcome", "n_det", "line_true", "line_assigned"]
t0 = time.time()
pdata = parent.select(*cols).get_data("numpy", wrap_single=True)
print(f"read {len(cols)} columns in {time.time() - t0:.0f} s")
summary = outcome_summary(pdata)
for k, v in summary.items():
    print(f"  {k:<15s} {v:>12,}  ({100 * v / len(parent):5.2f}%)")"""),
    code("""
fig, ax = plt.subplots(figsize=(9, 4.2))
plotting.bar(ax, [k.replace("_", "\\n") for k in summary], list(summary.values()), colors=[plotting.OUTCOME_COLORS[k] for k in summary])
ax.set_ylabel("galaxies"); ax.set_title("Identification outcomes, full lightcone"); plt.show()"""),
    md("""
## Redshift distributions per assigned line

`dN/dz` per deg² of assigned redshift of the galaxies *assigned* each identity (what an observer would call the
Hα sample, the [O III] sample, ...). Multi-line galaxies are counted under their brightest line.
"""),
    code("""
from romanmock.diagnostics import dndz

z_edges = np.linspace(0.0, 4.0, 161)
zc = 0.5 * (z_edges[1:] + z_edges[:-1])
has_z = np.isfinite(pdata["redshift_assigned"])
fig, ax = plt.subplots(figsize=(9, 4.6))
for name in ["Ha", "OIII_5007", "OII_3727", "Hb", "OIII_4959", "SIII_9532"]:
    sel = has_z & (pdata["line_assigned"] == pipe.line_set.bit(name))
    if sel.sum() == 0:
        continue
    y = dndz(pdata["redshift_assigned"][sel], z_edges, area)
    ax.step(zc, y, where="mid", color=plotting.feature_color(name), label=f"{name} ({sel.sum():,})")
ax.set_yscale("log"); ax.set_ylim(1, None)
ax.set_xlabel("redshift_assigned"); ax.set_ylabel("dN/dz per deg²"); ax.set_title(f"Assigned-line samples, F > {FLUX_LIMIT:.1e} (other lines > {SECONDARY_LIMIT:.1e}) erg/s/cm², 1.00–1.93 μm")
ax.legend(fontsize=8); plt.show()"""),
    md("""
## True versus assigned redshift

The density of galaxies in the (redshift_rsd, redshift_assigned) plane. Correct identifications lie on the
diagonal; each interloper track is the line ``(1+z_assigned) = (1+z_rsd) λ_i/λ_j``.
"""),
    code("""
sel = has_z
H, xe, ye = np.histogram2d(pdata["redshift_rsd"][sel], pdata["redshift_assigned"][sel], bins=[np.linspace(0, 4, 201), np.linspace(0, 4, 201)])
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.pcolormesh(xe, ye, np.log10(np.where(H > 0, H, np.nan)).T, cmap=plotting.sequential_cmap())
plt.colorbar(im, ax=ax, label="log10 galaxies per cell")
ax.plot([0, 4], [0, 4], color=plotting.TEXT_SECONDARY, lw=0.8, ls="--")
ax.set_xlabel("redshift_rsd (true)"); ax.set_ylabel("redshift_assigned"); ax.set_title("Assigned versus true redshift, all galaxies with a redshift")
ax.grid(False); plt.show()"""),
    md("""
## Contamination of the Hα and [O III] samples

`f_{i→j}(B)`: the fraction of the assigned-*j* sample in a bin of assigned redshift whose true line is *i*
(plan comment 4). It depends on the confusion matrix *and* on the relative abundances of the
populations along the line of sight, so it is not constant even though the matrix is.
"""),
    code("""
from romanmock.diagnostics import contamination_fraction

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (assigned, zr) in zip(axes, [("Ha", (0.5, 2.0)), ("OIII_5007", (1.0, 2.9))]):
    edges = np.linspace(*zr, 31)
    c = contamination_fraction(pdata, pipe.line_set, edges, assigned_line=assigned)
    zc = 0.5 * (edges[1:] + edges[:-1])
    for name in pipe.line_set.names:
        i = pipe.line_set.bit(name)
        if name == assigned or c["fraction"][i].max() < 0.002:
            continue
        ax.step(zc, 100 * c["fraction"][i], where="mid", color=plotting.feature_color(name), label=f"true {name}")
    ax.set_xlabel("redshift_assigned"); ax.set_ylabel(f"% of the assigned-{assigned} sample"); ax.set_title(f"Interlopers in the {assigned} sample")
    ax.legend(fontsize=8)
plt.tight_layout(); plt.show()"""),
    md("""
Two features of these curves are properties of the mock rather than of the confusion matrix.
Below assigned redshift 1.9 the assigned-[O III] sample is *entirely* Hα interlopers: any galaxy whose
[O III] 5007 is detected there also has Hα (always brighter in diffsky) inside the bandpass, so it
is a multi-line detection and is counted under Hα; the only single-line "[O III]" objects are
misidentified Hα emitters. Likewise the [O III] → Hα contamination of the Hα sample switches on
at assigned redshift 1.2, where true [O III] emitters at z ≈ 1.9 (Hα just redshifted out of the grism)
start to be single-line detections.
"""),
    md("""
## Completeness and the empirical confusion matrix
"""),
    code("""
from romanmock.diagnostics import completeness, empirical_confusion

fig, ax = plt.subplots()
for name, zr in [("Ha", (0.5, 2.0)), ("OIII_5007", (1.0, 2.9)), ("OII_3727", (1.7, 4.0))]:
    edges = np.linspace(*zr, 31)
    comp = completeness(pdata, pipe.line_set, edges, line=name)
    ax.step(0.5 * (edges[1:] + edges[:-1]), comp["completeness"], where="mid", color=plotting.feature_color(name), label=name)
ax.set_ylim(0, 1.05); ax.set_xlabel("true redshift"); ax.set_ylabel("fraction with a correct redshift")
ax.set_title("Completeness of detected emitters (brightest line = feature)"); ax.legend(); plt.show()

emp = empirical_confusion(pdata, pipe.line_set)
names = pipe.line_set.names + ["fail"]
print("empirical confusion among single-line galaxies (rows true, columns assigned):")
print("            " + " ".join(f"{n[:9]:>9s}" for n in names))
for i, n in enumerate(pipe.line_set.names):
    print(f"  {n[:9]:<9s} " + " ".join(f"{x:9.4f}" for x in emp[i]))"""),
    md("""
## Matching the emitter populations to the HLSS forecast

Wang et al. (2022, arXiv:2110.01829) tabulate the expected dN/dz per deg² of Hα and [O III]
emitters above a given line-flux limit (Tables 1–2, Galacticus on the UNIT lightcone). Those
numbers describe *true emitters before observational efficiency*, so the right quantity to match
is the mock's detected-line population in bins of true redshift, before any redshift assignment
or sample selection. `romanmock.reference_dndz` carries the tables; the comparison below uses the
run's primary flux limit (the package default, 1.0e-16) and the A_V = 1.65 dust model. Populations are
defined by the brightest detected line, which is always above the primary limit, so the lower secondary
limit (4.0e-17, for the other lines of a detected galaxy) does not enter this comparison.
"""),
    code("""
from romanmock.reference_dndz import wang2022_bins

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, name in zip(axes, ["Ha", "OIII_5007"]):
    edges, ref = wang2022_bins(name, FLUX_LIMIT, 1.65)
    _, ref192 = wang2022_bins(name, FLUX_LIMIT, 1.92)
    sel = pdata["line_true"] == pipe.line_set.bit(name)
    zc = 0.5 * (edges[1:] + edges[:-1])
    ax.step(zc, dndz(pdata["redshift_rsd"][sel], edges, area), where="mid", color=plotting.feature_color(name), label=f"diffsky mock, {name} brightest detected line")
    ax.step(zc, ref, where="mid", color=plotting.TEXT_PRIMARY, label="Wang+22 A_V=1.65")
    ax.step(zc, ref192, where="mid", color=plotting.TEXT_SECONDARY, ls="--", label="Wang+22 A_V=1.92")
    ax.set_xlabel("true redshift"); ax.set_ylabel("dN/dz per deg²"); ax.set_title(f"{name} emitters above {FLUX_LIMIT:.1e} erg/s/cm²"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()"""),
    md("""
The diffsky mock contains several times more Hα emitters than the Galacticus forecast at z > 1
(and fewer at z < 0.8, where thinning cannot help and the pipeline warns). Population thinning
draws `p_keep(z_true) = target / mock` per population and applies it before the analysis samples
are built, so interlopers in a sample are thinned consistently with the population they really
belong to. It is switched on with `population_thinning:` in the configuration.
"""),
    code("""
from romanmock.config import PopulationThinningConfig, SampleConfig

cfg.population_thinning = PopulationThinningConfig(
    targets={"Ha": {"reference": "wang2022", "dust_av": 1.65}, "OIII_5007": {"reference": "wang2022", "dust_av": 1.65}},
    by="line_true", z_column="redshift_rsd", outside_range="keep")
samples = [
    SampleConfig(name="halpha_z1p0_1p9", z_min=1.0, z_max=1.9, assigned_lines=["Ha"]),
    SampleConfig(name="oiii_z2p0_2p9", z_min=2.0, z_max=2.9, assigned_lines=["OIII_5007"]),
]
cfg.output.overwrite = True
print("default sample columns:", cfg.output.sample_columns)
cfg.output.sample_columns = None    # keep every parent column here: the composition plots below need line_true
t0 = time.time()
sample_files = pipe.build_samples(samples=samples, n_workers=min(N_WORKERS, 16))   # slice by slice, ~1.7 GB per worker
print(f"{(time.time() - t0) / 60:.1f} min")
sample_files"""),
    code("""
from romanmock.io import open_sample

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, path, name in zip(axes, sample_files, ["Ha", "OIII_5007"]):
    d = open_sample(path).select("redshift_assigned", "redshift_rsd", "id_outcome", "line_true").get_data("numpy", wrap_single=True)
    edges, ref = wang2022_bins(name, FLUX_LIMIT, 1.65)
    zc = 0.5 * (edges[1:] + edges[:-1])
    sel = pdata["line_assigned"] == pipe.line_set.bit(name)
    ax.step(zc, dndz(pdata["redshift_assigned"][sel], edges, area), where="mid", color=plotting.CATEGORICAL[0], label="parent, assigned line (assigned z)")
    ax.step(zc, ref, where="mid", color=plotting.TEXT_PRIMARY, label="Wang+22 target (true emitters)")
    ax.step(zc, dndz(d["redshift_assigned"], edges, area), where="mid", color=plotting.CATEGORICAL[2], label="sample after thinning (assigned z)")
    misid = np.mean(d["id_outcome"] == 3)
    ax.set_xlabel("redshift"); ax.set_ylabel("dN/dz per deg²")
    ax.set_title(f"{name} sample: {len(d['redshift_assigned']):,} galaxies, {100 * misid:.1f}% interlopers"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
print(open(os.path.join(OUT_DIR, "samples", "halpha_z1p0_1p9_metadata.yaml")).read()[-700:])"""),
    md("""
The sample dN/dz is close to, but not identical with, the target: the sample is defined on
*assigned* redshift and contains interlopers (thinned with their own population's `p_keep`)
while missing the emitters that failed or were misidentified. That difference is the point of
keeping the two steps separate (plan comment 5). For matching a forecast of the *final* sample
instead, use the per-sample `downsample` option on `redshift_assigned`.
"""),
    md("""
## Mock versus forecast, and confusion versus contamination

Two quantitative points are worth making explicit. First, how far the diffsky abundances are
from the Galacticus forecast: the ratio of the mock's detected-emitter dN/dz to the Wang et al.
table per bin. Ratios above one are what thinning removes; ratios below one are shortages that
thinning cannot fill (the pipeline warns about them).
"""),
    code("""
fig, ax = plt.subplots(figsize=(8, 4.2))
for name in ["Ha", "OIII_5007"]:
    edges, ref = wang2022_bins(name, FLUX_LIMIT, 1.65)
    zc = 0.5 * (edges[1:] + edges[:-1])
    sel = pdata["line_true"] == pipe.line_set.bit(name)
    ratio = dndz(pdata["redshift_rsd"][sel], edges, area) / ref
    ax.step(zc, ratio, where="mid", color=plotting.feature_color(name), label=f"{name}: mock / Wang+22 (A_V=1.65)")
    for z, r in zip(zc, ratio):
        if 0 < r < 100:
            ax.text(z, r, f"{r:.1f}", fontsize=7, ha="center", va="bottom", color=plotting.TEXT_SECONDARY)
ax.axhline(1.0, color=plotting.TEXT_SECONDARY, lw=1, ls="--")
ax.set_yscale("log"); ax.set_ylim(0.1, 20)
ax.set_xlabel("true redshift"); ax.set_ylabel("mock dN/dz / forecast dN/dz")
ax.set_title(f"Detected emitters in the mock relative to the HLSS forecast (F > {FLUX_LIMIT:.1e})"); ax.legend(fontsize=8)
plt.show()"""),
    md("""
Second, why the [O III] sample is far more contaminated than the confusion matrix alone suggests.
The matrix says a single-line Hα emitter is called [O III] with probability 5 %. But the sample of
galaxies *assigned* [O III] at assigned redshift 2–2.9 draws its interlopers from single-line Hα emitters at
z_rsd ≈ 1.3–1.95 (the redshifts that map onto 2–2.9 through λ_Hα/λ_[O III]), and those are many
times more numerous than the true [O III] emitters at z ≈ 2–2.9, even after thinning. The
contamination fraction is the abundance-weighted quantity

f(Hα → [O III]) = N_Hα,single(z_src) · C(Hα→[O III]) / N_sample([O III]),

which is plan comment 4 and review item 4: a fixed confusion matrix does not imply a fixed
interloper fraction.
"""),
    code("""
from romanmock import IdOutcome

i_ha, i_o3 = pipe.line_set.bit("Ha"), pipe.line_set.bit("OIII_5007")
d = open_sample(sample_files[1]).select("redshift_assigned", "redshift_rsd", "line_true", "id_outcome").get_data("numpy", wrap_single=True)
edges = np.linspace(2.0, 2.9, 10)
zc = 0.5 * (edges[1:] + edges[:-1])
true_o3 = d["line_true"] == i_o3
ha_interloper = (d["line_true"] == i_ha) & (d["id_outcome"] == IdOutcome.SINGLE_MISID)
n_true, _ = np.histogram(d["redshift_assigned"][true_o3], bins=edges)
n_int, _ = np.histogram(d["redshift_assigned"][ha_interloper], bins=edges)
n_all, _ = np.histogram(d["redshift_assigned"], bins=edges)

# source population: single-line Hα emitters (after thinning) at the redshifts that map into each bin
lam_ratio = pipe.line_set["Ha"].rest_wave / pipe.line_set["OIII_5007"].rest_wave
src_edges = (1 + edges) / lam_ratio - 1
c_ha_o3 = pipe.confusion.matrix[i_ha, i_o3]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
w = np.diff(edges)
ax1.bar(zc - 0.22 * w, n_true, width=0.42 * w, color=plotting.feature_color("OIII_5007"), label="true [O III] emitters")
ax1.bar(zc + 0.22 * w, n_int, width=0.42 * w, color=plotting.feature_color("Ha"), label="Hα emitters assigned [O III]")
ax1.set_xlabel("redshift_assigned"); ax1.set_ylabel("galaxies in the [O III] sample"); ax1.set_title("Composition of the assigned-[O III] sample"); ax1.legend(fontsize=8)
ax1.grid(axis="x", visible=False)
ax2.step(zc, 100 * n_int / n_all, where="mid", color=plotting.feature_color("Ha"), label="measured Hα interloper fraction")
ax2.axhline(100 * c_ha_o3, color=plotting.TEXT_SECONDARY, ls="--", lw=1, label=f"confusion matrix entry C(Hα→[O III]) = {100 * c_ha_o3:.0f}%")
ax2.set_ylim(0, 60); ax2.set_xlabel("redshift_assigned"); ax2.set_ylabel("% of the [O III] sample"); ax2.set_title("Contamination fraction versus confusion probability"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()
print(f"[O III] sample: {n_all.sum():,} galaxies; true [O III] {n_true.sum():,}; Hα interlopers {n_int.sum():,} "
      f"({100 * n_int.sum() / n_all.sum():.1f}%), from a source population of single-line Hα emitters at z_rsd ~ {src_edges[0]:.2f}-{src_edges[-1]:.2f}")
n_src = np.sum((pdata['line_true'] == i_ha) & (pdata['n_det'] == 1) & (pdata['redshift_rsd'] >= src_edges[0]) & (pdata['redshift_rsd'] <= src_edges[-1]))
print(f"single-line Hα emitters in the parent at those redshifts (before thinning): {n_src:,}; "
      f"x C(Hα→[O III]) = {c_ha_o3:.3f} -> {n_src * c_ha_o3:,.0f} interlopers before thinning, versus {n_true.sum():,} true [O III] emitters kept")"""),
    md("""
So the 5 % matrix entry becomes a ~30 % contamination because the source population is an order of
magnitude larger than the target population. Lowering the interloper fraction in this scenario
requires either a smaller C(Hα→[O III]) (the calibration the plan calls the main external input),
or the extra information the comments suggest (photometric redshifts, missing companion lines)
entering the confusion model; thinning to a forecast dN/dz does not change it.
"""),
    md("""
The sample files retain `ra`, `dec`, `redshift_true`, `redshift_rsd`, `redshift_assigned` and `id_outcome` by default
(`output.sample_columns`); the parent catalog additionally has `line_bitmask`, `line_true`,
`line_assigned` and `id_outcome`, so completeness, contamination and clustering can be checked
against the truth (plan section 4). Open them with `opencosmo.open(path)`.
"""),
]


# =====================================================================================
# 06 - ideal-detection dN/dz census (flux + redshift + bandpass cuts only)
# =====================================================================================
NB06 = [
    md("""
# 06 — dN/dz of emitters above the flux limit (no confusion, no downsampling)

A census of the mock with only three cuts: the line flux exceeds the limit (1.0e-16 or 1.5e-16 erg/s/cm²),
the observed line wavelength falls in the Roman grism range (1.00–1.93 μm), and 0.5 < z < 3.0. No confusion
matrix, no interlopers, no downsampling. This is the population the Wang et al. (2022) tables describe
(true emitters above the flux limit, "observational efficiency not included"), so it is the cleanest
mock-versus-forecast comparison.

Each line is counted separately, and galaxies are split by the number of lines of the set that pass the
cut: exactly one (single-line galaxies, whose redshift would rely on line identification) or more than one
(multi-line galaxies). `romanmock.census` does the counting slice by slice, in parallel, reusing the pipeline's
flux and detection code. Run it on a compute node (`sbatch scripts/execute_notebooks.sh 06`) or on a login
node with a few workers (~15 s per slice per worker).
"""),
    code(SETUP),
    code("""
import time
from romanmock.census import LineCensus, line_census
from romanmock.io import select_files, suggest_n_workers, memory_budget_gb, open_catalog
from romanmock.reference_dndz import wang2022_bins, wang2022_errors, WANG2022_CITATION
from romanmock.roman import GRISM_WAVE_RANGE_ANGSTROM
from romanmock.selection import sky_area_deg2

FLUX_LIMITS = [1.0e-16, 1.5e-16]                              # erg/s/cm², both tabulated by Wang et al. (2022)
Z_EDGES = np.round(np.arange(0.5, 3.0 + 1e-9, 0.1), 2)         # the tables' 0.1-wide bins
ZC = 0.5 * (Z_EDGES[1:] + Z_EDGES[:-1])
# every line of the set is counted; lines with no galaxy above the limit are left out of the legends and table
CENSUS_FILE = os.path.join(WORK, "census_z0p5_3p0.npz")

files = select_files(CATALOG, z_min=Z_EDGES[0], z_max=Z_EDGES[-1])
N_WORKERS = suggest_n_workers(max_workers=32, per_worker_gb=4.0)
print(f"{len(files)} slices overlap 0.5 < z < 3.0; memory budget {memory_budget_gb():.0f} GB -> {N_WORKERS} worker(s)")
print(f"bandpass {GRISM_WAVE_RANGE_ANGSTROM} Å; binned in redshift_obs (peculiar velocity included), D_L at redshift_true")"""),
    code("""
if os.path.exists(CENSUS_FILE):
    census = LineCensus.load(CENSUS_FILE)
    print(f"loaded {CENSUS_FILE} ({len(census.steps)} slices)")
else:
    t0 = time.time()
    census = line_census(CATALOG, FLUX_LIMITS, Z_EDGES, line_set="roman_transitions", n_workers=N_WORKERS, verbose=False)
    census.save(CENSUS_FILE)
    print(f"{len(census.steps)} slices in {(time.time() - t0) / 60:.1f} min -> {CENSUS_FILE}")
print(f"{census.n_all.sum():,} galaxies with 0.5 < z < 3.0")"""),
    md("""
## Sky area

Occupied HEALPix pixels of the input lightcone at the finest index level, edge pixels counted half. All
slices in the redshift range are opened together (a single slice is too sparse: empty pixels inside the
footprint would be mistaken for edges and bias the area low by a few per cent).
"""),
    code("""
t0 = time.time()
area = sky_area_deg2(open_catalog(files))
print(f"area = {area:.1f} deg² (nominal 5 % of the sky = {0.05 * 41252.96:.1f}), {time.time() - t0:.0f} s")"""),
    md("""
## dN/dz per line at both flux limits

Solid steps: the mock. Dashed and dotted steps with shaded ±1σ bands: Wang et al. (2022) Tables 1–2 (Galacticus)
for Hα and [O III] 5007 at the same flux limit, for their two dust models, A_V = 1.65 (dashed) and A_V = 1.92
(dotted); the tabulated uncertainties combine jackknife resampling and cosmic variance. Every galaxy with the line above the limit is counted, whatever its other lines do. All nine lines of
the `roman_transitions` set are counted; lines with fewer than one galaxy per deg² above the limit (the auroral
[O III] 4363, never, and [S III] 6312, ~150 galaxies in the whole footprint) are left out of the legends and the table.
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, lim in zip(axes, FLUX_LIMITS):
    for name in census.line_names:
        y = census.dndz(name, lim, area)
        if census.integrated(name, lim, area) < 1:          # auroral lines: a handful of galaxies in the whole footprint
            continue
        ax.step(ZC, y, where="mid", color=plotting.feature_color(name), label=f"{name} ({census.integrated(name, lim, area):,.0f} /deg²)")
    for name in ["Ha", "OIII_5007"]:
        for dust_av, ls in [(1.65, "--"), (1.92, ":")]:
            edges, ref = wang2022_bins(name, lim, dust_av)
            err = wang2022_errors(name, lim, dust_av)
            zc_ref = 0.5 * (edges[1:] + edges[:-1])
            ax.step(zc_ref, ref, where="mid", color=plotting.feature_color(name), ls=ls, lw=1.2,
                    label=f"{name} Wang+22 A$_V$={dust_av} ±1σ")
            ax.fill_between(edges, np.append(ref - err, ref[-1] - err[-1]), np.append(ref + err, ref[-1] + err[-1]),
                            step="post", color=plotting.feature_color(name), alpha=0.15, lw=0)
    ax.set_yscale("log"); ax.set_ylim(10, None); ax.set_xlim(Z_EDGES[0], Z_EDGES[-1])
    ax.set_xlabel("redshift"); ax.set_title(f"F > {lim:.1e} erg/s/cm²")
    ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
axes[0].set_ylabel("dN/dz per deg²")
fig.suptitle("Emitters above the flux limit in 1.00–1.93 μm, no confusion, no downsampling", y=1.02)
plt.show()"""),
    code("""
# integrated counts over 0.5 < z < 3.0: per deg² and over the whole footprint
from astropy.table import Table

rows = []
for lim in FLUX_LIMITS:
    for name in census.line_names:
        tot = census.integrated(name, lim, area)
        if tot < 1:
            continue
        single = census.integrated(name, lim, area, "single")
        rows.append({"flux_limit": f"{lim:.1e}", "line": name, "N/deg²": round(tot), "single-line /deg²": round(single),
                     "multi-line /deg²": round(tot - single), "single fraction": round(single / tot, 3),
                     "N total": int(census.hist(name, lim).sum())})
table = Table(rows=rows)
table.write(os.path.join(WORK, "census_z0p5_3p0.csv"), format="csv", overwrite=True)
table"""),
    md("""
## Single-line versus multi-line galaxies

For each line: all galaxies with the line above the limit (solid), those with **only** that line above the
limit inside the bandpass (dashed), and those with at least one other line above it (dotted). The single-line
population is the one exposed to misidentification; the multi-line one gets an unambiguous redshift.
"""),
    code("""
SHOW = ["Ha", "OIII_5007", "OII_3727", "Hb"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for ax, name in zip(axes.ravel(), SHOW):
    for lim, col in zip(FLUX_LIMITS, [plotting.CATEGORICAL[0], plotting.CATEGORICAL[1]]):
        for which, ls in [("all", "-"), ("single", "--"), ("multi", ":")]:
            y = census.dndz(name, lim, area, which)
            ax.step(ZC, y, where="mid", color=col, ls=ls, label=f"F > {lim:.1e}, {which}")
    ax.set_yscale("log"); ax.set_ylim(10, None); ax.set_xlim(Z_EDGES[0], Z_EDGES[-1])
    ax.set_title(name); ax.set_ylabel("dN/dz per deg²")
    ax.legend(fontsize=7, ncol=2)
for ax in axes[1]:
    ax.set_xlabel("redshift")
plt.tight_layout(); plt.show()"""),
    code("""
fig, ax = plt.subplots(figsize=(9, 4.4))
for name in SHOW:
    for lim, ls in zip(FLUX_LIMITS, ["-", "--"]):
        ax.step(ZC, census.single_fraction(name, lim), where="mid", color=plotting.feature_color(name), ls=ls,
                label=f"{name}, F > {lim:.1e}")
ax.set_ylim(0, 1); ax.set_xlim(Z_EDGES[0], Z_EDGES[-1])
ax.set_xlabel("redshift"); ax.set_ylabel("fraction with no other line above the limit")
ax.set_title("Single-line fraction of each line's emitters"); ax.legend(fontsize=7.5, ncol=2); plt.show()"""),
    md("""
Reading the single-line fraction: at z < 1 most Hα emitters also have [S III] 9532 (and 9069) above the limit,
so only ~30 % are single-line. Once the [S III] lines leave the grism (z ≈ 1.0–1.1) the remaining companions,
[O III] 5007 and Hβ, pass the flux cut for a minority of Hα emitters, and the single-line fraction climbs to
~0.8–0.85 by z ≈ 1.9. [O III] 5007 emitters are almost never single-line while Hα is in the bandpass (z < 1.94),
and ~85–90 % single-line beyond it, where only Hβ, [O III] 4959 and [O II] could accompany the line.
"""),
    md("""
## Ratio to the Wang et al. (2022) forecast

Mock over forecast per 0.1 bin, for the total population of each line (the tables do not split by line multiplicity).
Error bars propagate the tabulated ±1σ of the forecast; the mock's own Poisson error is negligible in comparison.
"""),
    code("""
fig, ax = plt.subplots(figsize=(9, 4.4))
for name in ["Ha", "OIII_5007"]:
    for lim, ls in zip(FLUX_LIMITS, ["-", "--"]):
        edges, ref = wang2022_bins(name, lim)
        zc_ref = np.round(0.5 * (edges[1:] + edges[:-1]), 3)
        mock = census.dndz(name, lim, area)
        sel = np.isin(np.round(ZC, 3), zc_ref)
        ratio = mock[sel] / ref
        ratio_err = ratio * wang2022_errors(name, lim) / ref
        ax.step(zc_ref, ratio, where="mid", color=plotting.feature_color(name), ls=ls, label=f"{name}, F > {lim:.1e}")
        ax.errorbar(zc_ref, ratio, yerr=ratio_err, fmt="none", ecolor=plotting.feature_color(name), elinewidth=0.9, capsize=2)
        lo, hi = edges[0], edges[-1]
        print(f"{name:10s} F > {lim:.1e}: mock/Wang+22 over {lo:.1f} < z < {hi:.1f} = "
              f"{census.integrated(name, lim, area, z_range=(lo, hi)) / (ref * np.diff(edges)).sum():.2f}  "
              f"(per-bin range {ratio.min():.2f}–{ratio.max():.2f})")
ax.axhline(1, color=plotting.TEXT_SECONDARY, lw=0.8)
ax.set_yscale("log"); ax.set_xlim(Z_EDGES[0], Z_EDGES[-1])
ax.set_xlabel("redshift"); ax.set_ylabel("mock / Wang et al. (2022)")
ax.set_title("Ratio to Wang et al. (2022), A_V = 1.65"); ax.legend(fontsize=8); plt.show()"""),
    md("""
Files written to `WORK`: `census_z0p5_3p0.npz` (all counts, reloadable with `LineCensus.load`) and
`census_z0p5_3p0.csv` (the integrated table). To add a flux limit or change the bins, delete the `.npz` and rerun;
`line_census` accepts any list of limits, tabulated or not.
"""),
]


# =====================================================================================
# 07 - two-step catalog construction: base flux catalog, then limits and redshifts
# =====================================================================================
NB07 = [
    md("""
# 07 — Two-step catalog construction

Reading the input mock (340 GB) is the expensive part of building a catalog, and it does not depend on
the choices one wants to vary afterwards. The two-step routine reads it once:

1. **Base flux catalog** (`romanmock base`, `pipe.build_base()`): every galaxy with at least one line above the
   primary flux limit inside the grism bandpass, with `gal_id`, `ra`, `dec`, `redshift_true`, `redshift_rsd`, the
   flux of **all** lines of the line set, and the bitmask at the base limits (`line_bitmask_base`). The copied
   columns are configurable.
2. **Detection and redshifts** (`romanmock assign`, `source: "base"`): read the base catalog, apply the flux
   limits of your choice (not deeper than the base), rebuild the bitmask and assign redshifts. The output is an
   ordinary parent catalog, so sample building and the diagnostics work as in the other notebooks.

This notebook follows the file-based route, which writes OpenCosmo lightcones. For everyday use there is a
shorter one, shown in notebook 09: `BaseCatalog(base_dir).make_catalog(...)` returns the catalog in memory in
15–20 s, with the same galaxies for the same settings.

Measured on the full lightcone with a base limit of 1.0e-16 erg/s/cm²: step 1 takes about one minute on a CPU
node (32 workers, 5.7 GB each) and 7.6 GB of disk; step 2 takes 1.5–2 minutes on a **login node** with 4
workers at 2.4 GB each.
"""),
    code(SETUP),
    md("""
## Step 1: the base flux catalog

The base limits are the `sensitivity` block of the configuration (here the package defaults), and `base.columns`
lists the columns copied from the input mock. Slices already built with the same settings are reused, so this
cell is fast the second time.
"""),
    code("""
import time, glob
from romanmock import RomanMockConfig, RomanMockPipeline
from romanmock.io import suggest_n_workers, memory_budget_gb, read_manifest, spatial_index_nbytes
from romanmock.roman import (DEFAULT_LINE_FLUX_LIMIT as BASE_LIMIT, DEFAULT_SECONDARY_LINE_FLUX_LIMIT as SECONDARY_LIMIT,
                             GRISM_WAVE_RANGE_ANGSTROM as GRISM)

TWO = os.path.join(WORK, "two_step")
BASE_DIR = os.path.join(TWO, "base")

base_cfg = RomanMockConfig.from_dict({
    "catalog": {"path": CATALOG},                                   # every slice, z = 0-4
    "line_set": "roman_transitions",
    "sensitivity": {"type": "constant", "flux_limit": BASE_LIMIT, "secondary_flux_limit": SECONDARY_LIMIT,
                    "wave_range": list(GRISM)},
    "base": {"dir": BASE_DIR, "columns": ["gal_id", "ra", "dec", "redshift_true"]},
})
base_pipe = RomanMockPipeline(base_cfg)
n1 = suggest_n_workers(max_workers=32, per_worker_gb=6.0)
print(f"memory budget {memory_budget_gb():.0f} GB -> {n1} worker(s) for step 1")
t0 = time.time()
base_files = base_pipe.build_base(n_workers=n1, verbose=False)
manifest = read_manifest(BASE_DIR, base_cfg.base.tag)
n_in = sum(v["n_input"] for v in manifest["slices"].values())
n_base = sum(v["n_galaxies"] for v in manifest["slices"].values())
print(f"{len(base_files)} files in {time.time() - t0:.0f} s; {n_base:,} of {n_in:,} galaxies ({100 * n_base / n_in:.1f}%) "
      f"have a line above {BASE_LIMIT:.1e} erg/s/cm² in the bandpass")"""),
    code("""
base = oc.open(*base_files)
print(f"{len(base):,} galaxies, z_range {base.z_range}")
for c in ["gal_id", "ra", "dec", "redshift_true", "redshift_rsd", "line_bitmask_base", "Ha_flux", "OIII_5007_flux"]:
    unit = str(base.units.get(c) or "-")
    print(f"  {c:<16s} [{unit:^14s}] {(base.descriptions.get(c) or '(no description in the input mock)')[:95]}")
print("flux columns:", [c for c in base.columns if c.endswith("_flux")])

one = oc.open(base_files[len(base_files) // 2]).select("gal_id", "redshift_rsd", "line_bitmask_base", "Ha_flux", "Hb_flux",
                                                         "OIII_5007_flux").get_data("numpy")
from astropy.table import Table
t = Table({k: one[k][:6] for k in one}); t["Ha_flux"].format = t["Hb_flux"].format = t["OIII_5007_flux"].format = ".2e"
t"""),
    md("""
Every line flux is stored, also for lines below the limits or outside the bandpass. That is what lets step 2
choose any secondary limit. The histogram shows the base selection: the brightest in-band line of every
galaxy is above the base limit, while the individual lines extend far below it.
"""),
    code("""
flux_cols = base_pipe.flux_columns
d = oc.open(base_files[len(base_files) // 2]).select(*flux_cols, "redshift_rsd").get_data("numpy")
flux = np.stack([d[c] for c in flux_cols], axis=1)
wave = base_pipe.line_set.rest_waves[None, :] * (1 + d["redshift_rsd"][:, None])
inband = (wave > GRISM[0]) & (wave < GRISM[1])
brightest = np.where(inband, flux, 0).max(axis=1)

fig, ax = plt.subplots(figsize=(8.5, 4.2))
bins = np.linspace(-18.5, -14.5, 80)
ax.hist(np.log10(brightest), bins=bins, histtype="step", color=plotting.TEXT_PRIMARY, lw=1.6, label="brightest in-band line")
for name in ["Ha", "OIII_5007", "Hb"]:
    i = base_pipe.line_set.bit(name)
    ax.hist(np.log10(flux[inband[:, i], i]), bins=bins, histtype="step", color=plotting.feature_color(name), label=f"{name} (in band)")
for lim, ls in [(BASE_LIMIT, "--"), (SECONDARY_LIMIT, ":")]:
    ax.axvline(np.log10(lim), color=plotting.TEXT_SECONDARY, lw=1, ls=ls)
ax.set_xlabel("log10 line flux [erg/s/cm²]"); ax.set_ylabel("galaxies"); ax.set_yscale("log")
ax.set_title("One slice of the base catalog: dashed = primary limit, dotted = secondary limit"); ax.legend(fontsize=8); plt.show()"""),
    md("""
## Step 2: flux limits, bitmask and redshifts from the base catalog

A configuration with `source: "base"` makes `RomanMockPipeline.run` read the base catalog instead of the input
mock. Three settings are run here: the base limits themselves, a brighter primary limit, and the brighter
limit without the second tier. Each writes its own parent catalog and the effective `config.yaml` next to it.
"""),
    code("""
def assign(flux_limit, secondary):
    tag = f"nb_F{flux_limit:.1e}_S" + ("none" if secondary is None else f"{secondary:.1e}")
    cfg = RomanMockConfig.from_dict({
        "source": "base",
        "base": {"dir": BASE_DIR},
        "output": {"dir": os.path.join(TWO, tag), "passthrough_columns": ["gal_id", "ra", "dec", "redshift_true"]},
        "sensitivity": {"type": "constant", "flux_limit": flux_limit, "secondary_flux_limit": secondary,
                        "wave_range": list(GRISM)},
        # confusion: omitted -> the package default scenario
        "seed": 2026,
    })
    pipe = RomanMockPipeline(cfg)
    t0 = time.time()
    files = pipe.run(n_workers=N2, verbose=False)
    cfg.to_yaml(os.path.join(cfg.output.dir, "config.yaml"))       # for `romanmock select` / `romanmock summary`
    print(f"{str(pipe.sensitivity):<75s} {len(files)} files, {time.time() - t0:5.0f} s -> {tag}")
    return pipe

N2 = suggest_n_workers(max_workers=16, per_worker_gb=2.5)
print(f"memory budget {memory_budget_gb():.0f} GB -> {N2} worker(s) for step 2")
SETTINGS = {"1.0e-16 / 4.0e-17": (BASE_LIMIT, SECONDARY_LIMIT), "1.5e-16 / 4.0e-17": (1.5e-16, SECONDARY_LIMIT),
            "1.5e-16, single tier": (1.5e-16, None)}
runs = {label: assign(*limits) for label, limits in SETTINGS.items()}"""),
    code("""
from romanmock.diagnostics import outcome_summary
from romanmock.selection import sky_area_deg2

cols = ["redshift_assigned", "id_outcome", "n_det", "line_assigned"]
data, rows = {}, []
for label, pipe in runs.items():
    parent = pipe.open_parent()
    data[label] = parent.select(*cols).get_data("numpy")
    s = outcome_summary(data[label]); n = len(parent)
    files = glob.glob(os.path.join(pipe.config.output.dir, "*.hdf5"))
    rows.append({"limits (primary / secondary)": label, "galaxies": n, "multi-line %": round(100 * s["MULTI_LINE"] / n, 2),
                 "single correct %": round(100 * s["SINGLE_CORRECT"] / n, 2), "interlopers %": round(100 * s["SINGLE_MISID"] / n, 3),
                 "disk GB": round(sum(os.path.getsize(f) for f in files) / 1e9, 2)})
area = sky_area_deg2(parent)
print(f"area = {area:.1f} deg²")
Table(rows)"""),
    md("""
The brighter primary limit removes galaxies; the second tier does not change which galaxies are detected, only how
many of their lines count, and therefore how many galaxies need a single-line identification.
"""),
    code("""
from romanmock.diagnostics import dndz

z_edges = np.round(np.arange(0.5, 3.0 + 1e-9, 0.1), 2); zc = 0.5 * (z_edges[1:] + z_edges[:-1])
line_set = runs["1.0e-16 / 4.0e-17"].line_set
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
for (label, d), col in zip(data.items(), plotting.CATEGORICAL):
    for name, ls in [("Ha", "-"), ("OIII_5007", "--")]:
        sel = (d["line_assigned"] == line_set.bit(name)) & np.isfinite(d["redshift_assigned"])
        ax1.step(zc, dndz(d["redshift_assigned"][sel], z_edges, area), where="mid", color=col, ls=ls,
                 label=f"{label}: {name}")
    counts = np.bincount(d["n_det"], minlength=8)[1:8] / len(d["n_det"])
    ax2.step(np.arange(1, 8), counts, where="mid", color=col, label=label)
ax1.set_yscale("log"); ax1.set_ylim(30, None); ax1.set_xlabel("redshift_assigned"); ax1.set_ylabel("dN/dz per deg²")
ax1.set_title("Assigned Hα (solid) and [O III] 5007 (dashed)"); ax1.legend(fontsize=7.5, ncol=1)
ax2.set_xlabel("number of detected lines"); ax2.set_ylabel("fraction of detected galaxies")
ax2.set_title("Lines per galaxy"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()"""),
    md("""
## Analysis samples from a step-2 catalog

`build_samples` (`romanmock select`) works on any step-2 output. It is built slice by slice as well: the
redshift histograms behind the thinning probabilities are accumulated in passes over the slices, and each
sample is written as one small file per slice in `samples/<name>/`. On the full catalog this takes about
three minutes on a login node with 4 workers at 1.7 GB each. `open_sample` opens a sample as one lightcone.
"""),
    code("""
from romanmock.config import PopulationThinningConfig, SampleConfig
from romanmock.io import open_sample

pipe = runs["1.0e-16 / 4.0e-17"]
pipe.config.population_thinning = PopulationThinningConfig(
    targets={"Ha": {"reference": "wang2022", "dust_av": 1.65}, "OIII_5007": {"reference": "wang2022", "dust_av": 1.65}},
    by="line_true", z_column="redshift_rsd", outside_range="keep")
pipe.config.output.overwrite = True
t0 = time.time()
sample_paths = pipe.build_samples(samples=[
    SampleConfig(name="halpha_z1p0_1p9", z_min=1.0, z_max=1.9, assigned_lines=["Ha"]),
    SampleConfig(name="oiii_z2p0_2p9", z_min=2.0, z_max=2.9, assigned_lines=["OIII_5007"]),
], n_workers=N2)
print(f"{time.time() - t0:.0f} s")
for p in sample_paths:
    s = open_sample(p)
    print(f"{os.path.basename(p):<18s} {len(s):>12,} galaxies in {len(glob.glob(os.path.join(p, '*.hdf5')))} slice files; columns {s.columns}")"""),
    md("""
## What step 2 can and cannot do

The requested selection must be contained in the base catalog. A primary limit deeper than the base, a wider
bandpass, or a column that was not stored is refused when the pipeline is created, before anything is written.
The secondary limit is free, and so are the confusion model and the seed.
"""),
    code("""
attempts = {
    "primary limit 5e-17 (deeper than the base)": {"sensitivity": {"type": "constant", "flux_limit": 5e-17, "wave_range": list(GRISM)}},
    "bandpass 0.75-1.93 um (wider than the base)": {"sensitivity": {"type": "constant", "flux_limit": 2e-16, "wave_range": [7500.0, 19300.0]}},
    "passthrough column logsm_obs (not stored)": {"output": {"dir": os.path.join(TWO, "nb_refused"), "passthrough_columns": ["ra", "dec", "logsm_obs"]}},
    "secondary limit 1e-17 (allowed)": {"sensitivity": {"type": "constant", "flux_limit": 1e-16, "secondary_flux_limit": 1e-17, "wave_range": list(GRISM)}},
}
for what, over in attempts.items():
    cfg = {"source": "base", "base": {"dir": BASE_DIR}, "output": {"dir": os.path.join(TWO, "nb_refused")}, **over}
    try:
        RomanMockPipeline(RomanMockConfig.from_dict(cfg))
        print(f"OK       {what}")
    except (ValueError, KeyError) as err:
        print(f"REFUSED  {what}\\n         {str(err)[:150]}")"""),
    md("""
## Recovering other columns through `gal_id`

Anything that was not stored can be joined from the input mock slice by slice. Here the stellar mass of the
detected galaxies of one slice is fetched and compared with all galaxies of that slice.
"""),
    code("""
from romanmock.io import select_files

step = 219
pipe = runs["1.0e-16 / 4.0e-17"]
det = oc.open(os.path.join(pipe.config.output.dir, f"lc_cores-{step}.romanmock.hdf5")).select("gal_id", "n_det").get_data("numpy")
src = oc.open(select_files(CATALOG, steps=[step])[0], synth_cores=False, keep_top_host=True).select("gal_id", "logsm_obs").get_data("numpy")
order = np.argsort(src["gal_id"])
pos = order[np.searchsorted(src["gal_id"][order], det["gal_id"])]
assert np.array_equal(src["gal_id"][pos], det["gal_id"])            # every detected galaxy is found, ids are unique
logsm = src["logsm_obs"][pos]

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(7, 12, 60)
ax.hist(src["logsm_obs"], bins=bins, histtype="step", color=plotting.TEXT_SECONDARY, label=f"all galaxies of the slice ({len(order):,})")
ax.hist(logsm, bins=bins, histtype="step", color=plotting.CATEGORICAL[0], label=f"detected ({len(logsm):,})")
ax.hist(logsm[det["n_det"] == 1], bins=bins, histtype="step", color=plotting.CATEGORICAL[1], label="detected, single line")
ax.set_yscale("log"); ax.set_xlabel("log10 stellar mass [Msun]"); ax.set_ylabel("galaxies")
ax.set_title(f"Slice {step}: stellar mass joined through gal_id"); ax.legend(fontsize=8); plt.show()"""),
    md("""
## Disk usage

OpenCosmo writes the HEALPix index of a lightcone uncompressed (268 MB per file). `romanmock` stores it
gzip-compressed and leaves the data untouched (`compress_index`, on by default); files written before can be
converted in place with `romanmock compress-index <dir>`.
"""),
    code("""
def usage(d):
    files = glob.glob(os.path.join(d, "*.hdf5"))
    return len(files), sum(os.path.getsize(f) for f in files) / 1e9, sum(spatial_index_nbytes(f) for f in files) / 1e9

rows = []
for label, d in [("base catalog", BASE_DIR), *[(f"step 2, {k}", p.config.output.dir) for k, p in runs.items()]]:
    n, total, index = usage(d)
    rows.append({"catalog": label, "files": n, "total GB": round(total, 2), "of which index GB": round(index, 3),
                 "with an uncompressed index GB": round(total - index + 0.268 * n, 1)})
Table(rows)"""),
    md("""
## The same from the command line

```bash
romanmock base   cfg.yaml --workers 32                       # step 1, once (sbatch scripts/run_base.sh cfg.yaml)
romanmock assign cfg.yaml --flux-limit 1.5e-16 --secondary-flux-limit 4e-17 \\
                 --output-dir $PSCRATCH/mock_F1.5e-16 --workers 4      # step 2, on a login node
romanmock summary $PSCRATCH/mock_F1.5e-16/config.yaml        # outcome counts, empirical confusion
romanmock select  $PSCRATCH/mock_F1.5e-16/config.yaml --workers 4   # analysis samples (add a `samples:` block first)
```

`cfg.yaml` needs `catalog.path`, the `sensitivity` block with the base limits, and `base.dir`
(see `configs/roman_grism_baseline.yaml`). `--secondary-flux-limit 0` switches the second tier off.
"""),
]


# =====================================================================================
# 08 - dN/dz of single- and multi-line galaxies for different secondary flux limits
# =====================================================================================
NB08 = [
    md("""
# 08 — Single-line and multi-line dN/dz versus the secondary flux limit

Notebook 06 counts emitters with one flux limit for every line. The pipeline uses two: a galaxy needs one line
above the **primary** limit inside the grism bandpass, and its other lines then count as detected above a lower
**secondary** limit. This notebook repeats the single-line / multi-line comparison of notebook 06 for secondary
limits of 8e-17 and 4e-17 erg/s/cm², next to the single-threshold case.

Same cuts as in notebook 06 otherwise: observed wavelength in 1.00–1.93 μm, 0.5 < z < 3.0, no confusion matrix,
no interlopers, no downsampling. Definitions, for a line X and limits (primary P, secondary S):

- **all**: galaxies in which X is detected, i.e. X > P, or X > S while another line of the galaxy is above P;
- **single**: X > P and no other line above S, the galaxies whose redshift rests on identifying one line;
- **multi**: X detected and at least one other line detected.

The secondary limit does not change which galaxies are detected. It moves galaxies from single to multi, and it
adds faint lines of already detected galaxies to the "all" counts of those lines.
"""),
    code(SETUP),
    code("""
import time
from romanmock.census import LineCensus, line_census
from romanmock.io import select_files, suggest_n_workers, memory_budget_gb, open_catalog
from romanmock.roman import GRISM_WAVE_RANGE_ANGSTROM
from romanmock.selection import sky_area_deg2

PRIMARY_LIMITS = [1.0e-16, 1.5e-16]                  # erg/s/cm²; 1.0e-16 is the package default
SECONDARY_LIMITS = [None, 8.0e-17, 4.0e-17]          # None = single threshold (secondary = primary)
LIMITS = [(p, s) for p in PRIMARY_LIMITS for s in SECONDARY_LIMITS]
Z_EDGES = np.round(np.arange(0.5, 3.0 + 1e-9, 0.1), 2)
ZC = 0.5 * (Z_EDGES[1:] + Z_EDGES[:-1])
CENSUS_FILE = os.path.join(WORK, "census_secondary_z0p5_3p0.npz")

# one colour per secondary limit, one line style per class, in every figure
COLOR = dict(zip(SECONDARY_LIMITS, plotting.CATEGORICAL))
STYLE = {"all": "-", "single": "--", "multi": ":"}
def label(s):
    return "single threshold" if s is None else f"secondary {s:.0e}"

files = select_files(CATALOG, z_min=Z_EDGES[0], z_max=Z_EDGES[-1])
N_WORKERS = suggest_n_workers(max_workers=32, per_worker_gb=5.0)
print(f"{len(files)} slices overlap 0.5 < z < 3.0; memory budget {memory_budget_gb():.0f} GB -> {N_WORKERS} worker(s)")"""),
    code("""
if os.path.exists(CENSUS_FILE):
    census = LineCensus.load(CENSUS_FILE)
    print(f"loaded {CENSUS_FILE} ({len(census.steps)} slices)")
else:
    t0 = time.time()
    census = line_census(CATALOG, LIMITS, Z_EDGES, line_set="roman_transitions", n_workers=N_WORKERS, verbose=False)
    census.save(CENSUS_FILE)
    print(f"{len(census.steps)} slices in {(time.time() - t0) / 60:.1f} min -> {CENSUS_FILE}")
print("limits (primary, secondary):", census.limits)
t0 = time.time()
area = sky_area_deg2(open_catalog(files))
print(f"area = {area:.1f} deg² ({time.time() - t0:.0f} s)")"""),
    md("""
## dN/dz per line: all, single-line and multi-line

One panel per line, one colour per secondary limit. Solid: all galaxies with the line detected. Dashed: those
with no other detected line. Dotted: those with at least one other detected line.
"""),
    code("""
SHOW = ["Ha", "OIII_5007", "Hb", "OII_3727"]

def plot_lines(primary, lines=SHOW, ylim=(10, None)):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharex=True)
    for ax, name in zip(axes.ravel(), lines):
        for s in SECONDARY_LIMITS:
            for which, ls in STYLE.items():
                ax.step(ZC, census.dndz(name, primary, area, which, secondary=s), where="mid", color=COLOR[s], ls=ls,
                        lw=1.6 if which == "all" else 1.3, label=f"{label(s)}, {which}")
        ax.set_yscale("log"); ax.set_ylim(*ylim); ax.set_xlim(Z_EDGES[0], Z_EDGES[-1])
        ax.set_title(name); ax.set_ylabel("dN/dz per deg²")
    for ax in axes[1]:
        ax.set_xlabel("redshift")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(f"Primary limit {primary:.1e} erg/s/cm², 1.00–1.93 μm, no confusion, no downsampling", y=1.0)
    plt.tight_layout(); plt.show()

plot_lines(1.0e-16)"""),
    code("""
plot_lines(1.5e-16)"""),
    md("""
Reading the panels:

- **Hα.** The solid curves coincide: Hα is almost always the brightest line, so a secondary limit adds very few
  Hα lines. What changes is the split. With a secondary limit of 8e-17 about a quarter of the single-line Hα
  emitters become multi-line; with 4e-17 nearly all of them do, because [O III] 5007 and Hβ (z > 1) or
  [S III] 9532/9069 (z < 1.1) are then detected as well.
- **[O III] 5007, Hβ, [O II].** Here the solid curves rise with a lower secondary limit: these lines are often
  fainter than the primary limit in galaxies that Hα (or [O III]) has already anchored.
- **[O III] 5007 at z > 1.94**, where Hα has left the grism: single-line with one threshold, still mostly
  single-line at 8e-17, never at 4e-17. Two properties of the mock cause this. The doublet ratio
  5007/4959 is fixed near 3.04, so [O III] 4959 exceeds 4e-17 as soon as 5007 exceeds 1.2e-16, and Hβ is above
  4e-17 in the remaining galaxies. The same happens for [S III] 9532/9069 (ratio 2.53).

Keep in mind that 4e-17 is 2.6σ for a 6.5σ limit of 1e-16: these counts assume that such a line is usable
once the galaxy is anchored, and they contain no noise.
"""),
    md("""
## Single-line fraction of each line's emitters
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
for ax, primary in zip(axes, PRIMARY_LIMITS):
    for name in ["Ha", "OIII_5007"]:
        for s, ls in zip(SECONDARY_LIMITS, ["-", "--", ":"]):
            ax.step(ZC, census.single_fraction(name, primary, secondary=s), where="mid", color=plotting.feature_color(name),
                    ls=ls, label=f"{name}, {label(s)}")
    ax.set_ylim(0, 1); ax.set_xlim(Z_EDGES[0], Z_EDGES[-1]); ax.set_xlabel("redshift")
    ax.set_title(f"Primary limit {primary:.1e} erg/s/cm²"); ax.legend(fontsize=8)
axes[0].set_ylabel("fraction with no other detected line")
plt.tight_layout(); plt.show()"""),
    md("""
## Distinct galaxies

Each galaxy counted once, whatever its lines. The detected population (solid black) is the same for every
secondary limit; what changes is how much of it is single-line.
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
norm = np.diff(Z_EDGES) * area
for ax, primary in zip(axes, PRIMARY_LIMITS):
    ax.step(ZC, census.galaxies(primary) / norm, where="mid", color=plotting.TEXT_PRIMARY, lw=1.8, label="detected galaxies")
    for s in SECONDARY_LIMITS:
        ax.step(ZC, census.galaxies(primary, "single", secondary=s) / norm, where="mid", color=COLOR[s], ls="--",
                label=f"single-line, {label(s)}")
    ax.set_yscale("log"); ax.set_ylim(10, None); ax.set_xlim(Z_EDGES[0], Z_EDGES[-1]); ax.set_xlabel("redshift")
    ax.set_title(f"Primary limit {primary:.1e} erg/s/cm²"); ax.legend(fontsize=8)
axes[0].set_ylabel("dN/dz per deg²")
plt.tight_layout(); plt.show()"""),
    code("""
from astropy.table import Table

rows = []
for primary in PRIMARY_LIMITS:
    for s in SECONDARY_LIMITS:
        g, gs = census.galaxies(primary, secondary=s).sum(), census.galaxies(primary, "single", secondary=s).sum()
        rows.append({"primary": f"{primary:.1e}", "secondary": "-" if s is None else f"{s:.1e}", "line": "any (distinct galaxies)",
                     "all /deg²": round(g / area), "single /deg²": round(gs / area), "multi /deg²": round((g - gs) / area),
                     "single fraction": round(gs / g, 3)})
        for name in census.line_names:
            tot = census.integrated(name, primary, area, secondary=s)
            if tot < 1:
                continue
            single = census.integrated(name, primary, area, "single", secondary=s)
            rows.append({"primary": f"{primary:.1e}", "secondary": "-" if s is None else f"{s:.1e}", "line": name,
                         "all /deg²": round(tot), "single /deg²": round(single), "multi /deg²": round(tot - single),
                         "single fraction": round(single / tot, 3)})
table = Table(rows)
table.write(os.path.join(WORK, "census_secondary_z0p5_3p0.csv"), format="csv", overwrite=True)
table[[r["line"] in ("any (distinct galaxies)", "Ha", "OIII_5007", "Hb") for r in rows]]"""),
    md("""
Files written to `WORK`: `census_secondary_z0p5_3p0.npz` (reload with `LineCensus.load`; limits are
`(primary, secondary)` pairs, e.g. `census.dndz("Ha", 1e-16, area, "single", secondary=4e-17)`) and
`census_secondary_z0p5_3p0.csv` with every line. To try other limits, edit `PRIMARY_LIMITS` /
`SECONDARY_LIMITS`, delete the `.npz` and rerun.
"""),
]


# =====================================================================================
# 09 - catalogs from the base catalog, in memory
# =====================================================================================
NB09 = [
    md("""
# 09 — Catalogs from the base catalog

The two-step routine in its simplest form.

1. **Step 1, once** (`romanmock base`, notebook 07): the base catalog stores every galaxy with a line above
   1.0e-16 erg/s/cm² in the grism bandpass, with positions, redshifts and the flux of all nine lines.
2. **Step 2, as often as you like**: one call, `base.make_catalog(...)`, with the flux limits, the confusion
   matrix, the lines and the redshift range of your choice. It returns the catalog in memory as numpy arrays.

Nothing is written unless you ask for it, and nothing is remembered between calls: change a setting and run the
cell again. A call takes 15–20 s on a login node.
"""),
    code(SETUP),
    code("""
from romanmock import BaseCatalog

base = BaseCatalog(os.path.join(WORK, "two_step", "base"))
print(base)
print("columns a catalog can contain:", base.columns)"""),
    md("""
## Make a catalog

Every argument is optional; `help(base.make_catalog)` lists them all.
"""),
    code("""
confusion = {                                   # single-line galaxies: {true line: {assigned line: probability}}
    "Ha":        {"OIII_5007": 0.2},            # lines that are not listed are identified correctly
    "OIII_5007": {"Ha": 0.2},
    "SIII_9532": {"Ha": 0.2},
    "Hb":        {"OIII_5007": 0.2},
}

cat = base.make_catalog(
    flux_limit=1.5e-16,               # primary limit: at least one line of the galaxy above it
    secondary_flux_limit=8.0e-17,     # limit for the other lines of such a galaxy (None: no second tier)
    confusion=confusion,              # None: the package default
    lines=["Ha"],                     # galaxies in which Ha is identified (None: every line)
    z_range=(1.0, 1.9),               # range of redshift_assigned
    seed=2026,
)
print(cat.summary())"""),
    md("""
The call prints a **line report** (`verbose=True`, the default; `cat.line_report()` returns the text and
`cat.info["line_report"]` the numbers):

- the lines inside the grism bandpass for the redshift cut, with the number of catalog galaxies in which the
  line is detected, identified (the redshift of the galaxy is right; every detected line of a multi-line galaxy
  counts), the brightest line, mistaken for another line, and the number of interlopers taken for it;
- the line interlopers, single-line galaxies given a wrong line. Each row is a *potential* interloper: a
  line that, taken for another one, passes the redshift cut, with its probability `P` in the confusion
  matrix and the range of true redshift it comes from. The last columns give the *actual* number in the
  catalog and its share of the sample of that line.
"""),
    code("""
from astropy.table import Table

print(type(cat["ra"]), cat["ra"].dtype, cat["ra"].shape)      # columns are numpy arrays
t = Table({c: cat[c][:8] for c in cat.columns})
for c in ("ra", "dec", "redshift_true", "redshift_rsd", "redshift_assigned"):
    t[c].format = ".5f"
t"""),
    md("""
`redshift_assigned` is the measured redshift. For a galaxy with more than one detected line it equals
`redshift_rsd` (the true redshift including the peculiar velocity), not `redshift_true`. For a single-line
galaxy it follows from the line identity drawn from the confusion matrix: equal to `redshift_rsd` when the
identification is right (`id_outcome` 2), displaced when it is wrong (`id_outcome` 3, an interloper).
"""),
    code("""
multi, right, wrong = (cat["id_outcome"] == k for k in (1, 2, 3))
print("multi-line:        redshift_assigned == redshift_rsd for all:", np.array_equal(cat["redshift_assigned"][multi], cat["redshift_rsd"][multi]),
      "| == redshift_true for", f"{np.mean(cat['redshift_assigned'][multi] == cat['redshift_true'][multi]):.4f}")
print("single, correct:   max |redshift_assigned - redshift_rsd| =", f"{np.abs(cat['redshift_assigned'][right] - cat['redshift_rsd'][right]).max():.1e}")
print("single, interloper: true redshifts", f"{cat['redshift_rsd'][wrong].min():.2f}-{cat['redshift_rsd'][wrong].max():.2f}",
      "assigned", f"{cat['redshift_assigned'][wrong].min():.2f}-{cat['redshift_assigned'][wrong].max():.2f}")"""),
    md("""
## Change anything and run again

Each call is independent. Here the same selection for three confusion matrices and two pairs of flux limits.
"""),
    code("""
import time

rows = []
for label, conf in [("package default", None), ("20 % (the matrix above)", confusion), ("perfect identification", {})]:
    for primary, secondary in [(1.0e-16, 4.0e-17), (1.5e-16, 8.0e-17)]:
        t0 = time.time()
        c = base.make_catalog(flux_limit=primary, secondary_flux_limit=secondary, confusion=conf, lines=["Ha"],
                              z_range=(1.0, 1.9), seed=2026, verbose=False)
        o = c.outcomes()
        rows.append({"confusion": label, "primary": f"{primary:.1e}", "secondary": f"{secondary:.1e}", "galaxies": len(c),
                     "multi-line %": round(100 * o["MULTI_LINE"] / len(c), 2),
                     "interlopers %": round(100 * o["SINGLE_MISID"] / len(c), 3), "seconds": round(time.time() - t0)})
Table(rows)"""),
    md("""
Other cuts work the same way: `z_column="redshift_true"` to cut on another redshift, `n_det_min=2` for
multi-line galaxies only, `outcomes=None` to keep galaxies without a redshift, `cuts={"dec": (60, 70)}` on any
column, and `columns=[...]` or `columns="all"` (line fluxes, bitmask, `gal_id`, ...) for what is returned.
"""),
    code("""
o3 = base.make_catalog(flux_limit=1.0e-16, lines=["OIII_5007"], z_range=(2.0, 2.9), cuts={"dec": (60.0, 70.0)},
                       columns=["gal_id", "ra", "dec", "redshift_assigned", "id_outcome", "n_det", "line_bitmask",
                                "OIII_5007_flux", "Hb_flux"], seed=2026, verbose=False)
print(o3)
print("median [O III] 5007 flux:", f"{np.median(o3['OIII_5007_flux']):.2e}", "erg/s/cm²; detected lines per galaxy:",
      {int(k): int(v) for k, v in zip(*np.unique(o3["n_det"], return_counts=True))})"""),
    md("""
## Downsampling to the Wang et al. (2022) forecast

Off by default. `downsample="wang2022"` thins the Hα and [O III] emitter populations to the forecast at the
primary flux limit, in bins of true redshift, before the cuts are applied (so interlopers are thinned with the
population they belong to). The flux limit must be one of the tabulated ones (5e-17, 7e-17, 8.5e-17, 1e-16,
1.5e-16, 2e-16). `downsample={"reference": "wang2022", "dust_av": 1.92}` selects the other dust model.
"""),
    code("""
from romanmock.reference_dndz import wang2022_bins, wang2022_errors

kw = dict(flux_limit=1.5e-16, secondary_flux_limit=8.0e-17, confusion=confusion, lines=["Ha"], z_range=(1.0, 1.9),
          seed=2026, verbose=False)
full = base.make_catalog(**kw)
thin = base.make_catalog(downsample="wang2022", **kw)        # warns where the mock has fewer emitters than the forecast
print(thin.summary())

edges, ref = wang2022_bins("Ha", 1.5e-16)
err = wang2022_errors("Ha", 1.5e-16)
zc = 0.5 * (edges[1:] + edges[:-1])
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.step(zc, full.dndz(edges), where="mid", color=plotting.CATEGORICAL[0], label=f"no downsampling ({len(full):,})")
ax.step(zc, thin.dndz(edges), where="mid", color=plotting.CATEGORICAL[2], label=f'downsample="wang2022" ({len(thin):,})')
ax.errorbar(zc, ref, yerr=err, fmt="o", ms=3.5, color=plotting.TEXT_PRIMARY, label="Wang et al. (2022), true Hα emitters")
ax.set_xlim(0.95, 1.95); ax.set_yscale("log"); ax.set_ylim(300, None)
ax.set_xlabel("redshift_assigned"); ax.set_ylabel("dN/dz per deg²")
ax.set_title("Assigned Hα, F > 1.5e-16 erg/s/cm²"); ax.legend(fontsize=8); plt.show()"""),
    md("""
## Writing to disk

`cat.write(path)` writes `.hdf5` (one dataset per column, the settings in the file attributes), `.fits` or
`.npz`, replacing an existing file. Random catalogs come from `cat.make_random(which="all", n_times=10,
write_to=...)`: uniform inside the footprint, redshifts drawn from the data, one file per multiple of the
data size, for `which` = `"all"`, `"galaxies"` (correct redshift) or `"interlopers"` (wrong redshift).
`make_catalog(..., write_to=path)` does both in one call, and
`Catalog.read(path)` brings a catalog back with its settings.
"""),
    code("""
from romanmock import Catalog

path = thin.write(os.path.join(WORK, "two_step", "halpha_z1p0_1p9.hdf5"))
print(path, f"{os.path.getsize(path) / 1e9:.2f} GB")
back = Catalog.read(path)
print(back)
print({k: back.settings[k] for k in ("flux_limit", "secondary_flux_limit", "lines", "z_range", "downsample", "seed")})"""),
    md("""
## From the command line

```bash
romanmock catalog $PSCRATCH/romanmock_demo/two_step/base -o halpha.hdf5 \\
          --flux-limit 1.5e-16 --secondary-flux-limit 8e-17 --confusion my_confusion.yaml \\
          --lines Ha --z-range 1.0 1.9 --downsample wang2022
```

`my_confusion.yaml` holds the rows, e.g. `Ha: {OIII_5007: 0.2}`. Inside a script, call `make_catalog` below
`if __name__ == "__main__":` because the slices are read by worker processes.
"""),
]


NB10 = [
    md("""
# 10 — Random catalogs

A clustering measurement needs, for each data catalog, a random catalog with the same angular footprint and
the same redshift distribution but no clustering. `cat.make_random(...)` makes it for a catalog built with
`base.make_catalog(...)` (notebook 09).

- **Sky positions** are uniform inside the footprint of the lightcone. The footprint is exact: the lightcone
  is made of sky patches, rectangles in the angles θ and φ listed in `lc_cores-decomposition.txt`, and each
  galaxy carries its patch number (`lc_patch`).
- **Redshifts** are drawn, with replacement, from the redshifts of the data catalog.
- A contaminated catalog gives **three data catalogs**, each with its own randoms:

| `which` | data catalog | `id_outcome` |
|---|---|---|
| `"all"` | galaxies and interlopers | 1, 2, 3 |
| `"galaxies"` | galaxies with a correct redshift | 1 (multi-line), 2 (single-line, correct) |
| `"interlopers"` | galaxies with a wrong redshift | 3 (single-line, misidentified) |

Randoms N times as numerous as the data are N catalogs of the size of the data, one file each.
"""),
    code(SETUP),
    md("""
## The footprint
"""),
    code("""
from romanmock import BaseCatalog, Catalog

base = BaseCatalog(os.path.join(WORK, "two_step", "base"))
footprint = base.footprint          # built once from the mock, then cached in the base directory
print(footprint)
print("patches:", footprint.patches.tolist())
print(f"area: {footprint.area_deg2:.2f} deg2 (exact)   {base.area_deg2:.2f} deg2 (estimate from the galaxies, used for dN/dz)")"""),
    md("""
## A contaminated catalog and its three data catalogs

An Hα sample at 1.0 < z < 1.9. `id_outcome` must be among the columns (it is by default): it tells the
galaxies with a correct redshift from the interlopers.
"""),
    code("""
confusion = {
    "Ha":        {"OIII_5007": 0.2},
    "OIII_5007": {"Ha": 0.2},
    "SIII_9532": {"Ha": 0.2},
    "Hb":        {"OIII_5007": 0.2},
}
cat = base.make_catalog(flux_limit=1.5e-16, secondary_flux_limit=8.0e-17, confusion=confusion,
                        lines=["Ha"], z_range=(1.0, 1.9), seed=2026)"""),
    code("""
data = {which: cat.subset(which) for which in ("all", "galaxies", "interlopers")}
for which, d in data.items():
    z_true, z = d["redshift_rsd"], d["redshift_assigned"]
    print(f"{which:<12s} {len(d):>11,} galaxies   true redshift {z_true.min():.3f} - {z_true.max():.3f}   "
          f"assigned redshift {z.min():.3f} - {z.max():.3f}")"""),
    md("""
## Randoms in memory

One call for the three kinds; `which="all"` alone returns just the list. Each random catalog has the size of
its data catalog and the columns `ra`, `dec` and the redshift column of the redshift cut.
"""),
    code("""
%%time
randoms = cat.make_random(which=["all", "galaxies", "interlopers"], n_times=2, seed=7)"""),
    code("""
from astropy.table import Table

ran = randoms["all"][0]
print(ran)
print("settings['random']:", ran.settings["random"])
t = Table({c: ran[c][:6] for c in ran.columns})
for c in t.colnames:
    t[c].format = ".5f"
t"""),
    md("""
## On the sky

Number of objects per square degree in cells of right ascension and declination, for the data and for one
random catalog. The outlines are the 60 patches of the footprint. The data show large-scale structure, the
randoms only shot noise; both stop at the same edges.
"""),
    code("""
ra_edges = np.linspace(0.0, 360.0, 241)
sin_edges = np.linspace(np.sin(np.radians(57.0)), np.sin(np.radians(76.0)), 61)      # equal-area cells
cell = np.degrees(1.0) ** 2 * np.radians(np.diff(ra_edges))[:, None] * np.diff(sin_edges)[None, :]
dec_edges = np.degrees(np.arcsin(sin_edges))

def density(c):
    return np.histogram2d(c["ra"], np.sin(np.radians(c["dec"])), bins=(ra_edges, sin_edges))[0] / cell

maps = {"data: galaxies and interlopers": density(cat), "randoms (one catalog)": density(ran)}
vmax = np.percentile(maps["data: galaxies and interlopers"], 99.5)
fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, constrained_layout=True)
for ax, (title, m) in zip(axes, maps.items()):
    im = ax.pcolormesh(ra_edges, dec_edges, np.where(m > 0, m, np.nan).T, cmap=plotting.sequential_cmap(), vmin=0, vmax=vmax,
                       rasterized=True)
    for lo, hi, d_lo, d_hi in footprint.rectangles():
        ax.plot([lo, hi, hi, lo, lo], [d_lo, d_lo, d_hi, d_hi, d_lo], color=plotting.TEXT_PRIMARY, lw=0.5)
    ax.set_title(title, loc="left")
    ax.set_ylabel("dec [deg]")
axes[-1].set_xlabel("ra [deg]")
fig.colorbar(im, ax=axes, label="objects per deg$^2$", shrink=0.8)
plt.show()"""),
    code("""
inside = footprint.contains(cat["ra"], cat["dec"])
print(f"randoms inside the footprint: {footprint.contains(ran['ra'], ran['dec']).mean():.6f}")
print(f"data inside the footprint:    {inside.mean():.6f}   ({np.sum(~inside):,} galaxies outside: satellites of haloes at the edge)")

d, r = density(cat) * cell, density(ran) * cell
centre_ra, centre_dec = np.meshgrid(0.5 * (ra_edges[1:] + ra_edges[:-1]), 0.5 * (dec_edges[1:] + dec_edges[:-1]), indexing="ij")
corners = [footprint.contains(a[:, None], b[None, :]) for a in (ra_edges[:-1], ra_edges[1:]) for b in (dec_edges[:-1], dec_edges[1:])]
interior = np.all(corners, axis=0) & footprint.contains(centre_ra, centre_dec)      # cells with the four corners inside
edge = (r > 0) & ~interior
print(f"data / randoms in the {interior.sum():,} cells inside the footprint: {d[interior].sum() / r[interior].sum():.4f}")
print(f"data / randoms in the {edge.sum():,} cells on its edge:            {d[edge].sum() / r[edge].sum():.4f}")"""),
    md("""
If the galaxies outside the footprint matter for your measurement, remove them with
`cat = cat[footprint.contains(cat["ra"], cat["dec"])]` before making the randoms.

## In redshift

Each random catalog follows the redshift distribution of its own data catalog. The interlopers have a
distribution of their own: the Hα sample contains [O III] 5007 emitters from z = 1.94–2.80 and [S III] 9532
emitters from z = 0.38–1.00, placed at the redshift their line would have as Hα. The [S III] interlopers
drop at an assigned redshift of 1.21: above z = 0.52 their own Hα line is inside the grism, which makes most of
them multi-line galaxies with a correct redshift. The [O III] interlopers start at an assigned redshift of 1.24.
"""),
    code("""
z_col = cat.settings["z_column"]
edges = np.linspace(1.0, 1.9, 37)
zc = 0.5 * (edges[1:] + edges[:-1])
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
for ax, which, color in zip(axes, data, plotting.CATEGORICAL):
    d = data[which].dndz(edges, z_col)
    r = np.mean([x.dndz(edges, z_col) for x in randoms[which]], axis=0)
    ax.step(zc, d, where="mid", color=color, lw=1.6, label=f"data ({len(data[which]):,})")
    ax.plot(zc, r, "o", ms=3, color=plotting.TEXT_PRIMARY, label=f"randoms (mean of {len(randoms[which])})")
    ax.set_title(which, loc="left")
    ax.set_xlabel("assigned redshift")
    ax.set_ylim(0, None)
    ax.legend(frameon=False)
axes[0].set_ylabel("dN/dz per deg$^2$")
plt.show()"""),
    md("""
## Randoms on disk: N files

With `write_to` the randoms are written one after the other instead of being returned, so that a large
multiple of the data never sits in memory at once. `name.hdf5` gives `name_<which>_00.hdf5`,
`name_<which>_01.hdf5`, ... (without `<which>` when a single kind is asked for).
"""),
    code("""
del randoms
out = os.path.join(WORK, "two_step", "randoms", "halpha_z1p0_1p9.hdf5")
cat.write(out)                                                                   # the data catalog, next to its randoms
files = cat.make_random(which=["all", "galaxies", "interlopers"], n_times=2, seed=7, write_to=out)
for which, paths in files.items():
    print(which, [f"{p.name} ({os.path.getsize(p) / 1e6:.0f} MB)" for p in paths])"""),
    code("""
back = Catalog.read(files["interlopers"][1])
print(back)
print(back.settings["random"], "| footprint area", round(back.area_deg2, 2), "deg2")"""),
    md("""
Random catalog `k` of a kind depends on `(seed, which, k)` only. It can be made again on its own, and more
can be added later with `first=` without remaking the existing ones.
"""),
    code("""
again = cat.make_random(which="interlopers", n_times=1, first=1, seed=7, verbose=False)[0]
print("catalog 1 made again equals the file:", all(np.array_equal(again[c], back[c]) for c in back.columns))

more = cat.make_random(which="interlopers", n_times=2, first=2, seed=7, write_to=out.replace(".hdf5", "_interlopers.hdf5"))
print([p.name for p in more])"""),
    md("""
## Cuts on the sky

A cut on `ra` or `dec` made with `cuts=` in `make_catalog` is applied to the randoms as well.
"""),
    code("""
part = base.make_catalog(flux_limit=1.5e-16, secondary_flux_limit=8.0e-17, confusion=confusion, lines=["Ha"],
                         z_range=(1.0, 1.9), cuts={"ra": (200.0, 300.0), "dec": (60.0, 70.0)}, seed=2026, verbose=False)
r = part.make_random(n_times=1)[0]
print(f"data:    {len(part):,}  ra {part['ra'].min():.2f} - {part['ra'].max():.2f}  dec {part['dec'].min():.2f} - {part['dec'].max():.2f}")
print(f"randoms: {len(r):,}  ra {r['ra'].min():.2f} - {r['ra'].max():.2f}  dec {r['dec'].min():.2f} - {r['dec'].max():.2f}")"""),
    md("""
## From the command line

```bash
romanmock catalog /path/to/base -o halpha.hdf5 --flux-limit 1.5e-16 --secondary-flux-limit 8e-17 \\
          --confusion my_confusion.yaml --lines Ha --z-range 1.0 1.9
romanmock random halpha.hdf5 -n 10 --which all galaxies interlopers -o randoms/halpha.hdf5
```

## Notes

- Redshifts drawn from the data carry the radial structure of the data into the randoms, which slightly
  suppresses the clustering of the modes along the line of sight. The effect is small for 2063 deg².
- The randoms of `"all"` are not the sum of the randoms of `"galaxies"` and `"interlopers"`: each set is drawn
  on its own, from the redshifts of its own data catalog.
"""),
]


NOTEBOOKS = {
    "01_explore_diffsky_catalog.ipynb": NB01,
    "02_pipeline_step_by_step.ipynb": NB02,
    "03_run_and_build_samples.ipynb": NB03,
    "04_extending_models.ipynb": NB04,
    "05_full_run.ipynb": NB05,
    "06_dndz_census.ipynb": NB06,
    "07_two_step_workflow.ipynb": NB07,
    "08_dndz_secondary_limit.ipynb": NB08,
    "09_catalog_from_base.ipynb": NB09,
    "10_random_catalogs.ipynb": NB10,
}


def build(execute: bool, timeout: int = 1800, only: list[str] | None = None):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    for name, cells in NOTEBOOKS.items():
        if only and not any(name.startswith(o) for o in only):
            continue
        nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                                  "language_info": {"name": "python"}})
        path = HERE / name
        if execute:
            from nbclient import NotebookClient

            client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
            client.execute()
        nbformat.write(nb, path)
        print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--only", nargs="*", help="notebook name prefixes, e.g. 03 04")
    args = ap.parse_args()
    build(args.execute, args.timeout, args.only)

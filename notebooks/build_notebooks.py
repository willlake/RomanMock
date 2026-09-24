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
from romanmock.roman import DEFAULT_LINE_FLUX_LIMIT

fig, ax = plt.subplots()
bins = np.linspace(-19, -14.5, 70)
for name in ["Ha", "OIII_5007", "OII_3727"]:
    ax.hist(np.log10(flux[f"{name}_flux"]), bins=bins, histtype="step", color=plotting.feature_color(name), label=name)
ax.axvline(np.log10(DEFAULT_LINE_FLUX_LIMIT), color=plotting.TEXT_SECONDARY, lw=1, ls="--")
ax.text(np.log10(DEFAULT_LINE_FLUX_LIMIT) + 0.05, ax.get_ylim()[1] * 0.95, f"{DEFAULT_LINE_FLUX_LIMIT:.1e} erg/s/cm² (default limit)", color=plotting.TEXT_SECONDARY, fontsize=8, va="top")
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
`ConstantFluxLimit` is the plan's initial model. Bit *i* of the bitmask is feature *i* of the
line set; `n_det = popcount(bitmask)`.
'''),
    code('''
from romanmock import ConstantFluxLimit, ThresholdDetection, detect_lines, unpack_bitmask

rng = np.random.default_rng(0)
from romanmock.roman import DEFAULT_LINE_FLUX_LIMIT
model = ThresholdDetection(ConstantFluxLimit(flux_limit=DEFAULT_LINE_FLUX_LIMIT, wave_range=(10000.0, 19300.0)))
det = detect_lines(flux, z_rsd, line_set, model, rng)

print("bitmask dtype", det.bitmask.dtype, "| example bitmasks", det.bitmask[:8])
print("n_det       ", det.n_det[:8])
print("bit 0 (Ha) of the first 8:", unpack_bitmask(det.bitmask[:8], len(line_set))[:, 0].astype(int))'''),
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
from romanmock.roman import DEFAULT_LINE_FLUX_LIMIT as FLUX_LIMIT

cfg = RomanMockConfig.from_dict({
    "line_set": "roman_transitions",
    "sensitivity": {"type": "constant", "flux_limit": FLUX_LIMIT, "wave_range": [10000.0, 19300.0]},
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
from romanmock.roman import DEFAULT_LINE_FLUX_LIMIT as FLUX_LIMIT

cfg = RomanMockConfig.from_dict({
    "catalog": {"path": CATALOG, "steps": [148, 219, 224]},   # the intended parent catalog: three slices
    "output": {"dir": os.path.join(WORK, "parent"), "overwrite": False, "store_fluxes": True,   # fluxes are optional; kept here to show them
               "passthrough_columns": ["gal_id", "ra", "dec", "redshift_true", "logsm_obs", "roman_F158"]},
    "line_set": "roman_transitions",
    "sensitivity": {"type": "constant", "flux_limit": FLUX_LIMIT, "wave_range": [10000.0, 19300.0]},
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
sample_files = pipe.build_samples(samples=samples)
sample_files'''),
    code('''
ds_sample = oc.open(sample_files[1])
zs = ds_sample.select("redshift_assigned").get_data("numpy")
fig, ax = plt.subplots()
ax.step(zc, mock, where="mid", color=plotting.CATEGORICAL[0], label="mock, assigned Hα in box")
ax.plot(zc, target["dndz"], color=plotting.CATEGORICAL[1], label="target")
ax.step(zc, dndz(zs, edges, fp_area), where="mid", color=plotting.CATEGORICAL[2], label="downsampled sample")
ax.set_xlabel("redshift_assigned"); ax.set_ylabel("dN/dz per deg²"); ax.set_title(f"Downsampling to a target dN/dz (footprint {fp_area:.0f} deg²)")
ax.legend(); plt.show()
print(open(os.path.join(WORK, "parent", "samples", "halpha_box_downsampled_metadata.yaml")).read()[-400:])'''),
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
with `file:` for a two-column text file).
'''),
    code('''
wave = np.linspace(10000, 19300, 50)
curve = 1e-16 * (1 + 2.0 * ((wave - 14650) / 4650) ** 2)      # toy: worse sensitivity towards the band edges
tab = TabulatedFluxLimit(wave, curve)

fig, ax = plt.subplots()
w = np.linspace(9500, 20000, 300)
ax.plot(w / 1e4, ConstantFluxLimit(1e-16).flux_limit(w) * 1e16, color=plotting.CATEGORICAL[0], label="constant")
ax.plot(w / 1e4, tab.flux_limit(w) * 1e16, color=plotting.CATEGORICAL[1], label="tabulated (toy)")
ax.set_ylim(0, 4); ax.set_xlabel("observed wavelength [μm]"); ax.set_ylabel("flux limit [1e-16 erg/s/cm²]")
ax.set_title("Sensitivity models"); ax.legend(); plt.show()

for name, model in [("constant", ConstantFluxLimit(1e-16)), ("tabulated", tab)]:
    det = detect_lines(flux, z_rsd, line_set, ThresholdDetection(model))
    print(f"{name:<10s} detected fraction (any line): {np.mean(det.n_det > 0):.3f}")'''),
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
from romanmock.roman import DEFAULT_LINE_FLUX_LIMIT as FLUX_LIMIT

from romanmock.io import memory_budget_gb, suggest_n_workers

# one slice per process, ~4.5 GB peak each: size the pool by the memory this session may use
# (node memory, or the per-user cgroup cap on a NERSC login node)
N_WORKERS = suggest_n_workers(max_workers=32)
print(f"memory budget {memory_budget_gb():.0f} GB -> {N_WORKERS} worker(s)")
OUT_DIR = os.path.join(WORK, "full")

cfg = RomanMockConfig.from_dict({
    "catalog": {"path": CATALOG},                        # every slice, z = 0-4
    "output": {"dir": OUT_DIR, "overwrite": False,
               "passthrough_columns": ["ra", "dec", "redshift_true"]},          # plus redshift_rsd, redshift_assigned, id_outcome, identification columns
    "line_set": "roman_transitions",
    "sensitivity": {"type": "constant", "flux_limit": FLUX_LIMIT, "wave_range": [10000.0, 19300.0]},
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
ax.set_xlabel("redshift_assigned"); ax.set_ylabel("dN/dz per deg²"); ax.set_title(f"Assigned-line samples, F > {FLUX_LIMIT:.1e} erg/s/cm², 1.00–1.93 μm")
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
run's flux limit (the package default, 1.5e-16) and the A_V = 1.65 dust model.
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
sample_files = pipe.build_samples(samples=samples)
print(f"{(time.time() - t0) / 60:.1f} min")
sample_files"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, path, name in zip(axes, sample_files, ["Ha", "OIII_5007"]):
    d = oc.open(path).select("redshift_assigned", "redshift_rsd", "id_outcome", "line_true").get_data("numpy", wrap_single=True)
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
d = oc.open(sample_files[1]).select("redshift_assigned", "redshift_rsd", "line_true", "id_outcome").get_data("numpy", wrap_single=True)
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
      f"x C(Hα→[O III]) = {c_ha_o3:.2f} -> {n_src * c_ha_o3:,.0f} interlopers before thinning, versus {n_true.sum():,} true [O III] emitters kept")"""),
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

Solid steps: the mock. Dashed steps with shaded ±1σ bands: Wang et al. (2022) Tables 1–2 (Galacticus, A_V = 1.65)
for Hα and [O III] 5007 at the same flux limit; the tabulated uncertainties combine jackknife resampling and cosmic
variance. Every galaxy with the line above the limit is counted, whatever its other lines do. All nine lines of
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
        edges, ref = wang2022_bins(name, lim)
        err = wang2022_errors(name, lim)
        zc_ref = 0.5 * (edges[1:] + edges[:-1])
        ax.step(zc_ref, ref, where="mid", color=plotting.feature_color(name), ls="--", lw=1.2, label=f"{name} Wang+22 ±1σ")
        ax.fill_between(edges, np.append(ref - err, ref[-1] - err[-1]), np.append(ref + err, ref[-1] + err[-1]),
                        step="post", color=plotting.feature_color(name), alpha=0.18, lw=0)
    ax.set_yscale("log"); ax.set_ylim(10, None); ax.set_xlim(Z_EDGES[0], Z_EDGES[-1])
    ax.set_xlabel("redshift"); ax.set_title(f"F > {lim:.1e} erg/s/cm²")
    ax.legend(fontsize=7.5, ncol=2)
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


NOTEBOOKS = {
    "01_explore_diffsky_catalog.ipynb": NB01,
    "02_pipeline_step_by_step.ipynb": NB02,
    "03_run_and_build_samples.ipynb": NB03,
    "04_extending_models.ipynb": NB04,
    "05_full_run.ipynb": NB05,
    "06_dndz_census.ipynb": NB06,
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

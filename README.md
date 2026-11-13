# romanmock

Build a Roman-like emission-line spectroscopic mock from a diffsky lightcone. Catalog access goes through
[OpenCosmo](https://github.com/ArgonneCPAC/OpenCosmo) and the
[diffsky](https://github.com/ArgonneCPAC/diffsky) loaders; the package adds only the
survey-specific steps (fluxes, detection, bitmask, redshift identification, sample selection).

Version 0.3.0. `CLAUDE.md` summarises the layout and conventions for anyone (or any assistant)
editing the package.

## Installation

```bash
# cd dir-to-RomanMock
# pip install -e .            # opencosmo, hdf5plugin, astropy, pyyaml, ... ; diffsky/dsps optional
```

On NERSC login nodes set `export JAX_PLATFORMS=cpu` (diffsky imports JAX; the CLI does
this by default).

## Input

A diffsky mock directory in OpenCosmo format, such as
`/global/cfs/cdirs/hacc/OpenCosmo/LastJourney/synthetic_galaxies/sdss_feniks_hizels_261013_10_14_2026/` (the
production mock since 2026-10-29): 67 lightcone slices `lc_cores-<step>.diffsky_gals.hdf5` (z = 0-4, 413 M
galaxies, 2578 deg², ray-traced lensing) plus the auxiliary `*_ssp_data.hdf5`, `*_param_collection.hdf5`,
`*_t_table.hdf5`, `*_transmission_curves.hdf5` and `lc_cores-decomposition.txt`. The previous mock,
`/global/cfs/cdirs/m4943/Simulations/sdss_feniks_hizels_260710_09_07_2026/` (324 M galaxies, 2063 deg², no
lensing), reads the same way.
The line columns (`Ba_alpha_6563`, `Ba_beta_4861`, `O_III_4363`, `O_III_4959`, `O_III_5007`,
`O_II_3726`, `O_II_3729`, `S_III_6312`, `S_III_9069`, `S_III_9532`) are dust-attenuated,
continuum-subtracted luminosities in erg/s, one column per transition. In the lensed mock they include the
magnification (`<line>_unlensed` x `magnification`); `ra`/`dec` are the unlensed positions and `ra_obs`/`dec_obs`
the lensed ones. Its synthetic cores (777 M centrals of log Mhalo 11.0–11.5 at random, unclustered positions)
are left out, as OpenCosmo does unless it is opened with `synth_cores=True`.
`redshift_obs` includes the line-of-sight peculiar velocity (`z_rsd` of the plan);
`redshift_true` is the cosmological redshift. The cosmology is read from the file header
(h = 0.6766, Ω_m = 0.3096, the Planck 2018 cosmology of the parent N-body simulation) and is
used for the luminosity distances.

`romanmock info <catalog_dir>` lists the slices, their redshift ranges and the cosmology.

The package also reads the Galacticus lightcone mocks of the Roman GRS-PIT (`romanmock.galacticus`, see
[Galacticus mocks](#galacticus-mocks) below).

## Quick start: the two-step workflow

Reading the 340 GB input mock is the expensive part and does not depend on the things one wants to
vary. So the mock is read once, and every catalog is made from the result.

**Step 1, once: the base catalog.** Every galaxy with at least one line above the base flux limit
(`sensitivity.flux_limit`, default 1.0e-16 erg/s/cm²) inside the bandpass, with `gal_id`, `ra`, `dec`,
`redshift_true`, `redshift_rsd` and the flux of every line of the line set.

```bash
sbatch scripts/production.sh                  # configs/production.yaml: base catalog and dN/dz census, about 5 minutes
romanmock base cfg.yaml --workers 32          # the base catalog alone
```

`cfg.yaml` needs `catalog.path`, `base.dir` and, to change the defaults, `sensitivity` and `base.columns`. Add
`r50_disk_2d` to `base.columns` (as `configs/production.yaml` does) to use the size of each galaxy in step 2.
`configs/production.yaml` is the configuration of the production run, `configs/roman_grism_baseline.yaml` a
commented template of every option.

**Step 2, as often as you like: a catalog.** One call with the flux limits, the confusion matrix, the lines and
the redshift range of your choice. It returns numpy arrays; nothing is written unless you ask, and nothing is
remembered between calls. A call takes 15 to 20 s on a login node for the full lightcone.

```python
from romanmock import BaseCatalog

base = BaseCatalog("/path/to/base")
cat = base.make_catalog(
    flux_limit=1.5e-16,                      # primary limit: at least one line of the galaxy above it
    secondary_flux_limit=8e-17,              # limit for the other lines of such a galaxy (None: no second tier)
    confusion={"Ha": {"OIII_5007": 0.2},     # single-line galaxies: {true line: {assigned line: probability}}
               "OIII_5007": {"Ha": 0.2}},    # None: the package default; "empirical": the line mix of the mock (below)
    lines=["Ha"],                            # galaxies in which Ha is identified
    z_range=(1.0, 1.9),                      # range of redshift_assigned
    seed=2026,
)
cat["ra"], cat["redshift_assigned"], len(cat)     # numpy arrays
print(cat.summary())                              # counts per identification outcome
cat.write("halpha.hdf5")                          # optional: .hdf5, .fits or .npz; Catalog.read() brings it back
```

The thresholds can also be S/N values in the Roman grism instead of flux limits (the second way of building a
catalog, see *S/N thresholds* below): a galaxy needs one line with S/N ≥ 5 and its other lines are detected with
S/N ≥ 2, where the noise depends on the observed wavelength and on the angular size of the galaxy, either one size
for all or the size of each galaxy from the mock.

```python
cat = base.make_catalog(snr=5, secondary_snr=2, size="r50_disk_2d", lines=["Ha"], z_range=(1.0, 1.9))
cat = base.make_catalog(snr=5, secondary_snr=2, size=0.3)        # 0.3 arcsec for every galaxy; 0 = point source
```

| argument | meaning | default |
|---|---|---|
| `flux_limit` | primary limit; cannot be deeper than the base catalog | the base catalog's limit |
| `secondary_flux_limit` | limit for the other lines of a detected galaxy; `None` = no second tier | 4.0e-17 |
| `snr`, `secondary_snr` | instead of the flux limits: S/N of the primary line and of the other lines (`None` = no second tier) in the Roman grism | 5, 2 |
| `size` | angular size of the galaxies for the S/N: arcsec for all, or a column of the base catalog in proper kpc (`"r50_disk_2d"`, must be in `base.columns`) | 0.3 |
| `exposure_time`, `background` | exposure [s] and background in units of the minimum zodiacal light, rescaling the noise | 3600, 2 |
| `sensitivity` | any sensitivity model config or object, instead of the arguments above | |
| `confusion` | rows of the confusion matrix, a full config, or a model object; `"empirical"` or a `LineFractionTable` for the line mix of the mock (below) | package default |
| `rescaling` | rescale the line fluxes before the detection: `"galacticus_like"` (the diffsky lines toward Galacticus) or any rescaling config; the catalog's flux columns are then the rescaled ones (see *Rescaling the line fluxes*) | stored fluxes |
| `doublets` | [O III] λ4959 a line of its own (`"separated"`), added to λ5007 (`"merged"`), or each where its doublet is resolved for the galaxy's size (`"per_galaxy"`, needs `r50_disk_2d`); see *The [O III] doublet* | separated |
| `lines` | keep galaxies in which one of these lines is identified | every line |
| `membership` | multi-line galaxies: `"detected"` = each detected line is identified (a galaxy with Hα and [O III] is in both samples); `"assigned"` = only the brightest line, disjoint samples. `romanmock select` defaults to `assigned` | `"detected"` |
| `z_range`, `z_column` | redshift range, on `redshift_assigned`, `redshift_rsd` or `redshift_true` | no cut, assigned |
| `outcomes` | identification outcomes to keep; `None` = all, including failed | every galaxy with a redshift |
| `n_det_min`, `cuts` | minimum number of detected lines; `{column: (min, max)}` on any column | none |
| `downsample` | `"wang2022"` thins the emitter populations to the Wang et al. (2022) forecast | no downsampling |
| `columns` | columns returned, or `"all"` (line fluxes, bitmask, `gal_id`, ...) | ra, dec, three redshifts, id_outcome |
| `wave_range`, `detection`, `sigma_z`, `seed`, `n_workers`, `write_to` | bandpass, detection model, redshift error, seed, workers, output file | |
| `area_deg2` | sky area of the lightcone, for dN/dz and the downsampling; also `BaseCatalog(path, area_deg2=...)` | estimated |
| `verbose` | print the progress and the line report (below) | `True` |

**What it prints.** With `verbose=True` the call prints a progress line, `[romanmock] catalog: scanning 67 slices
with 64 worker(s) … done in 28 s`, then two blocks:

* **the summary** (`cat.summary()`): thresholds, line fluxes, identification (for a confusion matrix, its entries),
  sky area, base galaxies detected, columns, and the galaxies per identification outcome;
* **the line report** (`cat.line_report()`, the numbers in `cat.info["line_report"]`): the lines inside the
  bandpass for the redshift cut, and the line interlopers, single-line galaxies given a wrong line. It lists the
  potential ones (a line that, taken for another, passes the redshift cut, with its probability `P` in the
  confusion matrix) next to the number that is actually in the catalog.

`make_proposal`, `line_fractions` (the line mix) and `Proposal.make_catalog` print in the same layout
(`romanmock/report.py`): a title rule, aligned settings, tables with a rule under the header, and notes with
footnote marks, at most 100 characters wide.

| column | galaxies of the catalog in which |
|---|---|
| detected | the line is detected |
| identified | it is detected and identified as this line: the redshift of the galaxy is right. Every detected line of a multi-line galaxy counts, also a secondary one |
| brightest | it is identified and is the brightest detected line (`line_assigned`) |
| mistaken | it is the only detected line and was taken for another line |
| interlopers | another line was taken for this line |

detected = identified + mistaken. The sample of a line is its identified galaxies plus its interlopers, and
`share` is the share of one interloper in it. For the Hα catalog at 1.0–1.2 made as in
`notebooks/01_catalog_z1p0_1p2.ipynb`, here with the default `"galacticus_like"` rescaling (the notebook passes
another one, so its numbers differ):

```text
━━ Catalog · Hα · 1.0 ≤ z_assigned ≤ 1.2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  thresholds       F > 1.50e-16 (other lines > 6.00e-17)
  line fluxes      rescaled (galacticus_like); [O III] doublet per_galaxy
  identification   confusion matrix: Ha → OIII_5007 0.005, Ha → SIII_9532 0.005, OIII_5007 → Ha
                   0.05, SIII_9532 → Ha 0.05
  sky area         2578.31 deg² (estimated from the sky patches of the mock)
  base catalog     100,805,121 of its 403,068,868 galaxies detected
  ...
  galaxies                                       number     share
  ─────────────────────────────────────────  ──────────  ────────
  multi-line                                 14,845,961   99.21 %
  single line, correct redshift                 114,349    0.76 %
  single line taken for another: interloper       4,040    0.03 %
  all                                        14,964,350  100.00 %

── Lines in the bandpass, 10000–19300 Å, for 1.0 ≤ z_assigned ≤ 1.2 ────────────────────────────────
  line         redshift     λ_obs [Å]      detected  identified  brightest  mistaken  interlopers
  ───────────  ───────────  ───────────  ──────────  ──────────  ─────────  ────────  ───────────
  Ha           1.000–1.200  13129–14442  14,960,310  14,960,310  7,250,586         0        4,040
  OIII_5007 ○  1.000–1.200  10017–11018  14,816,182  14,814,785  7,709,724     1,397            0
  ...

── Interlopers: single lines given a wrong line, and so a wrong redshift ───────────────────────────
  true line  taken for  true z       assigned z   λ_obs [Å]       P¹  in catalog  share²
  ─────────  ─────────  ───────────  ───────────  ───────────  ─────  ──────────  ──────
  OIII_5007  Ha         1.622–1.884  1.000–1.200  13129–14442  0.050       1,397  0.01 %
  SIII_9532  Ha         0.377–0.515  1.000–1.200  13129–14442  0.050       2,643  0.02 %
  ...
```

The assigned redshift of a galaxy with more than one detected line is `redshift_rsd` (the true redshift
including the peculiar velocity), never `redshift_true`. For a single-line galaxy it follows from the line
identity drawn from the confusion matrix, or from the line mix of the mock.

**Identifying single lines from the line mix of the mock.** A confusion matrix states, per true line, how
often it is taken for another; the numbers are assumed. The alternative is to let the mock say what a
single line at a given observed wavelength and flux is likely to be. `confusion="empirical"` builds a
`LineFractionTable` from the base catalog at the flux limits of the catalog: the single-line galaxies are
binned in the observed wavelength of their line (bins of `wave_bin`, default 250 Å, about twelve resolution
elements of the grism, which resolves 21–24 Å at R ≈ 475–900) and, inside each wavelength bin, ranked by
flux and cut into bins of equal count: as many as hold at least `galaxies_per_bin` galaxies each (default
500, so every cell has 500 to 999 galaxies and a populated wavelength bin gets finer flux bins than a sparse
one), or exactly `n_flux_bins`, or the bins between the quantiles `flux_quantiles` (e.g. `[0, 0.5, 0.9,
0.99, 1]`, to resolve the bright tail where the rare lines sit). Each cell records the
share of each line among its galaxies; lines that are the single line of fewer than `min_count` galaxies in
all (default 100) are left out and the shares renormalised. A single-line galaxy is then given a line with
the shares of its own cell (`mode="draw"`) or the most common line of its cell (`mode="max"`), whatever its
true line, so the interlopers are the galaxies whose line is a minority at their wavelength and flux. The
table costs one more pass over the base catalog; build it once and reuse it:

```python
table = base.line_fractions(flux_limit=1.5e-16, secondary_flux_limit=6e-17,    # same limits as the catalog
                            wave_bin=250.0, galaxies_per_bin=500, min_count=100,
                            write_to="line_fractions.npz")                     # optional
print(table.summary())                      # galaxies and shares per wavelength bin, the smallest cell
table.expected_confusion()                  # the confusion matrix it implies, averaged over the single-line galaxies of each line
cat = base.make_catalog(flux_limit=1.5e-16, secondary_flux_limit=6e-17, confusion=table, lines=["Ha"], z_range=(1.0, 1.9))
cat = base.make_catalog(..., confusion="empirical")                                  # builds the table first
cat = base.make_catalog(..., confusion={"type": "empirical", "galaxies_per_bin": 1000, "mode": "max"})
cat = base.make_catalog(..., confusion={"type": "empirical", "file": "line_fractions.npz"})
```

`romanmock line-fractions BASE -o line_fractions.npz --flux-limit 1.5e-16 --secondary-flux-limit 6e-17` writes
the table from the command line, `romanmock catalog ... --confusion line_fractions.npz` (or `--confusion
empirical`) uses it, and `confusion: {type: empirical, file: line_fractions.npz}` selects it in a pipeline
configuration. `make_catalog` warns when the table was built at other flux limits than the catalog uses. The
line report then quotes, as `P`, the expected probability over the single-line galaxies of the true line.
`cat.settings["confusion"]` keeps how the table was built and a hash of its content, not the numbers.

The same from the command line, writing a file:

```bash
romanmock catalog /path/to/base -o halpha.hdf5 --flux-limit 1.5e-16 --secondary-flux-limit 8e-17 \
          --confusion my_confusion.yaml --lines Ha --z-range 1.0 1.9 --downsample wang2022
```

Notebook 02 walks through this. In a script, call `make_catalog` below `if __name__ == "__main__":` because
the slices are read by worker processes.

**Random catalogs.** A random catalog belongs to one data catalog and depends on nothing else: uniform on the sky
inside the footprint of the lightcone (and inside the data's cuts on `ra` and `dec`), with redshifts drawn, with
replacement, from the redshifts of the data. `make_random` takes the data catalog only, so it works for any catalog, a
whole one or any part of one:

```python
ran = cat.make_random(10)                                   # 10 x len(cat) points: ra, dec, redshift_assigned
ran = cat.subset("OIII_5007_as_Ha").make_random(10)         # the randoms of one kind of interloper
ran = cat[cat["redshift_assigned"] < 1.1].make_random(10)   # of any selection
cat.subset("interlopers").make_random(10, write_to="randoms/halpha_interlopers.hdf5")   # one file, written block by block
ran = Catalog.read("randoms/halpha_interlopers.hdf5")
```

`cat.subset(which)` takes the parts of a contaminated catalog, and `cat.components()` lists those present with their
sizes:

| `which` | the part | rows |
|---|---|---|
| `"all"` | the whole catalog | every galaxy |
| `"multi_line"` | multi-line galaxies | `id_outcome` 1 |
| `"single_correct"` | single-line galaxies with a correct redshift | `id_outcome` 2 |
| `"galaxies"` | galaxies with a correct redshift | `id_outcome` 1, 2 |
| `"interlopers"` | galaxies with a wrong redshift | `id_outcome` 3 (single-line, misidentified) |
| `"OIII_5007_as_Ha"` | one kind of interloper: [O III] λ5007 single lines taken for Hα | `id_outcome` 3 with `line_true`, `line_assigned` |

The points come in blocks of the size of the data, block `k` from the random stream `(seed, hash of the data's
redshifts, k)`. The same data and seed therefore always give the same randoms, a larger `n_times` adds points and keeps
the first ones, and two different data catalogs (two parts of one catalog) get independent randoms with the same seed.
From the command line: `romanmock random halpha.hdf5 -n 10 [--subset OIII_5007_as_Ha] [-o out.hdf5]`. Notebook
`notebooks/03_randoms.ipynb` makes them for the whole catalog and for its parts.

The footprint is exact, not estimated from the galaxies. The lightcone is made of sky patches, rectangles in
the angles theta and phi listed in `lc_cores-decomposition.txt`, and each galaxy carries its patch number
(`lc_patch`). The mock holds 60 patches, 2062.65 deg² (`base.footprint`, a `PatchFootprint` with `sample`,
`contains` and `area_deg2`). 0.006 % of the galaxies lie outside it: satellites of haloes near the edge.

**The file-based route.** `romanmock assign` (step 2 written as OpenCosmo files, one per slice) followed by
`romanmock select` (analysis samples) does the same through a YAML configuration, with manifests and
configuration hashes that refuse to mix different settings in one directory. It gives identical galaxies for
identical settings and seed (tested) and is the route to take when the output must be an OpenCosmo lightcone.
`romanmock run cfg.yaml` does stage A in one pass from the input mock.

## Pipeline

| Plan step | Module | What it does |
|---|---|---|
| fluxes | `romanmock.fluxes`, `romanmock.cosmology` | `F = L / (4 pi D_L^2)` with the catalog cosmology from OpenCosmo (`astropy` backend) or `dsps.cosmology.luminosity_distance_to_z` (`dsps` backend; agree to 5e-6) |
| 1. detectability | `romanmock.sensitivity`, `romanmock.detection`, `romanmock.sizes` | `lambda_obs = lambda_0 (1+z_rsd)`; `ConstantFluxLimit` (primary limit 1.0e-16 erg/s/cm², secondary limit 4.0e-17), `TabulatedFluxLimit` (wavelength dependent, `secondary_ratio`) or `RomanGrismSensitivity` (S/N thresholds 5 and 2; noise from the WFI grism tables against wavelength and source size, exposure time and background); `ThresholdDetection` or `GaussianNoiseDetection` |
| 2. bitmask | `romanmock.lines`, `romanmock.detection` | bit *i* = feature *i* of a `LineSet`; `n_det = popcount`; ordering written to the run metadata |
| 3. redshift | `romanmock.confusion`, `romanmock.redshift` | multi-line: `z_assigned = z_rsd` (optional `sigma_z`); single-line: draw identity *j* from `C_ij` (+ fail column) or from the line mix of the mock at the line's observed wavelength and flux rank (`LineFractionTable`), `z_assigned = lambda_i,0 (1+z_rsd)/lambda_j,0 - 1` |
| two-step | `RomanMockPipeline.build_base`, `romanmock.catalog.BaseCatalog` | base flux catalog written once; catalogs made from it in memory (`make_catalog`) or as files (`source: base`) |
| 4. samples | `romanmock.samples`, `romanmock.selection` | footprint, `redshift_assigned` range, tracer (`membership: assigned` = brightest line, disjoint samples; `detected` = any detected line for multi-line galaxies), outcome quality, population thinning, downsampling with `p_keep = target/mock`; built slice by slice |
| checks | `romanmock.diagnostics`, `romanmock.census` | empirical confusion, contamination fraction `f_{i->j}(B)`, detection and identification efficiency, completeness; `dN/dz` of emitters above a flux limit, single- versus multi-line |

### Two-tier flux limits

A galaxy enters the catalog only if at least one line exceeds the **primary** limit inside the bandpass
(`sensitivity.flux_limit`, default 1.0e-16 erg/s/cm², `romanmock.roman.DEFAULT_LINE_FLUX_LIMIT`). For such a
galaxy every other line inside the bandpass counts as detected above the **secondary** limit
(`sensitivity.secondary_flux_limit`, default 4.0e-17). A galaxy without a primary line has no detected line,
however many of its lines exceed the secondary limit.

```yaml
sensitivity:
  type: constant
  flux_limit: 1.0e-16
  secondary_flux_limit: 4.0e-17   # null switches the second tier off; must not exceed flux_limit
  wave_range: [10000.0, 19300.0]
```

The set of detected galaxies is the same as with the primary limit alone; what changes is `line_bitmask` /
`n_det`, i.e. more galaxies are multi-line and get an unambiguous redshift. On the full lightcone with a primary
limit of 1.5e-16, a secondary limit of 4.0e-17 raises the multi-line fraction from 44 % to 99.6 % and lowers the
interloper fraction from 0.56 % to 0.02 %; with the defaults (1.0e-16, 4.0e-17) 94 % are multi-line. The
brightest detected line (`line_true`) is always above the primary limit, so population thinning to a reference
dN/dz at the primary limit is unaffected; `membership: detected` samples do include secondary detections.
`TabulatedFluxLimit` takes the second tier as `secondary_ratio` (off by default), and
`GaussianNoiseDetection` applies it in signal-to-noise terms. Note that 4e-17 is 2.6σ for a 6.5σ limit of 1e-16.

### S/N thresholds: the Roman grism sensitivity

The second way of fixing the detection (`type: roman_grism`, `RomanGrismSensitivity`) replaces the two flux limits by
two S/N thresholds and a noise model from the NASA WFI technical page (grism section, v1.4): the 5σ line-flux limit at
ten observed wavelengths for one hour at twice the minimum zodiacal light, for a point source and for a source
with r50 = 0.3", scaling as `t^-1/2 b^1/2`. The 5σ flux of a line at wavelength λ in a galaxy of angular size r is

```
F_5σ(λ, r) = F_5σ,point(λ) · [1 + (r / r_c(λ))²]^p · (t / 3600 s)^-1/2 · (b / 2)^1/2        S/N = 5 F / F_5σ
```

with `r_c(λ)` fixed by the two tabulated sizes (0.12–0.14", close to the PSF) and `p = size_exponent` (0.5). A galaxy
needs one line with `S/N >= snr` (5) inside the bandpass; its other lines are detected with `S/N >= secondary_snr`
(2): one detected line makes a single-line galaxy, a primary line plus secondary ones a multi-line galaxy. The size
is one number in arcsec for all galaxies (`size: 0.3`; `0` = point source) or a column of the mock in proper kpc
(`size: r50_disk_2d`, the half-mass radius of the disk, converted at `redshift_true`; `romanmock base` then stores the
column, and `make_catalog` needs it in `base.columns`). `exposure_time` and `background` (in units of the minimum
zodiacal light; a typical HLWAS field is about 1.4) rescale the noise. With `snr: 6.5, size: 0.3` the limit at 1.5 µm
is 1.05e-16, the baseline of the flux-limit route. Sizes above 0.3" extrapolate the two tabulated rows.

```yaml
sensitivity:
  type: roman_grism
  snr: 5.0
  secondary_snr: 2.0            # null: single tier
  size: r50_disk_2d             # or 0.3 (arcsec); full form {column: r50_disk_2d, unit: kpc, redshift_column: redshift_true, scale: 1.0, floor: 0.0, fallback: null}
  exposure_time: 3600.0
  background: 2.0
output:
  store_snr: true               # <line>_snr: the S/N of every line
```

A galaxy whose size is not finite or negative is an error unless `fallback` gives the size [arcsec] it takes; the
catalog reports how many galaxies used it (`cat.info["n_fallback_size"]`). The floor is a smallest size, not a
missing-data policy.

A base catalog contains a later selection when its primary limits are nowhere deeper (checked on a grid of
wavelengths and, for sizes read from a column, of raw sizes of the same galaxy through each model's `scale` and
`floor`; `sensitivity.limits_contained`). A base and a request that read the size from different columns cannot be
compared and are refused, as is a `size_exponent` below the base one. A base built at a constant 1e-17 or with
`size: 0` serves every later choice. A noisy detection model (`detection: {type: gaussian_noise}`) scatters
galaxies across the limit, so `make_catalog` and the pipeline warn when the base is less than 1.5 times deeper than
the requested limit. The S/N route is explained in `RomanMock_SNR_Sensitivity_Note.md` of the project directory.

### Line features and bit ordering

The default `LineSet` (`roman_transitions`) follows comment 2 of the plan: one bit per
resolvable transition, with the [O II] doublet blended (Roman does not resolve it):

```
bit 0 Ha          6564.723  Ba_alpha_6563
bit 1 Hb          4862.763  Ba_beta_4861
bit 2 OIII_5007   5008.314  O_III_5007
bit 3 OIII_4959   4960.370  O_III_4959
bit 4 OII_3727    3728.619  O_II_3726+O_II_3729
bit 5 SIII_9532   9533.378  S_III_9532
bit 6 SIII_9069   9071.246  S_III_9069
bit 7 OIII_4363   4364.294  O_III_4363
bit 8 SIII_6312   6313.902  S_III_6312
```

`roman_transitions_paa` adds Paα as bit 9 (`PaA`, 18756.4 Å, column `Pa_alpha_1p8751um`), bits 0–8 unchanged. The
mocks carry Paα from `sdss_feniks_hizels_261013` onward, and `configs/production.yaml` uses this preset for them.
Paα is inside the grism at z < 0.029 only. There it is the only line in the bandpass: [S III] 9532 drops out below
z = 0.049. A single line at 18 756–19 300 Å can be Paα at z < 0.029 or Hα at z = 1.86–1.94.

`plan_species` reproduces the five-species bitmask of section 2 (Ha, Hb, [O III], [O II],
[S III]) and `all_transitions` gives every column its own bit. Multi-component features are
either unresolved blends (`blend: true`, detected as one feature at the representative
wavelength; the [O II] doublet) or separated transitions (`blend: false`, the default; each
component is tested at its own wavelength and the species bit is set when any component is
detected; the [O III] and [S III] doublets of `plan_species`). The species flux column is the
sum of the components. Any custom list of `{name, rest_wave, components, blend}` dicts in the
YAML `line_set` entry defines a new ordering. Rest wavelengths are taken from the catalog's
`ssp_data` file when available.

## Outputs

Every catalog is a set of OpenCosmo files, one per lightcone slice, `lc_cores-<step>.<tag>.hdf5`, that
`opencosmo.open(*files)` reassembles into a Lightcone (header, cosmology and HEALPix index are
preserved). OpenCosmo adds its own `redshift` column (equal to `redshift_true`). The diffsky input column
`redshift_obs` is z_rsd, not a measured redshift; it is never written under that name.

**Parent catalog** (`romanmock run`, `romanmock assign`), default columns:

| column | meaning |
|---|---|
| `ra`, `dec` | position (deg), copied from the input |
| `redshift_true` | cosmological redshift, copied from the input |
| `redshift_rsd` | true redshift including the line-of-sight peculiar velocity (the plan's z_rsd; copy of the input `redshift_obs`) |
| `redshift_assigned` | redshift assigned by the identification procedure, NaN when no reliable redshift |
| `id_outcome` (int8) | 0 NOT_DETECTED, 1 MULTI_LINE, 2 SINGLE_CORRECT, 3 SINGLE_MISID (interloper), 4 SINGLE_FAILED |
| `line_bitmask` (uint32) | detected features, bit *i* = feature *i* (`output.store_identification`) |
| `n_det` (uint8) | `popcount(line_bitmask)` |
| `line_true` (int8) | index of the brightest detected feature, -1 if none |
| `line_assigned` (int8) | identity assigned by the redshift procedure, -1 = none/failed |

`output.passthrough_columns` copies further columns (`gal_id`, and with `romanmock run` any input column such as
`logsm_obs`, `roman_F158`); `store_fluxes`, `store_wave_obs`, `store_snr`, `store_measured_flux` add per-feature
columns (`<feature>_flux` in erg/s/cm², `<feature>_snr` = `snr_at_limit × F / F_lim`, the S/N estimate of the
`roman_grism` sensitivity, ...).

**Base catalog** (`romanmock base`): `base.columns` (default `gal_id, ra, dec, redshift_true`), `redshift_rsd`,
`<feature>_flux` for every feature of the line set (float64; `base.flux_dtype: float32` saves 4 bytes per line
and galaxy, at the price of exact reproducibility), and `line_bitmask_base`, the bitmask at the base limits
(`base.store_bitmask`). Step 2 always computes its own `line_bitmask`.

**In-memory catalogs** (`BaseCatalog.make_catalog`): a `Catalog` of numpy arrays with `settings`, `info`
(counts, thinning probabilities) and `area_deg2`; `cat.write(path)` stores the columns as plain HDF5 datasets
(or FITS / npz) with the settings in the file attributes.

**Samples** (`romanmock select`): a directory `samples/<name>/` with one file per slice,
`lc_cores-<step>.<name>.hdf5`, opened as one lightcone with `romanmock.io.open_sample(path)`. The files hold
`output.sample_columns` (default `ra, dec, redshift_true, redshift_rsd, redshift_assigned, id_outcome`; `null`
keeps every parent column). `samples/<name>_metadata.yaml` records the sample definition, the number of galaxies
and `p_keep` per bin when thinning was applied. `output.sample_layout: single` writes the one file
`samples/<name>.hdf5` instead; OpenCosmo then holds the index of every slice in memory (about 20 GB for the
full lightcone). There is no `is_interloper` column: interlopers are `id_outcome == 3`.

**Provenance.** A `<tag>_metadata.yaml` sidecar records the configuration, the bit ordering, the outcome codes
and the model parameters; `<tag>_manifest.yaml` records the completed slices and a hash of the scientific
configuration. A rerun with the same configuration reuses finished slices; a changed configuration refuses to
mix with existing outputs unless `overwrite` is set.

### Disk space and the spatial index

OpenCosmo writes the HEALPix index of a lightcone uncompressed, a fixed 268 MB per file. `romanmock` stores it
gzip-compressed (about 1 MB) and leaves the data columns untouched; OpenCosmo reads the files as before.
This is on by default (`output.compress_index`, `base.compress_index`). Files written earlier are converted in
place, losslessly, with

```bash
romanmock compress-index $PSCRATCH/my_mock --workers 4
```

| full lightcone | galaxies | uncompressed index | default |
|---|---|---|---|
| base catalog, limit 1.0e-16, all line fluxes | 64.6 M | 25.2 GB | 7.6 GB |
| base catalog, limit 1.0e-17 (production) | 222.0 M | — | 25.9 GB |
| parent catalog at 1.0e-16 / 4.0e-17 | 64.6 M | 20.8 GB | 3.2 GB |
| parent catalog at 1.5e-16 / 4.0e-17 | 44.9 M | 19.9 GB | 2.3 GB |
| Hα sample, z 1.0 to 1.9, six columns, 22 slice files | 17.0 M | 6.6 GB | 0.73 GB |

## Usage

```bash
romanmock info <catalog_dir>                      # slices, redshift ranges, cosmology
romanmock lines                                   # print the bit ordering of the presets
romanmock example-config --catalog DIR --output OUT > my.yaml

romanmock base my.yaml --workers 32               # two-step, step 1: base flux catalog
romanmock catalog BASE_DIR -o out.hdf5 --flux-limit 1.5e-16 --lines Ha --z-range 1.0 1.9   # step 2: one catalog file
romanmock line-fractions BASE_DIR -o fractions.npz --flux-limit 1.5e-16   # the line mix of the single-line galaxies, for --confusion fractions.npz
romanmock assign my.yaml --flux-limit 1.5e-16 --output-dir OUT --workers auto   # step 2 as OpenCosmo files
romanmock run my.yaml --workers 32                # stage A in one pass from the input mock
romanmock run my.yaml --steps 219 224             # a subset of slices (job arrays: scripts/run_slices.sh)
romanmock summary my.yaml                         # outcome counts + empirical confusion
romanmock select my.yaml --workers 4              # analysis samples, slice by slice
romanmock compress-index DIR --workers 4          # convert existing files: compressed spatial index
```

`--workers auto` fits the pool to the memory budget (node memory, or the 30 GB cap of a NERSC login node).

Python:

```python
import opencosmo as oc
from romanmock import BaseCatalog, RomanMockPipeline, RomanMockConfig, add_line_fluxes
from romanmock.io import open_sample

# two-step: base catalog once, then catalogs in memory
RomanMockPipeline.from_yaml("cfg.yaml").build_base(n_workers=32)
cat = BaseCatalog(BASE_DIR).make_catalog(flux_limit=1.5e-16, lines=["Ha"], z_range=(1.0, 1.9))

# the file-based route: OpenCosmo files, one per slice
cfg = RomanMockConfig.from_dict({
    "source": "base", "base": {"dir": BASE_DIR},
    "output": {"dir": OUT_DIR, "passthrough_columns": ["gal_id", "ra", "dec", "redshift_true"]},
    "sensitivity": {"type": "constant", "flux_limit": 1.5e-16, "secondary_flux_limit": 4e-17,
                    "wave_range": [10000.0, 19300.0]},
})
pipe = RomanMockPipeline(cfg)
pipe.run(n_workers=4)
catalog = pipe.open_parent()                # OpenCosmo Lightcone of every slice
halpha = catalog.filter(oc.col("line_assigned") == pipe.line_set.bit("Ha"))
samples = pipe.build_samples(samples=[...], n_workers=4)     # list of sample directories
sample = open_sample(samples[0])

# or just add flux columns to any OpenCosmo dataset
ds = oc.open(".../lc_cores-219.diffsky_gals.hdf5")
ds = add_line_fluxes(ds, "roman_transitions")   # Ha_flux, Hb_flux, ... in erg/s/cm^2
```

Everything in `process_arrays` / `process_fluxes` works on plain dictionaries of numpy arrays, so the physics
can be exercised without touching the catalog (see `tests/`).

## Reference dN/dz, population thinning and the census

`romanmock.reference_dndz` holds Tables 1 and 2 of Wang et al. (2022, arXiv:2110.01829): the
Roman HLSS forecast of Hα and [O III] emitters per deg² in Δz = 0.1 bins, for six flux limits
(5e-17 to 2e-16 erg/s/cm²) and two dust models (A_V = 1.65, 1.92), with the tabulated ±1σ
(`wang2022_errors`; jackknife + cosmic variance). Values and uncertainties were checked bin by bin against the
LaTeX source of the paper. These are *true emitters above the flux limit*, before completeness and purity
("observational efficiency is not included"), so they are matched to the mock's detected-line populations in
bins of true redshift, before any assigned-redshift selection (`population_thinning:` in the config; on in
the baseline). Interlopers are then thinned consistently with the population they belong to.

A target spec is `{reference: wang2022, line: Ha, flux_limit: 1e-16, dust_av: 1.65}` (line and
flux limit default to the population / the run's primary flux limit), `{file: dndz.txt}` or
`{z: [...], dndz: [...]}`; the same specs drive the per-sample `downsample` on assigned redshift.
Only the six tabulated flux limits exist: a run with another primary limit needs an explicit `flux_limit` in
the target, a target of its own, or no thinning.

`poly_degree` fits a smooth selection function instead of per-bin ratios, so the realisation's
own radial fluctuations are not divided out. Shortages (target above mock, including empty bins)
are reported, never filled. For this mock at 1e-16 the Hα targets are attainable for z > 0.8 and
the [O III] targets for z < 2.85.

`romanmock.census.line_census` counts emitters directly from the input mock with flux, bandpass and redshift
cuts only, for several limits in one pass, split into single-line and multi-line galaxies. A limit is a number
(one threshold for every line, the population of the Wang et al. tables) or a `(primary, secondary)` pair (the
two-tier rule). `romanmock census` writes it to a file and notebook 01 plots it. `romanmock flux-distribution`
(`romanmock.fluxdist.FluxDistribution`) histograms the flux of every line of every galaxy per redshift bin with no
cut, for `quantile(line, q)`, `fraction_above(line, flux)` and `number_above` as functions of redshift (notebook 04; `rebin(step)` merges redshift bins). With one threshold at 1.0e-16 the mock has 1.5 times the forecast Hα
counts over 0.5 < z < 2.0 and 3 times the [O III] counts over 1.0 < z < 3.0.

## Galacticus mocks

`/global/cfs/cdirs/m4943/Galacticus/mockCatalogs/GP-MCMC-Calibration1_allUNIT_16sqDegMock_XMM-LSS/` (Andrew
Robertson, June 2026) is a 4° × 4° lightcone (16 deg², centred on RA 35.71°, Dec −4.75°) from the UNIT simulation:
`galaxyCatalogs/` holds 400 files `romanUNIT-d<a>_<b>_16sqDeg.hdf5` (431 GB, most of it star-formation histories),
random subsamples of the merger trees that each cover the whole volume, 38.5 M galaxies at 0 < z < 9 in all. The
galaxy properties are the datasets of `/Lightcone/Output1/nodeData` (dust-free) and
`/Lightcone/Output1/dustAttenuatedNodeData` (same names, dust applied: Calzetti law, stellar-mass and redshift
dependent attenuation of the lines). Seventeen emission lines (`balmerAlpha6565`, `balmerBeta4863`, `balmerGamma4342`,
`balmerDelta4103`, `nitrogenII6550`, `nitrogenII6585`, `oxygenII3727`, `oxygenII3730`, `oxygenIII4933`, `oxygenIII4960`,
`oxygenIII5008`, `paschenAlpha18756`, `paschenBeta12822`, `sulfurII6718`, `sulfurII6733`, `sulfurIII9071`,
`sulfurIII9533`) come as one dataset per component, `luminosityEmissionLine{Disk,Spheroid,AGN}:<line>` in erg/s;
the luminosity of a line is the sum of the three. `lightconeRedshiftCosmological` is the cosmological redshift,
`lightconeRedshiftObserved` includes the peculiar velocity. Cosmology: flat ΛCDM, H0 = 67.74, Ω_m = 0.3089 (no
radiation; it reproduces the stored comoving positions to 2e-5).

```python
from romanmock.galacticus import GalacticusMock

mock = GalacticusMock("/global/cfs/cdirs/m4943/Galacticus/mockCatalogs/GP-MCMC-Calibration1_allUNIT_16sqDegMock_XMM-LSS")
print(mock)                                     # 400 files, dust-attenuated lines, 16 deg², H0, Om0
data = mock.read(["redshift_true", "redshift_rsd", "ra", "dec", "balmerAlpha6565", "oxygenIII5008"],
                 z_range=(1.0, 2.0), n_workers=4)       # combined arrays of all files; a line = disk + spheroid + AGN
fluxes = mock.read_fluxes("galacticus_transitions", z_range=(0.5, 2.9), n_workers=4)   # Ha_flux, Hb_flux, ... erg/s/cm²
for path, chunk in mock.iter_files(["redshift_true", "balmerAlpha6565"]):             # file by file, a few MB each
    ...
dist = mock.flux_distribution(np.arange(0, 4.001, 0.01), n_workers=6)                  # FluxDistribution, as notebook 04
```

Columns are the package's names (`redshift_true`, `redshift_rsd`, `ra`, `dec`, `gal_id`, `is_central`), Galacticus line
names (summed over the components, dust-attenuated by default, `GalacticusMock(path, dust=False)` or `read(..., dust=False)`
for the dust-free values), explicit component datasets (`luminosityEmissionLineAGN:balmerAlpha6565`), magnitudes
(`apparentMagnitudeRomanWFI:F158`) or any other dataset of `nodeData` (`diskMassStellar`). `z_range=(lo, hi)` keeps
`lo <= z < hi` in `z_column` (default `redshift_true`); `files=` restricts to some files; `n_workers` reads files in
parallel processes (I/O bound: the whole mock's redshifts and one line take 3 minutes with four workers on a login node).

The line set `galacticus_transitions` (`romanmock lines`) has 16 features: the seven of `roman_transitions` under the
same names (`Ha`, `Hb`, `OIII_5007`, `OIII_4959`, `OII_3727` blend, `SIII_9532`, `SIII_9069`), then `NII_6585`, `NII_6550`,
`SII_6718`, `SII_6733`, `Hg`, `Hd`, `PaB`, `PaA`, `OIII_4933`. Rest wavelengths are the vacuum values of the air wavelengths
stored in the files (`romanmock.lines.GALACTICUS_LINE_WAVELENGTHS`, `galacticus.air_to_vacuum`). `romanmock
galacticus-flux-distribution <mock_dir> -o out.npz --dz 0.01 --workers 6` is the command-line form of
`mock.flux_distribution` (4 minutes for the whole mock on a login node); notebook 07 is notebook 04 for this mock.

## Rescaling the line fluxes

`romanmock.rescaling` changes the line fluxes of the galaxies before any selection, to test how the results depend on
the line model of the mock. A rescaling multiplies the flux of each feature by a constant and, optionally, by a
log-normal factor drawn once per galaxy:

`log10 F'_X = log10 F_X + offset_X + sigma_X * eps`, with `eps ~ N(0, 1)` a fixed function of the galaxy's `gal_id` and
of a seed (`galaxy_normal`): a galaxy gets the same deviate in every slice, rerun and subsample, and all its lines with a
scatter move together. Fluxes that are zero, negative or NaN, and features without an offset, are left as they are.

Models are chosen by `type`, like the other models (`build_rescaling`, `RESCALING_MODELS`):

* `line_factors` (`LineFactors(log_offsets={...}, scatter={...}, seed=...)`): explicit offsets and scatters per feature.
* `galacticus_like` (`GalacticusLike()`): the diffsky lines rescaled toward the Galacticus mock of the GRS-PIT, from
  `Projects/Roman/Fullshape/Catalog/diffsky_line_rescaling.ipynb` (2026-11-09). Hα is the anchor and never changes
  (diffsky's attenuated Hα is fitted to the HiZELS Hα luminosity functions). Every other feature that Galacticus also has
  (the names shared with `galacticus_transitions`; in `roman_transitions_paa` all but [O III] λ4363 and [S III] λ6312) gets
  a dust term, `delta` and `sigma`, with the Calzetti (2000) curve k at the feature's own rest wavelength:
  * dust. Galacticus attenuates its lines by a median A(Hα) of 0.89 mag (Calzetti curve, an A(Hα)–stellar-mass
    relation), diffsky by 0.34 mag with its steeper stellar curve. The blue lines (Hβ, [O III], [O II]) get the constant
    `-0.4 da_ha [k(λ)/k(Hα) - 1]` with `da_ha = 0.5` mag. The lines in `galaxy_dust` (default [S III] λλ9069, 9532 and
    Paα, whose ratio to Hα has a floor at its dust-free value) get their dust galaxy by galaxy: diffsky's own reddening
    relative to Hα, A_D (r - 1) with A_D the galaxy's A(Hα) read from its Paα/Hα (`galaxy_a_ha`) and r diffsky's
    A(X)/A(Hα), is replaced by the Calzetti reddening of the Galacticus A(Hα) at the same quantile of the two dust
    distributions (`dust_map`, `galacticus_a_ha`: 0.35 → 0.89 mag at the median, the least dusty tenth → no dust).
    Galaxies without dust keep the dust-free ratio; a constant term would put the floor of [S III]/Hα at 0.40 and
    leave almost no single-line galaxy below z = 1. `galaxy_dust=()` gives the first version, a constant term for
    every line;
  * `delta` (dex) for the nebular model and the AGN: [O II] +0.37 (diffsky's one log U = −2, solar-gas model gives an
    intrinsic [O II]/Hβ of 1.04, Galacticus 2.09), [O III] +0.19 (Galacticus's AGN and nebular model), Hβ +0.04 and
    Paα −0.055 (case-B ratios); members of a doublet share it;
  * `sigma` (dex), the galaxy-to-galaxy scatter that diffsky's fixed U and gas metallicity leave out: [O III] 0.19 and
    [O II] 0.11 with one deviate for both.

  Factors in `roman_transitions_paa` (`model.typical_offsets`, the per-galaxy lines at diffsky's median dust): Hβ 0.92,
  [O III] 5007 and 4959 1.32, [O II] 1.65, [S III] 9532 1.19 and 9069 1.17, Paα 1.31 (1.0 and 0.88 for a galaxy without
  dust). The median line ratios of the detected galaxies match Galacticus within 0.02 dex and Paα/Hα takes its
  distribution (`Projects/Roman/Fullshape/Catalog/diffsky_line_rescaling.ipynb`; the single-line fractions are in
  `single_line_fraction_rescaled.ipynb` of the same directory).

In a catalog, `make_catalog(..., rescaling="galacticus_like")` rescales the fluxes of every galaxy before the detection
(the flux columns of the catalog are then the rescaled ones), and `line_fractions(..., rescaling=...)` builds the
empirical table from the same fluxes; `make_proposal` uses it by default (below). The YAML pipeline does not apply it.
At the array level, apply it to the fluxes read from the base catalog, then select; the per-galaxy dust reads the Hα and
Paα fluxes of each galaxy, so read those columns too (a line set without Paα, such as the earlier mock's, needs
`GalacticusLike(galaxy_dust=())`).

```python
from romanmock import BaseCatalog, GalacticusLike

base = BaseCatalog(".../base")
cat = base.make_catalog(flux_limit=1.5e-16, secondary_flux_limit=6e-17, rescaling="galacticus_like", z_range=(1.0, 1.2))
model = GalacticusLike()
print(model.describe(base.line_set))                    # the factor and scatter of every feature
for data in base.read_slices(["gal_id", "redshift_rsd", *base.flux_columns], z_range=(0.5, 3.0)):
    data = model.apply_columns(data, base.line_set, suffix=base.flux_suffix)   # every <line>_flux column rescaled
    ...                                                 # select on the rescaled fluxes
flux = model.apply(flux, base.line_set, gal_id)         # the same for an (n, n_features) array, Hα and Paα included
```

`model.max_log_boost(line_set, n_sigma=5)` is the largest brightening in dex: a base catalog complete above F holds
every galaxy the rescaled selection can reach down to F × 10^boost (1.07 dex for `galacticus_like`, 11.8 times). The
merged [O III] doublet adds a factor 1 + 1/2.98 = 1.34 (15.8 times together), so a base at 1e-17 is complete for
primary thresholds from 1.6e-16 up only. `make_catalog` and `make_proposal` warn when the thresholds are closer to the
base limit than that. That is why the production base is at 1e-18: the S/N route reaches 3.1e-17 for a point source.

## The [O III] doublet: one line or two

[O III] λλ4959, 5007 are 47.94 (1 + z) Å apart (96–185 Å in the bandpass) with a fixed flux ratio 2.98. In the slitless
grism the image of a line along the dispersion is the galaxy itself, so whether the two lines separate depends on the
galaxy's size. `romanmock.doublets.DoubletRule` models each line as a face-on exponential disk of the galaxy's
half-light radius projected on the dispersion (10.85 Å per 0.11" pixel), smoothed by the 2-pixel resolution of a point
source, and calls the doublet resolved when the profile dips between the lines by at least 20 % of the λ4959 peak: the
largest resolved half-light radius is 0.35" at z = 1.0, 0.55" at 1.94, 0.74" at 2.85 (`rule.resolved_radius(z)`; a dip
of 10 % or 30 % moves it by about ±15 %). Derivation: `Projects/Roman/Diffsky_line_rescaling_and_OIII_doublet.md`.

`make_catalog(..., doublets=...)`, `line_fractions` and `make_proposal` apply it to the feature fluxes before the
detection, in the galaxies whose two lines are both inside the bandpass:

| `doublets` | λ4959 |
|---|---|
| `"per_galaxy"` | a line of its own where the doublet is resolved for the galaxy's `r50_disk_2d` (proper kpc at `redshift_true`; must be in `base.columns`), added to λ5007 where it is blended |
| `"merged"` | added to λ5007 in every galaxy: a doublet never confirms a redshift |
| `"separated"`, `None` (default of `make_catalog`) | a line of its own in every galaxy |

`{"mode": "per_galaxy", "size": 0.3, "min_dip": 0.3}` sets one size for all and another threshold. A merged doublet is
detected at λ5007's rest wavelength with the sum of the two fluxes; the column `doublet_merged` says where. At the
production base the rule resolves the doublet for 29 % of the [O III]-detected galaxies at 1.0 < z < 1.2 and 74–80 %
above z = 1.94 (with the rescaled lines).

## Two steps to a catalog with interlopers: proposal and catalog

How many galaxies of a catalog rest on a single line, and which single lines elsewhere could pass for its line, depends
only on the thresholds; how they are identified is a separate choice. The routine therefore has two steps: a proposal,
which counts, and the catalog, built either *by design* from the interloper fractions you want or from an
identification model. A proposal without a secondary limit and without an expected interloper fraction is for
*single-line clean catalogs* instead, with no interlopers (below).

**Step 1, `make_proposal`: range, line, primary threshold, expected interloper fraction.** One pass over the slices of
the base catalog that matter, with the line rescaling and the doublet rule (by default `"galacticus_like"` and
`"per_galaxy"`):

```python
from romanmock import BaseCatalog

base = BaseCatalog("/global/cfs/cdirs/m4943/Simulations/RomanMocks/sdss_feniks_hizels_261013_10_14_2026/base")
proposal = base.make_proposal((1.0, 1.2), line="Ha", flux_limit=1.5e-16, interloper_fraction=0.01)   # secondary: step 2
proposal = base.make_proposal((1.0, 1.2), line="Ha", flux_limit=1.5e-16, secondary_flux_limit=6e-17)  # two tiers
proposal = base.make_proposal((1.0, 1.2), line="Ha", snr=5, size="r50_disk_2d", interloper_fraction=0.03)  # or S/N
proposal.plot_secondary()     # the most each kind of interloper can make of the catalog, against the secondary limit
proposal.plot()               # single-line fractions and potential interlopers against redshift
```

It reports, against true redshift in bins of `dz` (0.01):

* the **single-line galaxies inside the range**: their share of the detected galaxies, in all and per line;
* the **potential interlopers**: single-line galaxies at other redshifts whose line falls where the target line would be
  for a redshift inside the range (Hα at 1.0–1.2: 13 130–14 440 Å, so [O III] λ5007 at 1.62–1.88, [S III] λ9532 at
  0.38–0.52, [O II] at 2.52–2.87, ...): their single-line fraction at their own redshift, how many per deg², and the
  share of the catalog they would make if every one were taken (the most any identification gives: "at most");
* **interlopers against the secondary limit**: the same "at most", per kind and in total, for secondary limits from where
  any interloper becomes possible up to the primary limit. A galaxy's other lines are detected above the secondary
  limit, so a higher one leaves more galaxies with a single line. With `interloper_fraction` (the total expected in the
  catalog) it says from which secondary limit that total is possible. Given `interloper_fraction` and no secondary
  limit, the proposal's own tables are those of a single tier (the secondary limit at the primary one, the ⁴ column),
  and step 2 chooses the secondary limit.

```
── Interlopers against the secondary limit ─────────────────────────────────────────────────────────
  secondary F [1e−17]  2      3      4   4.5     6   7.5     8     10    15⁴
  ───────────────────  ─  ─────  ─────  ────  ────  ────  ────  ─────  ─────
  SIII_9532            0      0      0     0  0.36  4.66  5.84   9.75  16.26
  OIII_5007            0  <0.01  <0.01  0.01  0.19  0.78  1.09   2.90  10.03
  other lines          0      0      0     0     0     0     0  <0.01   0.02
  total                0  <0.01  <0.01  0.01  0.55  5.44  6.93  12.65  26.31
  ⁴ this proposal's secondary flux limit, 1.5e-16 (the tables above); primary F > 1.5e-16
  below a secondary flux limit of 2.3e-17 every potential interloper keeps a second line: none can be made
  expected 1.00 % of interlopers in all: possible from a secondary flux limit of 6.1e-17 up; ...
```

No [S III] λ9532 galaxy is single below 5.8 × 10⁻¹⁷: the mock's λ9532/λ9069 ratio is fixed (2.57), so λ9069 stays above
any lower secondary limit. Between 6 and 8 × 10⁻¹⁷ the possible interlopers jump from 0.55 % to 6.9 % of the catalog.

The scan costs nothing extra: every sensitivity model puts the secondary limit at a fixed fraction of the primary one
at each wavelength, so one number per galaxy (its second-brightest line in units of the primary threshold) decides it
for every secondary limit. Counts are exact on a grid of 0.001 dex and at the proposal's own secondary limit, linear in
log between. A line single inside the range is minor when it is the single line of fewer than `min_count` (100)
galaxies or of less than `min_share` (0.1 %) of the single-line galaxies there; a potential interloper, when it has
fewer than 100 galaxies or less than 0.1 % of the sample. Minor potential interlopers stay out of the summary's table
and of the confusion template, but are listed under it and drawn as thin lines by `proposal.plot()`, so that the
figure shows every pair the worst case can make. The summary also gives the worst case (Step 2c below).
`proposal.save(path)` / `Proposal.load(path)` keep it; the numbers are in `proposal.single_lines_inside()`,
`interlopers()`, `interloper_scan(secondary)`, `secondary_for(total)`, `single_line_fraction(ratio)`,
`fraction_inside(line)`, `fraction_interloper(line)`, `contamination_at_most(line)` and `proposal.counts`.

**`line` decides which galaxies the catalog holds** (`make_proposal` prints this as a hint):

| `line` | the catalog | its sample before interlopers |
|---|---|---|
| `"Ha"` (default) | the galaxies in which Hα is identified, with the redshift from Hα (observed wavelength / 6564.7 Å − 1) inside the range: Hα detected, alone or with other lines, or a single line taken for Hα (an interloper) | the galaxies with Hα detected inside the range |
| `"OIII_5007"`, any line | the same for that line; a galaxy with Hα and [O III] belongs to both catalogs | the galaxies with that line detected |
| `"all"` | every galaxy with a redshift inside the range, whichever line gave it (`make_catalog(lines=None)`) | every detected galaxy inside the range |

With `line="all"` a potential interloper is a pair: a single line that, taken for another line, gets a redshift inside
the range. Only lines that single-line galaxies inside the range actually show (above the minor-line cut,
`proposal.assignable`) count as the line it is taken for: by wavelength alone a single Hα line could pass for
[S III] λ6312, but no identification would assign a line that is never single. The cut is on the share of the
*single-line* galaxies, not of every detected galaxy. At 1.0 < z < 1.2 (flux limits 1.5 / 0.6 × 10⁻¹⁶, the rescaling
of the notebooks) [O III] λ5007 is the single line of 0.04 % of the detected galaxies but 1.4 % of the single-line
ones, and the single Hα lines of 0.53 < z < 0.68 that could be taken for it are 721,584, at most 5.0 % of the catalog
(6.0 % in all with the [S III] λ9532 pairs). Until 2026-11-13 the share was of every detected galaxy, which left these
pairs out and gave at most 0.23 %. `proposal.interlopers()` is then `{taken for: {line: ...}}`, and in the figures the
colour is the true line and the line style the line it is taken for.
A single line can fall where two such lines would be (wide ranges), so the summed "at most" is an upper bound.

**Step 2a, a catalog by design: `proposal.make_catalog(fractions=..., out_dir=...)`.**

```python
table = {"Ha": {"OIII_5007": 0.005, "SIII_9532": 0.005}}      # [O III] and [S III] taken for Hα: 0.5 % of the catalog each
proposal.design(table, secondary_flux_limit=7.5e-17)          # what it takes (no catalog yet)
cat = proposal.make_catalog(fractions=table, secondary_flux_limit=7.5e-17, out_dir="/pscratch/.../Ha_design")
table = proposal.fraction_template(0.01, secondary_flux_limit=8e-17)   # 1 % shared in proportion to the numbers
```

`fractions` is `{line taken for: {interloper line: fraction}}`, each fraction a share of the catalog (of the Hα sample for
`line="Ha"`; of every galaxy of the catalog for `line="all"`). With `N` the sample, `n` the potential interlopers of a
pair at the chosen secondary limit and `F` the sum of the fractions, the catalog holds `N / (1 − F)` galaxies (for
`line="all"`, interlopers whose true redshift is already inside the range only move: `N / (1 − Σ f n_out / n)`), and a
single line of the pair, inside the window where it gets a redshift inside the range, is taken for the target with
`P = f N_cat / n`. Everywhere else, and for the lines not listed, the identification is correct, so no galaxy of the
sample is lost (`type: windowed`, `romanmock.confusion.WindowedConfusion`). A table that needs more interlopers than
there are single lines (`P > 1`) is refused with the lowest secondary limit that meets it. `make_catalog` prints the
design (the potential interlopers and `P` of each pair), the catalog, and the fractions asked and made; the summary file
keeps both (`design`). The secondary limit (`secondary_flux_limit` for flux limits, `secondary_snr` for S/N) defaults to
the proposal's (a single tier for a proposal made without one).

**Step 2b, an identification model: `proposal.make_catalog(confusion, out_dir)`.**

```python
rows = proposal.confusion_template()                      # fill in the probabilities (an assumed scenario)
rows = {"OIII_5007": {"Ha": 0.05}, "SIII_9532": {"Ha": 0.05}, "Ha": {"OIII_5007": 0.005}}
cat = proposal.make_catalog(rows, "/pscratch/.../Contaminated", n_randoms=10)
cat = proposal.make_catalog("empirical", "/pscratch/.../Contaminated_empirical")      # the line mix of the mock
```

A confusion table does not depend on the wavelength: a line taken for another is taken wherever it is, which can also
lose galaxies of the sample (for `line="all"`, the template leaves out loss entries toward lines that are not
assignable for that reason). `"empirical"` builds the line-fraction table at the same thresholds, rescaling and doublet
rule first.

Both write `<name>.hdf5` (default name `Ha_z1p0_1p2`, `all_z1p0_1p2`), `<name>_summary.json` (settings, outcome counts,
line report, the proposal's numbers, the design), `<name>_proposal.npz` and, with `n_randoms`, `<name>_random.hdf5`
(the randoms of the whole catalog; `cat.subset(which).make_random(...)` for a part). `confusion={}` gives the clean
catalog. Notebooks: `notebooks/01_catalog_z1p0_1p2.ipynb` and `02_catalog_z1p6_2p0.ipynb` (confusion table and line
mix), `04_catalog_by_design.ipynb` (interloper fractions), `05_single_line_clean_catalogs.ipynb` (below),
`06_worst_case.ipynb` (the worst case, below).

**Step 2c, the worst case: `proposal.make_catalog("worst_case", out_dir)`.** Every single line is an interloper: it is
taken for another line, for one that gives it a redshift inside the range wherever there is one (`type: worst_case`,
`romanmock.confusion.WorstCaseConfusion`, no random numbers). No single-line galaxy keeps its redshift: those of the
range leave it (taken for a line that puts them outside), and those of other redshifts whose line falls where a
taken-for line would be inside the range come in.

```python
proposal = base.make_proposal((1.0, 1.2), line="all", flux_limit=1.5e-16, secondary_flux_limit=6e-17, rescaling=RESCALING)
print(proposal.worst_case_summary())      # the catalog it makes, from the counts, and every case it leaves out
cat = proposal.make_catalog("worst_case", "/pscratch/.../all_z1p0_1p2_worst", n_randoms=20)
```

The lines a single line can be taken for are `proposal.worst_case_lines()`, the lines of the potential interlopers
(`proposal.assignable`): the single line of at least `min_count` galaxies inside the range and of `min_share` of its
single-line galaxies, the most common first. At 1.0 < z < 1.2 with that rescaling these are Hα and [O III] λ5007, and
the single Hα lines of 0.53 < z < 0.68 taken for [O III] make 5.1 % of the worst-case catalog (721,584 of 872,310
interlopers, 6.19 %; the potential interlopers make at most 6.03 % of a catalog that keeps its sample). A line that is
never, or hardly ever, the single line of a galaxy of the range (Hβ, [O III] λ4959, [S III] λ6312, ...) is not taken
for. `worst_case_lines(min_count=100)` drops the share cut (a harsher worst case: with a single tier at 1.6 < z < 2.0 it
adds Hβ and [O II], the single line of 207 and 136 of 5.5 M single-line galaxies), and `{"type": "worst_case",
"taken_for": [...]}` chooses the lines. `proposal.worst_case()` gives the numbers (interlopers per pair, those
from inside the range, the single lines that leave it, the catalog size and share), `make_catalog` prints them with the
numbers made (equal, the selection being deterministic) and keeps both in the summary file (`worst_case`).
`proposal.misidentifications()` lists every way a single line can get a redshift inside the range, including the single
lines *of* the range that stay inside it with a wrong redshift. That needs two lines closer in wavelength than the
range is wide in 1 + z (1.10 at 1.0–1.2, 1.15 at 1.6–2.0), e.g. Hα taken for [S III] λ6312 (ratio 1.040) or [O III]
λ5007 for λ4959 (1.010) or Hβ (1.030); Hα and [O III] (1.311) never do it in these bins, and the lines that do are never
single, so the worst case has no such galaxies there. In a wider range it has them (`WorstCaseConfusion` gives them their
wrong redshift inside the range), and where the windows of two taken-for lines overlap a single line goes to the first.

**Single-line clean catalogs: no secondary limit, no interlopers.**

```python
proposal = base.make_proposal((1.0, 1.2), line="Ha", flux_limit=1.5e-16)       # no secondary limit, no interloper_fraction
cat = proposal.make_catalog(out_dir="/pscratch/.../Ha_single_clean")            # one line per galaxy
cat = proposal.make_catalog(secondary_flux_limit=7.5e-17, out_dir="/pscratch/.../Ha_single_clean_7p5e-17")
```

Without a secondary limit and without `interloper_fraction` (`flux_limit` alone, or `snr` alone), the proposal is for
single-line clean catalogs: the galaxies with the target line above the primary limit (every detected galaxy for
`line="all"`), each with its true redshift, and no interlopers. The summary gives, against redshift, the share of them
that are *true* single-line galaxies, with no other line above 0.3, 0.4 and 0.5 times the primary limit
(`secondary_ratios`; `proposal.single_line_fraction(ratio)`):

```
── Hα single-line galaxies against redshift ────────────────────────────────────────────────────────
  z          Hα galaxies¹  4.5e-17  6e-17  7.5e-17
  ─────────  ────────────  ───────  ─────  ───────
  1.00–1.01       707,113     0.00   0.12     1.07
  ...
  1.19–1.20       553,949     0.26   1.17     3.13
  all          12,617,496     0.17   0.92     2.55
```

`make_catalog(out_dir=...)` makes the catalog with one line per galaxy: every galaxy is a single-line galaxy
(`id_outcome` 2), its line the target line also where another is brighter (with the rescaling and the merged doublet,
[O III] outshines Hα in 43 % of them at 1.0–1.2); for `line="all"` each galaxy keeps its brightest line.
`make_catalog(secondary_flux_limit=..., out_dir=...)` (or `secondary_snr`) makes the same galaxies with their other lines
above that limit (detection `primary_line`: only the galaxies with the target line above the primary limit are
detected). The single-line galaxies (2) are then those without another line, as many as the proposal counts (321,887,
2.55 %, at 7.5 × 10⁻¹⁷), and the others are multi-line (1). Neither takes a confusion table or interloper fractions, and
the interloper methods (`design`, `interloper_scan`, `confusion_template`, ...) refuse a single-line proposal. For
interlopers, give the proposal `interloper_fraction` or a secondary limit.

## Resources

Measured on the full lightcone (324 M input galaxies, base limit 1.0e-16, 64.6 M detected):

| step | where | wall time | memory per worker |
|---|---|---|---|
| `romanmock base` (or `run`) | CPU node, 32 workers | about 1 min | 5.7 GB (4.4 GB for `run`) |
| `make_random` (in memory) | login node, 1 process | 5 s per 21 M points | 0.5 GB per 21 M points |
| `make_catalog` (in memory) | login node, 8 workers | 15 to 20 s | 3.3 GB |
| `romanmock assign` | login node, 4 to 8 workers | 1 to 2 min | 2.4 GB |
| `romanmock select`, thinning + two samples | login node, 4 workers | 3.4 min | 1.7 GB |
| `romanmock compress-index`, 198 files | login node, 4 workers | 1.7 min | 0.3 GB |
| `line_census`, 42 slices, 6 limits | CPU node, 32 workers | 0.8 min | about 5 GB |

NERSC login nodes cap each user at 30 GB, so steps that read the input mock fit with at most 5 workers there; a
short batch job is simpler (`scripts/production.sh`). 

**Streaming the base catalog.** `base.read_slices(["redshift_rsd", "Ha_flux"], z_range=(0.5, None))` yields the
stored columns one slice at a time (a few hundred MB), for histograms and scans of the base catalog without
`make_catalog`; notebook 01 uses it for the flux distributions.

**Sky area.** Give it if you know it: `BaseCatalog(path, area_deg2=2500.0)` or `make_catalog(area_deg2=2500.0)`.
Otherwise it is estimated, from the sky patches of the mock (2062.65 deg² for this one) or, when the mock has
no patch decomposition, from the galaxies in the HEALPix index. `cat.info["area_source"]` records which:
`user`, `footprint` or `healpix`. The file-based route always uses the HEALPix estimate (`sky_area_deg2`).

## Production data and notebooks

The products of the current mock are in
`/global/cfs/cdirs/m4943/Simulations/RomanMocks/sdss_feniks_hizels_261013_10_14_2026/`:

* `base/`: 403.1 M galaxies with a line above 1.0e-18 erg/s/cm² (97.7 % of the mock), lensed fluxes, line set
  `roman_transitions_paa`, 67 slices, 61 GB. Columns: `gal_id`, `ra`, `dec`, `redshift_true`, `r50_disk_2d`, `ra_obs`,
  `dec_obs`, `magnification`, `redshift_rsd`, the line fluxes and `line_bitmask_base`. The unlensed flux is
  flux / `magnification`.
* `census/`.

Both come from `sbatch scripts/production.sh`. The base was rebuilt at 1.0e-18 on 2026-11-09 (job 61070403, 1 min 52 s
for the base alone). The 1.0e-17 base before it (344.5 M galaxies, 52.5 GB) is a subset of the new one, every column
identical but `line_bitmask_base`; it was deleted on 2026-11-09 after that check. The table below describes the products of the
previous mock, `/global/cfs/cdirs/m4943/Simulations/RomanMocks/sdss_feniks_hizels_260710_09_07_2026/`:

| directory | content | made by |
|---|---|---|
| `base/` | base catalog: 222.0 M galaxies with a line above 1.0e-17 erg/s/cm² (74 % of the mock), 66 slices, 25.9 GB | `sbatch scripts/production.sh` |
| `census/` | `dndz_census_z0p5_3p0.npz`: dN/dz per line (redshift bins of 0.01) for primary limits 1.0e-16, 1.5e-16 and secondary limits none, 8e-17, 6e-17, 4e-17; `flux_distribution.npz`: histograms of log flux (0.02 dex) per line and redshift bin (0.01) of all 324 M galaxies | `sbatch scripts/production.sh` |
| `catalogs/` | `halpha_z1p0_1p9.hdf5`: Hα sample at 1.0 < z < 1.9 | the former notebook 02 (removed 2026-11-09) |
| `randoms/` | `halpha_z1p0_1p9_<which>_<index>.hdf5`: 10 random catalogs each for all, galaxies and interlopers | the former notebook 03 (removed 2026-11-09) |

`notebooks/` holds six executed examples on the current base catalog. They write to `$PSCRATCH/RomanMocks/examples/`.
Since 2026-11-12 they pass their own rescaling to `make_proposal` instead of the default `"galacticus_like"`:
`LineFactors(log_offsets={"OII_3727": 0.25}, scatter={"OIII_5007": 0.19, "OIII_4959": 0.19, "OII_3727": 0.11})`, which
gives [O III] a scatter of 0.19 dex, brightens [O II] by 0.25 dex with a scatter of 0.11 dex, and leaves the other
lines raw (the introduction of notebook 01 says why). The proposal and catalog numbers quoted in this README are for
the default rescaling.
Run them on a CPU node (03 also runs on a login node, after 01):
`sbatch scripts/run_notebook.sh notebooks/01_catalog_z1p0_1p2.ipynb notebooks/02_catalog_z1p6_2p0.ipynb notebooks/03_randoms.ipynb notebooks/04_catalog_by_design.ipynb notebooks/05_single_line_clean_catalogs.ipynb notebooks/06_worst_case.ipynb`

1. `01_catalog_z1p0_1p2.ipynb`: the Hα catalog at 1.0 < z < 1.2 in the two steps. The proposal (flux limits 1.5 / 0.6 × 10⁻¹⁶, S/N 5 / 2, `line="all"`), then the catalog with an assumed confusion table and with the line mix of the mock, its components, and the files.
2. `02_catalog_z1p6_2p0.ipynb`: the same at 1.6 < z < 2.0 for every galaxy (`line="all"`): Hα leaves the grism at z = 1.94, and [O III] λ5007 carries the range to 2.0.
3. `03_randoms.ipynb`: the random catalog of the catalog of notebook 01 and of each of its parts (`cat.subset(which).make_random(...)`): multi-line galaxies, single-line galaxies with a correct redshift, all interlopers, each kind of interloper. It checks them (inside the footprint, the n(z) of the data) and shows that they are reproducible, extensible and independent between parts.
4. `04_catalog_by_design.ipynb`: catalogs by their interloper fractions: the proposal with an expected total and its scan of the secondary limit, `design`, a table that cannot be met, `fraction_template`, the S/N scan, and every galaxy at 1.6 < z < 2.0.
5. `05_single_line_clean_catalogs.ipynb`: single-line clean catalogs, without interlopers: the true single-line share against redshift at 0.3, 0.4 and 0.5 times the primary limit, the catalog with one line per galaxy and with a secondary limit of 7.5 × 10⁻¹⁷ (the same galaxies, single-line as the proposal counts), the S/N proposal.
6. `06_worst_case.ipynb`: the worst case, every single line an interloper, for every galaxy at 1.0 < z < 1.2 and 1.6 < z < 2.0 (flux limits 1.5 / 0.6 × 10⁻¹⁶): `worst_case_summary()`, `misidentifications()` (every way a single line can land in the range, and why none of the range stays inside it with a wrong redshift), the catalogs (made = expected), their components and the true redshifts of their interlopers, and other lines to take the single lines for.

## Extending (hooks for the plan's comments)

* **Wavelength-dependent sensitivity** (comment 6): `sensitivity: {type: tabulated, file: curve.txt}`,
  `{type: roman_grism, ...}` (wavelength, source size, exposure, background; the WFI tables can be replaced by
  the constructor arguments `wave`, `flux_5sigma_point`, `flux_5sigma_extended`), or register a class in
  `romanmock.sensitivity.SENSITIVITY_MODELS`. A model that defines `secondary_flux_limit(wave_obs)` gets the
  two-tier rule; a model with `uses_size = True`, a `size_spec` (`romanmock.sizes.SizeSpec`) and
  `flux_limit(wave_obs, size)` gets the angular size of each galaxy from the pipeline.
* **Detection scatter / overlap**: implement `DetectionModel.detect(flux, wave_obs, rng, size=None)`
  and register it in `DETECTION_MODELS` (`GaussianNoiseDetection` is the first example).
* **Confusion depending on `lambda_obs`, S/N, photometry** (comment 3): implement
  `ConfusionModel.assign(true_index, context, rng)`; `context` carries `wave_obs`, `snr`,
  `flux`, `z_rsd` and any `redshift.context_columns` from the catalog (in the two-step workflow they must be
  in `base.columns`). `WavelengthBinnedConfusion` is the first example; `LineFractionTable` (`type: empirical`)
  the second: it uses `wave_obs` and `flux` and not the true line. Matrices can be loaded from YAML files.
* **Multi-line redshift errors / wrong multi-line identifications**: implement
  `MultiLineModel.assign(det, z_rsd, sel, rng)` (default `PerfectMultiLine(sigma_z)`).
* **Feature definition** (comment 2): change the `line_set` list; blends are sums of columns.
* **Footprints**: `BoxFootprint`, `HealpixFootprint` (pixel list or map), or any callable `(ra, dec) -> mask`.
* **Target dN/dz** (comment 5): `TargetDNDZ` table; `mock_dndz` can be smoothed; shortages
  are reported, never filled.

The default confusion matrix (`romanmock.confusion.DEFAULT_CONFUSION_ROWS`: true Hα called [O III] 5007 with
probability 0.5 %, true [O III] 5007 or [S III] 9532 called Hα with 5 %, true Hβ called [O III] 5007 with 5 %,
no failures) is an explicitly assumed scenario, not a Roman calibration. With the default secondary limit it
matters for about 6 % of the detected galaxies, the single-line ones. The empirical alternative
(`confusion: {type: empirical, ...}`) replaces the assumed numbers by the line mix of the mock itself, which is
only as good as the mock's line ratios.

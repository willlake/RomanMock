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

A diffsky mock directory such as
`/global/cfs/cdirs/m4943/Simulations/sdss_feniks_hizels_260710_09_07_2026/`: 67 lightcone
slices `lc_cores-<step>.diffsky_gals.hdf5` (z = 0-4, 324 M galaxies, 2063 deg²) plus the auxiliary
`*_ssp_data.hdf5`, `*_param_collection.hdf5`, `*_t_table.hdf5`, `*_transmission_curves.hdf5`.
The line columns (`Ba_alpha_6563`, `Ba_beta_4861`, `O_III_4363`, `O_III_4959`, `O_III_5007`,
`O_II_3726`, `O_II_3729`, `S_III_6312`, `S_III_9069`, `S_III_9532`) are dust-attenuated,
continuum-subtracted luminosities in erg/s, one column per transition.
`redshift_obs` includes the line-of-sight peculiar velocity (`z_rsd` of the plan);
`redshift_true` is the cosmological redshift. The cosmology is read from the file header
(h = 0.6766, Ω_m = 0.3096, the Planck 2018 cosmology of the parent N-body simulation) and is
used for the luminosity distances.

`romanmock info <catalog_dir>` lists the slices, their redshift ranges and the cosmology.

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

`cfg.yaml` needs `catalog.path`, `base.dir` and, to change the defaults, `sensitivity` and `base.columns`.
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
               "OIII_5007": {"Ha": 0.2}},    # None: the package default
    lines=["Ha"],                            # galaxies in which Ha is identified
    z_range=(1.0, 1.9),                      # range of redshift_assigned
    seed=2026,
)
cat["ra"], cat["redshift_assigned"], len(cat)     # numpy arrays
print(cat.summary())                              # counts per identification outcome
cat.write("halpha.hdf5")                          # optional: .hdf5, .fits or .npz; Catalog.read() brings it back
```

| argument | meaning | default |
|---|---|---|
| `flux_limit` | primary limit; cannot be deeper than the base catalog | the base catalog's limit |
| `secondary_flux_limit` | limit for the other lines of a detected galaxy; `None` = no second tier | 4.0e-17 |
| `confusion` | rows of the confusion matrix, a full config, or a model object | package default |
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

**Line report.** With `verbose=True` the call prints which lines fall inside the bandpass for the redshift
cut, and the line interlopers: single-line galaxies given a wrong line. It lists the potential ones (a line
that, taken for another, passes the redshift cut, with its probability `P` in the confusion matrix) next to
the number that is actually in the catalog. `cat.line_report()` returns the same text and
`cat.info["line_report"]` the numbers.

| column | galaxies of the catalog in which |
|---|---|
| detected | the line is detected |
| identified | it is detected and identified as this line: the redshift of the galaxy is right. Every detected line of a multi-line galaxy counts, also a secondary one |
| brightest | it is identified and is the brightest detected line (`line_assigned`) |
| mistaken | it is the only detected line and was taken for another line |
| interlopers | another line was taken for this line |

detected = identified + mistaken. The sample of a line is its identified galaxies plus its interlopers, and
`contamination` is the share of one interloper in it.

```text
[romanmock] lines inside 10000 - 19300 A for redshift_assigned in 1 - 1.9 (lightcone: z_rsd 0.01 - 4.06)
  line        rest [A]    redshift      observed [A]       detected   identified    brightest    mistaken  interlopers
  Ha            6564.7  1.000 - 1.900   13129 - 19038    22,505,531   21,129,958   21,125,951   1,375,573      333,709
  OIII_5007     5008.3  1.000 - 1.900   10017 - 14524    18,199,510   18,004,823            0     194,687    1,375,573
  ...
[romanmock] line interlopers: single-line galaxies given a wrong line and so a wrong redshift
  true line  taken for   true z_rsd     assigned z     observed [A]        P   in catalog  contamination
  OIII_5007  Ha         1.622 - 2.801  1.000 - 1.900   13129 - 19038   0.500      194,687          0.91%
  SIII_9532  Ha         0.377 - 0.997  1.000 - 1.900   13129 - 19038   0.500      139,022          0.65%
  Ha         OIII_5007  0.526 - 1.212  1.000 - 1.900   10017 - 14524   0.500    1,375,573          7.10%
  ...
```

The assigned redshift of a galaxy with more than one detected line is `redshift_rsd` (the true redshift
including the peculiar velocity), never `redshift_true`. For a single-line galaxy it follows from the line
identity drawn from the confusion matrix.

The same from the command line, writing a file:

```bash
romanmock catalog /path/to/base -o halpha.hdf5 --flux-limit 1.5e-16 --secondary-flux-limit 8e-17 \
          --confusion my_confusion.yaml --lines Ha --z-range 1.0 1.9 --downsample wang2022
```

Notebook 02 walks through this. In a script, call `make_catalog` below `if __name__ == "__main__":` because
the slices are read by worker processes.

**Random catalogs.** `cat.make_random` makes the randoms of a catalog: uniform on the sky inside the footprint
of the lightcone, with redshifts drawn (with replacement) from the redshifts of the data. A contaminated
catalog gives three data catalogs, each with its own randoms:

| `which` | data catalog | `cat.subset(which)` keeps `id_outcome` |
|---|---|---|
| `"all"` | galaxies and interlopers | every one |
| `"galaxies"` | galaxies with a correct redshift | 1 (multi-line), 2 (single-line, correct) |
| `"interlopers"` | galaxies with a wrong redshift | 3 (single-line, misidentified) |

```python
randoms = cat.make_random(which="all", n_times=10)             # list of 10 catalogs: ra, dec, redshift_assigned
galaxies, interlopers = cat.subset("galaxies"), cat.subset("interlopers")      # the data that go with the randoms

cat.make_random(which=["all", "galaxies", "interlopers"], n_times=10, write_to="randoms/halpha.hdf5")
# -> randoms/halpha_all_00.hdf5 ... halpha_all_09.hdf5, halpha_galaxies_00.hdf5 ..., halpha_interlopers_00.hdf5 ...
ran = Catalog.read("randoms/halpha_all_03.hdf5")
```

Randoms N times as numerous as the data are N catalogs of the size of the data, one file each. Catalog `k`
depends on `(seed, which, k)` only, so more can be added later (`first=10`) without remaking the others. A
cut on `ra` or `dec` made with `cuts=` applies to the randoms too. From the command line:
`romanmock random halpha.hdf5 -n 10 --which all galaxies interlopers -o randoms/halpha.hdf5`.

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
| 1. detectability | `romanmock.sensitivity`, `romanmock.detection` | `lambda_obs = lambda_0 (1+z_rsd)`; `ConstantFluxLimit` (primary limit 1.0e-16 erg/s/cm², secondary limit 4.0e-17) or `TabulatedFluxLimit` (wavelength dependent, `secondary_ratio`); `ThresholdDetection` or `GaussianNoiseDetection` |
| 2. bitmask | `romanmock.lines`, `romanmock.detection` | bit *i* = feature *i* of a `LineSet`; `n_det = popcount`; ordering written to the run metadata |
| 3. redshift | `romanmock.confusion`, `romanmock.redshift` | multi-line: `z_assigned = z_rsd` (optional `sigma_z`); single-line: draw identity *j* from `C_ij` (+ fail column), `z_assigned = lambda_i,0 (1+z_rsd)/lambda_j,0 - 1` |
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
columns (`<feature>_flux` in erg/s/cm², ...).

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
two-tier rule). `romanmock census` writes it to a file and notebook 01 plots it. With one threshold at 1.0e-16 the mock has 1.5 times the forecast Hα
counts over 0.5 < z < 2.0 and 3 times the [O III] counts over 1.0 < z < 3.0.

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

**Sky area.** Give it if you know it: `BaseCatalog(path, area_deg2=2500.0)` or `make_catalog(area_deg2=2500.0)`.
Otherwise it is estimated, from the sky patches of the mock (2062.65 deg² for this one) or, when the mock has
no patch decomposition, from the galaxies in the HEALPix index. `cat.info["area_source"]` records which:
`user`, `footprint` or `healpix`. The file-based route always uses the HEALPix estimate (`sky_area_deg2`).

## Production data and notebooks

The products for the mock `sdss_feniks_hizels_260710_09_07_2026` are in
`/global/cfs/cdirs/m4943/Simulations/RomanMocks/sdss_feniks_hizels_260710_09_07_2026/`:

| directory | content | made by |
|---|---|---|
| `base/` | base catalog: 64.6 M galaxies with a line above 1.0e-16 erg/s/cm², 66 slices, 7.6 GB | `sbatch scripts/production.sh` |
| `census/` | `dndz_census_z0p5_3p0.npz`: dN/dz per line for primary limits 1.0e-16, 1.5e-16 and secondary limits none, 8e-17, 4e-17 | `sbatch scripts/production.sh` |
| `catalogs/` | `halpha_z1p0_1p9.hdf5`: Hα sample at 1.0 < z < 1.9 | notebook 02 |
| `randoms/` | `halpha_z1p0_1p9_<which>_<index>.hdf5`: 10 random catalogs each for all, galaxies and interlopers | notebook 03 |

`notebooks/` holds three executed examples, all of which run on a login node:

1. `01_dndz_flux_limits.ipynb` – dN/dz per line and the single-line fraction for the primary and secondary flux limits.
2. `02_build_catalog.ipynb` – a catalog from the base catalog: flux limits, confusion matrix, line and redshift selection, the line report, writing and reading.
3. `03_build_randoms.ipynb` – the random catalogs of that catalog for the contaminated sample, the galaxies and the interlopers.

## Extending (hooks for the plan's comments)

* **Wavelength-dependent sensitivity** (comment 6): `sensitivity: {type: tabulated, file: curve.txt}`
  or register a class in `romanmock.sensitivity.SENSITIVITY_MODELS`. A model that defines
  `secondary_flux_limit(wave_obs)` gets the two-tier rule.
* **Detection scatter / source size / overlap**: implement `DetectionModel.detect(flux, wave_obs, rng)`
  and register it in `DETECTION_MODELS` (`GaussianNoiseDetection` is the first example).
* **Confusion depending on `lambda_obs`, S/N, photometry** (comment 3): implement
  `ConfusionModel.assign(true_index, context, rng)`; `context` carries `wave_obs`, `snr`,
  `flux`, `z_rsd` and any `redshift.context_columns` from the catalog (in the two-step workflow they must be
  in `base.columns`). `WavelengthBinnedConfusion` is the first example. Matrices can be loaded from YAML files.
* **Multi-line redshift errors / wrong multi-line identifications**: implement
  `MultiLineModel.assign(det, z_rsd, sel, rng)` (default `PerfectMultiLine(sigma_z)`).
* **Feature definition** (comment 2): change the `line_set` list; blends are sums of columns.
* **Footprints**: `BoxFootprint`, `HealpixFootprint` (pixel list or map), or any callable `(ra, dec) -> mask`.
* **Target dN/dz** (comment 5): `TargetDNDZ` table; `mock_dndz` can be smoothed; shortages
  are reported, never filled.

The default confusion matrix (`romanmock.confusion.DEFAULT_CONFUSION_ROWS`: true Hα called [O III] 5007 with
probability 0.5 %, true [O III] 5007 or [S III] 9532 called Hα with 5 %, true Hβ called [O III] 5007 with 5 %,
no failures) is an explicitly assumed scenario, not a Roman calibration. With the default secondary limit it
matters for about 6 % of the detected galaxies, the single-line ones.

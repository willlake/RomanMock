# romanmock

Build a Roman-like emission-line spectroscopic mock from a diffsky lightcone, following
`Roman_Mock_Plan_and_Comments.pdf`. Catalog access goes through
[OpenCosmo](https://github.com/ArgonneCPAC/OpenCosmo) and the
[diffsky](https://github.com/ArgonneCPAC/diffsky) loaders; the package adds only the
survey-specific steps (fluxes, detection, bitmask, redshift identification, sample selection).

## Installation

```bash
cd /global/u2/w/willlake/Packages/RomanMock
pip install -e .            # opencosmo, hdf5plugin, astropy, pyyaml, ... ; diffsky/dsps optional
pytest tests                # unit tests; the real-catalog test is skipped if the mock is absent
```

On Perlmutter login nodes set `export JAX_PLATFORMS=cpu` (diffsky imports JAX; the CLI does
this by default).

## Input

A diffsky mock directory such as
`/global/cfs/cdirs/m4943/Simulations/sdss_feniks_hizels_260710_09_07_2026/`: 67 lightcone
slices `lc_cores-<step>.diffsky_gals.hdf5` (z = 0-4, 324 M galaxies) plus the auxiliary
`*_ssp_data.hdf5`, `*_param_collection.hdf5`, `*_t_table.hdf5`, `*_transmission_curves.hdf5`.
The line columns (`Ba_alpha_6563`, `Ba_beta_4861`, `O_III_4363`, `O_III_4959`, `O_III_5007`,
`O_II_3726`, `O_II_3729`, `S_III_6312`, `S_III_9069`, `S_III_9532`) are dust-attenuated,
continuum-subtracted luminosities in erg/s, one column per transition.
`redshift_obs` includes the line-of-sight peculiar velocity (`z_rsd` of the plan);
`redshift_true` is the cosmological redshift.

`romanmock info <catalog_dir>` lists the slices, their redshift ranges and the cosmology.

## Pipeline

| Plan step | Module | What it does |
|---|---|---|
| fluxes | `romanmock.fluxes`, `romanmock.cosmology` | `F = L / (4 pi D_L^2)` with the catalog cosmology from OpenCosmo (`astropy` backend) or `dsps.cosmology.luminosity_distance_to_z` (`dsps` backend; agree to 5e-6) |
| 1. detectability | `romanmock.sensitivity`, `romanmock.detection` | `lambda_obs = lambda_0 (1+z_rsd)`; `ConstantFluxLimit` (initial model) or `TabulatedFluxLimit` (wavelength dependent); `ThresholdDetection` or `GaussianNoiseDetection` |
| 2. bitmask | `romanmock.lines`, `romanmock.detection` | bit *i* = feature *i* of a `LineSet`; `n_det = popcount`; ordering written to the run metadata |
| 3. redshift | `romanmock.confusion`, `romanmock.redshift` | multi-line: `z_assigned = z_rsd` (optional `sigma_z`); single-line: draw identity *j* from `C_ij` (+ fail column), `z_assigned = lambda_i,0 (1+z_rsd)/lambda_j,0 - 1` |
| 4. samples | `romanmock.selection` | footprint, `redshift_assigned` range, tracer (`membership: assigned` = brightest line, disjoint samples; `detected` = any detected line for multi-line galaxies), outcome quality, population thinning, downsampling with `p_keep = target/mock` |
| checks | `romanmock.diagnostics` | empirical confusion, contamination fraction `f_{i->j}(B)`, detection efficiency vs a reference population, identification efficiency given detection, final-sample completeness, `dN/dz` |

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

### Outputs

Stage A writes one OpenCosmo file per input slice, `lc_cores-<step>.<tag>.hdf5`, that
`opencosmo.open(*files)` reassembles into a Lightcone (header, cosmology and HEALPix index
are preserved). The default columns are

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

OpenCosmo adds its own `redshift` column (equal to `redshift_true`). `output.passthrough_columns`
copies further input columns (`gal_id`, `logsm_obs`, `roman_F158`, ...); `store_fluxes`,
`store_wave_obs`, `store_snr`, `store_measured_flux` add per-feature columns (`<feature>_flux` in
erg/s/cm^2, ...). A `<tag>_metadata.yaml` sidecar records the configuration, the bit ordering, the
outcome codes and the model parameters, and `<tag>_manifest.yaml` the completed slices and the
configuration hash.

Stage B writes `samples/<name>.hdf5` with `output.sample_columns` (default `ra, dec,
redshift_true, redshift_rsd, redshift_assigned, id_outcome`; `null` keeps every parent column),
plus a metadata file with `p_keep` per bin when thinning was applied. Note that the diffsky
input column `redshift_obs` is z_rsd, not a measured redshift; it is never written under that
name to avoid confusion with the assigned redshift.

## Usage

```bash
romanmock lines                                   # print the bit ordering of the presets
romanmock example-config --catalog DIR --output OUT > my.yaml
romanmock run configs/roman_grism_baseline.yaml   # stage A over all configured slices
romanmock run my.yaml --steps 219 224             # a subset of slices (job arrays: scripts/run_slices.sh)
romanmock run my.yaml --workers 32                # whole mock on one exclusive CPU node (one slice per process)
romanmock run my.yaml --workers auto              # fit the memory budget (node memory or the 30 GB login-node cap)
romanmock summary my.yaml                         # outcome counts + empirical confusion
romanmock select my.yaml                          # stage B: analysis samples
```

Python:

```python
import numpy as np, opencosmo as oc
from romanmock import RomanMockPipeline, RomanMockConfig, add_line_fluxes

pipe = RomanMockPipeline.from_yaml("configs/roman_grism_baseline.yaml")
pipe.run(steps=[219])                       # writes lc_cores-219.romanmock.hdf5
pipe.run(n_workers=32)                      # every slice, one process per slice
parent = pipe.open_parent()                 # OpenCosmo Lightcone of every processed slice
halpha = parent.filter(oc.col("line_assigned") == pipe.line_set.bit("Ha"))

# or just add flux columns to any OpenCosmo dataset
ds = oc.open(".../lc_cores-219.diffsky_gals.hdf5")
ds = add_line_fluxes(ds, "roman_transitions")   # Ha_flux, Hb_flux, ... in erg/s/cm^2
```

Everything in `process_arrays` works on plain dictionaries of numpy arrays, so the physics
can be exercised without touching the catalog (see `tests/`).

## Reference dN/dz and population thinning

`romanmock.reference_dndz` holds Tables 1 and 2 of Wang et al. (2022, arXiv:2110.01829): the
Roman HLSS forecast of Hα and [O III] emitters per deg² in Δz = 0.1 bins, for six flux limits
(5e-17 to 2e-16 erg/s/cm²) and two dust models (A_V = 1.65, 1.92). These are *true emitters above
the flux limit*, before completeness and purity ("observational efficiency is not included"), so
they are matched to the mock's detected-line populations in bins of true redshift, before any
assigned-redshift selection (`population_thinning:` in the config; on by default in the baseline).
Interlopers are then thinned consistently with the population they belong to. The transcription is
checked against the paper's summary rows by a unit test.

A target spec is `{reference: wang2022, line: Ha, flux_limit: 1e-16, dust_av: 1.65}` (line and
flux limit default to the population / the run's constant flux limit), `{file: dndz.txt}` or
`{z: [...], dndz: [...]}`; the same specs drive the per-sample `downsample` on assigned redshift.
`poly_degree` fits a smooth selection function instead of per-bin ratios, so the realisation's
own radial fluctuations are not divided out. Shortages (target above mock, including empty bins)
are reported, never filled. For this mock at 1e-16 the Hα targets are attainable for z > 0.8 and
the [O III] targets for z < 2.85.

## Notebooks

`notebooks/` holds four executed tutorials (regenerate with `python notebooks/build_notebooks.py --execute`;
edit the builder, not the `.ipynb` files):

1. `01_explore_diffsky_catalog.ipynb` – slices, columns, units, rest wavelengths, luminosities to fluxes with `add_line_fluxes`, which lines fall in the grism.
2. `02_pipeline_step_by_step.ipynb` – fluxes, detection and bitmask, confusion matrix, redshift assignment on arrays; input versus empirical confusion.
3. `03_run_and_build_samples.ipynb` – `RomanMockPipeline.run` on three slices, the parent catalog, contamination and completeness, downsampling a sample to a target dN/dz.
4. `04_extending_models.ipynb` – tabulated sensitivity, noisy detection, an S/N-dependent confusion model registered for YAML, custom line sets.
5. `05_full_run.ipynb` – all 67 slices in parallel on an exclusive CPU node, then dN/dz per assigned line, the (z_rsd, z_assigned) plane, contamination, completeness, population thinning to the Wang et al. (2022) forecast and two HLSS-like samples.

Memory: one stage-A worker peaks at 4.4 GB on the largest slice; `build_samples` on the full 64 M-galaxy
parent catalog peaks at about 22 GB while writing a 16 M-galaxy sample. `--workers auto` /
`romanmock.io.suggest_n_workers()` size the pool from the node's available memory and any cgroup cap
(NERSC login nodes: 30 GB per user, so at most 5 workers there). Notebook 5 needs a compute node:
`sbatch scripts/execute_notebooks.sh 05`.

`CLAUDE.md` summarises the layout and conventions for anyone (or any assistant) editing the package.

## Extending (hooks for the plan's comments)

* **Wavelength-dependent sensitivity** (comment 6): `sensitivity: {type: tabulated, file: curve.txt}`
  or register a class in `romanmock.sensitivity.SENSITIVITY_MODELS`.
* **Detection scatter / source size / overlap**: implement `DetectionModel.detect(flux, wave_obs, rng)`
  and register it in `DETECTION_MODELS` (`GaussianNoiseDetection` is the first example).
* **Confusion depending on `lambda_obs`, S/N, photometry** (comment 3): implement
  `ConfusionModel.assign(true_index, context, rng)`; `context` carries `wave_obs`, `snr`,
  `flux`, `z_rsd` and any `redshift.context_columns` from the catalog.
  `WavelengthBinnedConfusion` is the first example. Matrices can be loaded from YAML files.
* **Multi-line redshift errors / wrong multi-line identifications**: implement
  `MultiLineModel.assign(det, z_rsd, sel, rng)` (default `PerfectMultiLine(sigma_z)`).
* **Feature definition** (comment 2): change the `line_set` list; blends are sums of columns.
* **Footprints**: `BoxFootprint`, `HealpixFootprint` (pixel list or map), or any callable `(ra, dec) -> mask`.
* **Target dN/dz** (comment 5): `TargetDNDZ` table; `mock_dndz` can be smoothed; shortages
  are reported, never filled.

The shipped confusion matrices are explicitly assumed scenarios, not Roman calibrations.

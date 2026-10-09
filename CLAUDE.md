# CLAUDE.md — romanmock

Roman-like emission-line spectroscopic mock built from a diffsky lightcone (version 0.3.0). The design
follows `/global/u2/w/willlake/Projects/Roman/RomanMock/Roman_Mock_Plan_and_Comments.pdf`
(read it before changing the physics). README.md documents the user-facing behaviour; keep both files
in step with the code.

## Environment

- Conda env: `Roman-dev` (`/pscratch/sd/w/willlake/desi/conda/envs/Roman-dev`, Python 3.13).
  Package is installed editable: `pip install -e .`. There is no pandas in the env; use astropy Tables.
- Always `export JAX_PLATFORMS=cpu` on login nodes (diffsky imports JAX and probes CUDA).
  The CLI sets this itself; scripts and notebooks must do it before importing diffsky.
- Reading the mock needs `import hdf5plugin` (done in `romanmock.io`; opencosmo does it too).
- Tests: `python -m pytest -q tests` (65 tests, ~6 min: most of the time is OpenCosmo writing files in the
  real-catalog tests). Tests on the real mock skip when it is absent; point `ROMANMOCK_TEST_CATALOG`
  elsewhere to use another mock. `python -m pytest -q tests -k "not real and not two_step and not slicewise"`
  is the quick array-level subset.
- NERSC login nodes cap each user at 30 GB (user-slice cgroup) and 12.5% CPU; `suggest_n_workers()` in
  `io.py` reads that cap. Anything that reads the input mock (`run`, `base`, `line_census`) belongs on a CPU
  node (`sbatch`, account m4943, `-C cpu -q debug`); `assign`, `select`, `summary`, `compress-index` run on a
  login node. Scripts that use a process pool need an `if __name__ == "__main__":` guard (spawn context).
- A PreToolUse hook blocks shell globs such as `du *.hdf5` on Perlmutter; measure sizes with Python.

## Data

- Input mock: `/global/cfs/cdirs/m4943/Simulations/sdss_feniks_hizels_260710_09_07_2026/`
  (67 slices `lc_cores-<step>.diffsky_gals.hdf5`, 324 M galaxies, z 0–4, 340 GB, 174 columns; up to 10 M rows
  per file). Steps 219/224 (z ≈ 1.15–1.26) are handy for quick Hα tests; the smallest files are at z < 0.4,
  where only [S III] is inside the grism bandpass.
- Footprint: all right ascensions at declination 57.6–75.9 deg, 2062.5 deg² (5 % of the sky).
- Line columns are per-transition, dust-attenuated luminosities in erg/s. `redshift_obs` includes
  peculiar velocity (the plan's z_rsd); `redshift_true` is cosmological.
- Rest wavelengths come from the mock's `*_ssp_data.hdf5` (`ssp_emline_wave`); `lines.py` carries
  the same values as a fallback.
- Cosmology: OpenCosmo builds it from `header/simulation/cosmology` (h 0.6766, Ω_m 0.30964, Ω_b 0.04897,
  σ8 0.8102, n_s 0.9665; T_CMB 0). `header/simulation/parameters` (box 3400 Mpc/h, 10752³) is HACC Last Journey,
  which is what diffsky's `haccsims.py` lists; the `simulation/name` attribute and the embedded raw parameter
  files describe a 128 Mpc/h hydro run instead, apparently a mislabel from `reformat_hacc` (same cosmology,
  so nothing downstream is affected; worth confirming with Andrew Hearin).
- Properties of the mock that show up in results: [O III] 5007/4959 ≈ 3.04 and [S III] 9532/9069 ≈ 2.53 are
  nearly fixed, Hα/Hβ has a median of 4.0; Hα is almost always the brightest line where it is in the bandpass.

## Layout

| Module | Role |
|---|---|
| `roman.py` | defaults: grism range 10000–19300 Å, primary limit 1.0e-16, secondary limit 4.0e-17, S/N at limit 6.5 |
| `lines.py` | `LineFeature`/`LineSet`, presets, bit ordering (bit i = feature i, uint32) |
| `cosmology.py`, `fluxes.py` | erg/s → erg/s/cm² via the OpenCosmo header cosmology (astropy or dsps backend) |
| `sensitivity.py` | `SensitivityModel` + `DetectionModel` protocols and registries; two-tier rule (`apply_two_tier`) |
| `detection.py` | `detect_lines`, `pack_bitmask`, `popcount`, `DetectionResult` |
| `confusion.py` | `ConfusionMatrix` (fail column), `WavelengthBinnedConfusion`, `DEFAULT_CONFUSION_ROWS`, `build_confusion` |
| `redshift.py` | `assign_redshifts`, `IdOutcome` codes, `MultiLineModel` hook |
| `selection.py` | footprints, `SampleDefinition`, `sample_mask`, `TargetDNDZ`, thinning helpers, `sky_area_deg2` |
| `samples.py` | slice-wise stage B: `process_slice`, `run_pass`, `thinning_keep`, `sample_keep` |
| `pipeline.py` | `RomanMockPipeline`: `process_arrays`/`process_fluxes` (pure numpy) → `run` (stage A, from the mock or from the base catalog), `build_base`, `build_samples` |
| `footprint.py` | `PatchFootprint`: union of the theta-phi sky patches of the lightcone (`from_mock`, `sample`, `contains`, `area_deg2`) |
| `randoms.py` | `make_random`: randoms of a `Catalog` for `which` = all / galaxies / interlopers; redshifts drawn from the data |
| `catalog.py` | `BaseCatalog.make_catalog` -> `Catalog`: step 2 in memory from the base catalog (the everyday route), `Catalog.write/read` |
| `reference_dndz.py` | Wang et al. (2022) Tables 1–2 with uncertainties |
| `fluxdist.py` | `flux_distribution` / `FluxDistribution`: log-flux histograms of every galaxy per line and redshift bin, no cut; quantiles, fraction above a threshold |
| `census.py` | `line_census` / `LineCensus`: dN/dz of emitters above flux limits, single/multi-line, no confusion |
| `diagnostics.py`, `plotting.py` | checks (confusion, contamination, completeness, dN/dz); notebook plot style |
| `config.py`, `cli.py`, `io.py` | YAML config dataclasses, CLI, OpenCosmo I/O, manifests, index compression, memory budget |

CLI commands: `info`, `lines`, `example-config`, `run`, `base`, `census`, `flux-distribution`, `catalog`, `random`, `assign`,
`summary`, `select`, `compress-index`.
Scripts: `production.sh` (sbatch: base catalog + census from `configs/production.yaml`), `run_slices.sh` (job array
for `run`), `example_analysis.py`.

## Conventions

General:

- Keep the physics in array-level functions (`process_fluxes` and below) that take dicts of numpy
  arrays; only the dataset/file methods touch OpenCosmo. Add tests at the array level.
- New models are classes registered in `SENSITIVITY_MODELS`, `DETECTION_MODELS`,
  `CONFUSION_MODELS` (or the footprint factory) and selected by `type:` in YAML. Do not add
  `if type == ...` branches inside the pipeline.
- A `LineSet` fixes the bit ordering; never reorder presets (existing outputs depend on it).
  Add new presets instead.
- Confusion matrices ship as assumed scenarios only; never describe them as Roman calibrations. The default
  (`confusion: {type: default}`) is `romanmock.confusion.DEFAULT_CONFUSION_ROWS` (Ha->OIII_5007 0.5 %,
  OIII_5007->Ha 5 %, SIII_9532->Ha 5 %, Hb->OIII_5007 5 %, no failures, `default_p_fail` 0), translated to
  the line set's names.

Detection (two-tier since 0.3.0):

- A galaxy needs one line above the primary limit (`flux_limit`, default 1.0e-16) inside the bandpass; its other
  lines are then detected above `secondary_flux_limit` (default 4.0e-17, `null` = off, must not exceed the primary
  limit; `secondary_ratio` for `TabulatedFluxLimit`, off by default). Implemented in the detection models via
  `sensitivity.apply_two_tier`; sensitivity models without `secondary_flux_limit()` stay single-tier.
- `Detection.primary` / `DetectionResult.primary` flag lines above the primary limit (not written to disk).
  `line_true` (brightest detected line) is always a primary line, so population definitions are unaffected.
- In `census.py` a plain number is a single threshold (comparable to Wang et al.); a `(primary, secondary)`
  pair applies the two-tier rule, and the accessors take `secondary=`.

Two-step workflow:

- What the user wants (stated 2026-09-28): step 1 stores everything step 2 could need; step 2 gives complete
  freedom over flux limits, confusion matrix, redshift cut, lines, seed, with no bookkeeping in the way. That is
  `catalog.BaseCatalog.make_catalog(...)`: in memory by default, `cat.write(path)` / `write_to=` for disk,
  `downsample=None` by default and `"wang2022"` on request. No manifests, no hashes, no refusal on changed
  settings; output files are replaced. Default to this route when asked for a catalog; the pipeline route
  (`assign` + `select`, OpenCosmo files, manifests) is for when an OpenCosmo lightcone on disk is needed.
- Randoms: `cat.make_random(which, n_times, write_to)`. Footprint = the 60 patches in `lc_patch`, rectangles from
  `lc_cores-decomposition.txt` in the mock directory; `dec = 90 - theta`, `ra = phi - 180` (offset measured in
  `PatchFootprint.from_mock`, cached in `<base>/<tag>_footprint.json`). Exact area 2062.65 deg², which is `base.area_deg2` and
  `cat.area_deg2` (the HEALPix estimate, 2063.0, is the fallback and what `select` uses). 0.006 % of the galaxies lie outside (satellites).
  Stream of random catalog k: `(seed, 30000, kind index, k)`. One file per multiple of the data size.
  Neither OpenCosmo nor diffsky makes randoms for a footprint; diffsky has
  `experimental.lc_utils.mc_lightcone_random_ra_dec` for a single rectangle.
- `make_catalog(lines=..., membership="detected")`: sample of a line = galaxies with a correct redshift in which the
  line is detected (any line of a multi-line galaxy, `selection.tracer_membership`) plus the single-line galaxies
  taken for it. `membership="assigned"` = `line_assigned` only (brightest line), the default of the file route
  (`SampleConfig.membership`). In the mock [O III] 5007 is never the brightest line at 1 < z < 1.9, so with
  "assigned" an [O III] sample there holds interlopers only. The line report counts detected / identified /
  brightest / mistaken / interlopers per line.
- Sky area: the user's value when given (`BaseCatalog(..., area_deg2=)`, `make_catalog(area_deg2=)`, `romanmock
  catalog --area`), else estimated (footprint, then HEALPix); `base.area_source` / `cat.info["area_source"]` say which.
  The area affects dN/dz and the downsampling only, not the randoms (they use the footprint geometry).
- The footprint is never assumed. `BaseCatalog.footprint` reads `lc_patch` from the three most populated input
  slices and the rectangles from the decomposition file of that mock, checks the result against the galaxies of
  the largest base slice (`PatchFootprint.check`: more than 99 % inside, under 1 % of interior HEALPix pixels
  empty, else ValueError), and caches it with the identity of the base catalog (`_identity`), so a rebuilt base
  catalog or another mock gets its own footprint. If the mock has no decomposition, `area_deg2` warns and falls
  back to the HEALPix estimate. The numbers 60 patches / 2062.65 deg² appear only in tests and documentation.
- When replacing a section of a document by slicing between two markers, check that the markers are adjacent
  (on 2026-09-28 a slice removed the sections Conventions to OpenCosmo facts of this file; restored from git).
- Mock line model facts (from notebook 04): Hβ/[O III]5007 = 0.45 (0.42–0.47) and [O III]4959/5007 = 0.329 nearly fixed;
  [S III]9532/Hα median 0.47 (0.35–0.8); [O III]5007/Hα median 0.44; Hβ/Hα 0.19. Median Hα flux 3–5e-17 at every z,
  so the flux histogram per dex is flat between 1e-17 and 1e-16 (no faint-end rise: mock population).
- Line report: `make_catalog(verbose=True)` prints `cat.line_report()`; the numbers are in
  `cat.info["line_report"]` (`lines`, `interlopers`). Geometry in `catalog.line_report` (potential interlopers:
  line i inside the bandpass that passes the redshift cut when taken for j; lightcone bounds padded by
  `RSD_MARGIN`), counts in `catalog.line_counts` (per slice in the workers, after the thinning when downsampling).
- `BaseCatalog.read_slices(columns, z_range)` streams the stored columns slice by slice (no workers, < 1 GB): the
  way to histogram or scan the base catalog. Notebook 01's flux histograms use it: 135 s and 0.8 GB peak for the
  1e-17 base on a CPU node, so the notebook runs on a login node whatever the base limit.
- `_catalog_slice` processes a slice in chunks of `CHUNK_ROWS` (1 M rows): the memory of a worker no longer grows
  with the slice (8.4 M rows per slice at 1e-17 broke the pool on a login node). Chunking is exact because the
  random draws are consumed in row order (`test_chunked_slices_equal_one_pass`).
- `make_catalog` reuses `detect_lines` / `assign_redshifts` with the pipeline's random streams (`(seed, step)` for
  the confusion draws, `(seed, 20000, step)` for the thinning, one uniform per detected galaxy in slice order), so
  it equals `assign` + `select` row for row (`test_catalog.py`). Thinning is done in one pass: workers return
  the population histograms plus the selected rows with their uniform deviates, the main process applies `p_keep`.
- The base bitmask column is `line_bitmask_base` (`BASE_BITMASK_COLUMN`), never `line_bitmask`: see the OpenCosmo
  name-collision fact below. `assign` refuses base catalogs of the older layout.
- Assigned redshifts: multi-line galaxies get `redshift_rsd` exactly (never `redshift_true`); single-line correct
  ones get it to rounding (4e-16); interlopers get `rest_true (1 + z_rsd) / rest_assigned - 1`; failed ones NaN.
- `build_base` (`romanmock base`) writes the base flux catalog (`BaseConfig`: dir, tag, `columns` default
  gal_id/ra/dec/redshift_true, `store_bitmask`, `flux_dtype`, `compress_index`) with its own manifest
  (`kind: base`, the `base_description()` incl. resolved line set and sensitivity). The base selection is always
  the deterministic threshold rule on true fluxes. Every line flux is stored for every base galaxy.
- `config.source = "base"` makes `run` read the base catalog instead of the mock (`romanmock assign`, which also
  writes the effective `config.yaml` to the output directory). The pipeline then takes the line set (rest
  wavelengths) from the base manifest, validates limits/bandpass/columns in `_check_base`, and hashes
  `base_config_hash` into `config_hash()`.
- Keep the property that assign at the base limits equals a direct run bit for bit (`test_two_step`, and verified
  on all 64.6 M rows); it relies on float64 fluxes and on rng draws happening only for single-line rows.

Stage B (slice-wise):

- `build_samples` makes a picklable `plan` and runs passes of `samples.process_slice` over the parent files:
  population histograms -> sample histograms -> selection + writing, with a range pass first when bins are not
  fixed by the config. RNG per slice: `(seed, 20000, step)` for population thinning, `(seed, 10000 + k, step)` for
  sample k, one uniform per galaxy, so passes, workers and layouts agree.
- Output `samples/<name>/lc_cores-<step>.<name>.hdf5` (`io.open_sample`); `sample_layout: single` merges through
  OpenCosmo. `build_samples` returns directories (or files for `single`), only for non-empty samples.
- Never go back to `parent.take_rows(global_idx)` + one `oc.write`: measured 20 GB and 330 s for a 17 M-row
  sample, because OpenCosmo holds every slice's uncompressed index (268 MB each) in memory.
- The array-level `population_thinning_mask` / `downsample_to_dndz` remain for in-memory use and tests; they share
  `dndz_from_counts`, `keep_probability`, `probability_per_galaxy` with the slice-wise code. Sample downsampling
  smooths the mock dN/dz by default (`smooth_bins` 1.0), so per-bin agreement with the target is not exact.

File format:

- Outcome codes (`IdOutcome`) and output column names are part of the file format; changing
  them requires bumping `_version.py` and the metadata sidecar. The redshift columns
  are `redshift_true`, `redshift_rsd` (input `redshift_obs`, peculiar velocity included) and
  `redshift_assigned`; never write the input name `redshift_obs` to an output. Default parent
  columns: ra, dec, redshift_true, redshift_rsd, redshift_assigned, id_outcome + the identification
  columns (`store_identification`); default sample columns are the first six (`sample_columns`). There is no
  `is_interloper` column (interloper = `id_outcome == 3`). Fluxes are off by default in parent catalogs.
- Sky area: `sky_area_deg2` uses the finest HEALPix index level with at least `min_per_pixel` (50) galaxies
  per occupied pixel, edge pixels counted half (2062.5 deg^2 for this mock); plain counting
  at nside 256 overestimates by 2.5 %, and a single sparse slice at nside 1024 underestimates by 5 % (holes
  look like edges), so estimate the area from the whole lightcone or the parent catalog, not one slice.

Randomness and provenance:

- Per-slice randomness is seeded with `(seed, step)` so results do not depend on which files a
  job processes or on `n_workers`. Keep that property (there are tests for serial == parallel).
- Parallel workers rebuild the pipeline from `RomanMockPipeline._worker_spec()` (config dict with the
  line set resolved); anything a worker needs must be serialisable through that dict.
- `pipeline.config_hash()` hashes `scientific_config()`; the manifest `<tag>_manifest.yaml` records completed
  slices and every output file carries `romanmock_config_hash` in its root attrs. Anything that changes the
  content of a slice output must be part of `scientific_config()` (or `base_description()`), otherwise reruns
  will silently reuse stale outputs. `compress_index` and `sample_layout` are deliberately not hashed.
  `parent_files()` validates hash and completeness; `expected_steps()` comes from `config.catalog` (or the base
  manifest), so a partial run (`--steps`) needs `--allow-partial` in stage B.
- Slices with no selected galaxy write a `.empty` marker (OpenCosmo cannot re-open an empty file).

## Review items

- Multi-component features: `LineFeature.blend` decides between blend detection and per-transition detection
  (`detect_lines(..., component_flux=...)`); the pipeline passes component fluxes automatically, and the base
  catalog stores the component fluxes of separated features.
- `SampleDefinition.membership` ("assigned" | "detected") is the tracer definition of review item M2;
  `diagnostics` separates detection efficiency, identification efficiency and final-sample completeness (M3).
- The external review lives at `Projects/Roman/RomanMock/RomanMock_Bug_Fixes_and_Methodology.md`; B1-B6 are
  fixed and tested in `tests/test_review_fixes.py`, M1 is addressed by `poly_degree`/`smooth_bins` (an ensemble
  selection function is still the recommended end state), M2/M3 by the options above.

## Reference dN/dz

- `reference_dndz.py` transcribes Wang et al. (2022) Tables 1-2 (arXiv:2110.01829). They are true
  emitters above the flux limit *without* observational efficiency; match them to the detected-line
  populations in true redshift (`population_thinning`), not to assigned-redshift samples. A test
  checks the transcription against the paper's summary rows; keep it when editing the numbers.
- Only the six tabulated flux limits are valid; there is no interpolation in flux limit on purpose. Nothing is
  tabulated below 5e-17, so default thinning fails for such a primary limit.
- `wang2022_errors` returns the tables' +-1 sigma (jackknife + cosmic variance); both central values and errors
  were checked bin by bin against the arXiv LaTeX source (e-print 2110.01829, tables `tab:Halpha`/`tab:OIII`).
- Mock versus forecast with one threshold at 1.0e-16 (A_V 1.65): Hα 1.5x over 0.5 < z < 2.0 (0.55x at z 0.5,
  6x at z 1.8), [O III] 3x over 1.0 < z < 3.0; at 1.5e-16 Hα 1.9x. Shortage warnings at z < 0.8 are expected.

## OpenCosmo (1.3.11) facts that shaped the code

- `oc.open(one_file)` returns a `Lightcone` (dict step → Dataset), no `close()`; it exposes a
  synthetic `redshift` column equal to `redshift_true`. The global row order of a multi-file lightcone is the
  concatenation of `.values()`, and `take_rows` uses it.
- `filter()` after `with_new_columns` drops the new columns' descriptions → the pipeline selects
  rows with `take_rows(int64 index)` first, then adds columns. Boolean masks are rejected.
- `with_new_columns` accepts numpy arrays or astropy Quantities; units and descriptions survive
  `oc.write` and re-open. Some input columns (`gal_id`) have no description.
- `with_new_columns(name=...)` on a dataset whose file already has a column `name` is accepted and reads back
  correctly in memory, but `oc.write` then writes the *stored* column. This silently put the base catalog's
  bitmask into `assign` outputs whenever the secondary limit differed from the base one (found 2026-09-28; only
  `line_bitmask` was affected). Never add a column under a name that exists in the source file, even unselected.
- `oc.write` needs a `.hdf5` extension and stores the full HEALPix spatial index (levels 0-10) uncompressed in
  every output file: a fixed ~268 MB per slice file (two int64 per pixel of the whole sky); its serial writer has
  no compression option, and writing a multi-slice lightcone holds all those indices in memory.
  `io.write_dataset` therefore writes to a hidden temporary file and copies it with the `index` groups
  gzip-compressed (`compress_spatial_index`, ~1 MB); data columns are copied untouched. Controlled by
  `output.compress_index` / `base.compress_index` (default true). `romanmock compress-index <dir>` converts
  existing files in place. OpenCosmo reads both layouts (the input mock's index is compressed too).
- `dsps.cosmology.luminosity_distance_to_z` is scalar (wrap with `jax.vmap`).

## Measured resources (full lightcone, base limit 1.0e-16, 64.6 M detected galaxies)

| step | where | wall | memory per worker | disk |
|---|---|---|---|---|
| `base` / `run` | CPU node, 32 workers | ~1 min | 5.7 / 4.4 GB | base 7.6 GB, parent 3.2 GB |
| `make_catalog` | login node, 8 workers | 15–20 s (first `downsample`: +40 s for the area, then cached) | 3.3 GB | none; Hα sample 7.8 M rows written: 0.29 GB |
| `assign` | login node, 4–8 workers | 1–2 min | 2.4 GB | 3.2 GB (1.0e-16), 2.3 GB (1.5e-16) |
| `select` (thinning + 2 samples) | login node, 4 workers | 3.4 min | 1.7 GB | Hα sample 17 M rows: 0.73 GB |
| `line_census` (42 slices, 6 limits) | CPU node, 32 workers | 0.8 min | ~5 GB | — |

Data columns cost 48 B per galaxy for the default parent columns and 116 B for the base catalog. At 1e-17 the
mock has 203 M galaxies with a line in 0.5 < z < 3.0 (74 % of all), against 59 M at 1e-16 and 41 M at 1.5e-16.

## Production data (`/global/cfs/cdirs/m4943/Simulations/RomanMocks/<mock version>/`)

- `base/` (limit 1.0e-17, 222.0 M galaxies, 25.9 GB; rebuilt 2026-09-30 by `scripts/production.sh`, job 59129636, 2 min
  on a CPU node; the 1.0e-16 base of 2026-09-28 had 64.6 M galaxies, 7.6 GB), `census/dndz_census_z0p5_3p0.npz`,
  `census/flux_distribution.npz` (job 59160266, 1 min), `catalogs/Every_z1p5_1p9.hdf5` (notebook 02),
  `randoms/Every_z1p5_1p9_<which>_<index>.hdf5` (notebook 03).
- `$PSCRATCH/romanmock_demo` holds the outputs of the earlier tutorial notebooks (scratch, purged in time).

## Working style

- Plots (notebooks) use `romanmock.plotting` (fixed categorical order, single axis, thin marks).
- `notebooks/` holds six production examples (01 dN/dz and flux limits, 02 catalog, 03 randoms, 04 line fluxes of the
  whole mock: quantiles, fractions above thresholds, companion lines; explains 01; 05 single-line galaxies of the base
  catalog against observed wavelength: 0–90/90–99.9 % flux quantiles and the stacked line mix above a threshold per 775 Å
  range, lines under 1 % in every range dropped; in the mock they are 92 % Hα because of the fixed line ratios; 06 stacked single-line/multi-line composition of the
  detected galaxies by line combination against redshift), edited as
  notebooks; there is no generator. The ten earlier tutorials and their generator were moved by the user to
  `~/Projects/Roman/RomanMock/Notebooks/` (2026-09-28). Keep notebooks minimal: only what is needed for the task.
- `configs/roman_grism_baseline.yaml` is the commented template for every config block; keep it complete (it was
  found truncated after the sensitivity block on 2026-09-28 and restored).
- The user commits; leave changes uncommitted in the working tree unless asked.

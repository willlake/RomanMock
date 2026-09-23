# CLAUDE.md — romanmock

Roman-like emission-line spectroscopic mock built from a diffsky lightcone. The design
follows `/global/u2/w/willlake/Projects/Roman/RomanMock/Roman_Mock_Plan_and_Comments.pdf`
(read it before changing the physics). README.md documents the user-facing behaviour.

## Environment

- Conda env: `Roman-dev` (`/pscratch/sd/w/willlake/desi/conda/envs/Roman-dev`, Python 3.13).
  Package is installed editable: `pip install -e .`
- Always `export JAX_PLATFORMS=cpu` on login nodes (diffsky imports JAX and probes CUDA).
  The CLI sets this itself; scripts and notebooks must do it before importing diffsky.
- Reading the mock needs `import hdf5plugin` (done in `romanmock.io`; opencosmo does it too).
- Tests: `python -m pytest -q tests` (~10 s). The real-catalog test skips when the mock is absent;
  point `ROMANMOCK_TEST_CATALOG` elsewhere to use another mock.
- Do not run the full 67-slice pipeline on a login node. On an exclusive CPU node use
  `romanmock run cfg.yaml --workers 32` / `pipe.run(n_workers=32)` (spawned processes, one slice each,
  2–5 GB peak per worker; ~10 min for the whole mock, I/O bound); otherwise `scripts/run_slices.sh`.

## Data

- Input mock: `/global/cfs/cdirs/m4943/Simulations/sdss_feniks_hizels_260710_09_07_2026/`
  (67 slices `lc_cores-<step>.diffsky_gals.hdf5`, 324 M galaxies, z 0–4, 340 GB; 2–9 M rows per file).
  Steps 219/224 (z ≈ 1.15–1.26) are handy for quick Hα tests; `lc_cores-475` is the smallest file.
- Line columns are per-transition, dust-attenuated luminosities in erg/s. `redshift_obs` includes
  peculiar velocity (the plan's z_rsd); `redshift_true` is cosmological.
- Rest wavelengths come from the mock's `*_ssp_data.hdf5` (`ssp_emline_wave`); `lines.py` carries
  the same values as a fallback.

## Layout and conventions

| Module | Role |
|---|---|
| `lines.py` | `LineFeature`/`LineSet`, presets, bit ordering (bit i = feature i, uint32) |
| `cosmology.py`, `fluxes.py` | erg/s → erg/s/cm² via the OpenCosmo header cosmology (astropy or dsps backend) |
| `sensitivity.py` | `SensitivityModel` + `DetectionModel` protocols and registries |
| `detection.py` | `detect_lines`, `pack_bitmask`, `popcount` |
| `confusion.py` | `ConfusionMatrix` (fail column), `WavelengthBinnedConfusion`, `build_confusion` |
| `redshift.py` | `assign_redshifts`, `IdOutcome` codes, `MultiLineModel` hook |
| `selection.py` | footprints, `SampleDefinition`, `downsample_to_dndz`, `sky_area_deg2` |
| `pipeline.py` | `RomanMockPipeline`: `process_arrays` (pure numpy) → `process_dataset` → `run` / `build_samples` |
| `config.py`, `cli.py`, `io.py`, `diagnostics.py` | YAML config dataclasses, CLI, OpenCosmo I/O, checks |

- Keep the physics in array-level functions (`process_arrays` and below) that take dicts of numpy
  arrays; only `process_dataset`/`run` touch OpenCosmo. Add tests at the array level.
- New models are classes registered in `SENSITIVITY_MODELS`, `DETECTION_MODELS`,
  `CONFUSION_MODELS` (or the footprint factory) and selected by `type:` in YAML. Do not add
  `if type == ...` branches inside the pipeline.
- A `LineSet` fixes the bit ordering; never reorder presets (existing outputs depend on it).
  Add new presets instead.
- Confusion matrices ship as assumed scenarios only; never describe them as Roman calibrations.
- Outcome codes (`IdOutcome`) and output column names are part of the file format; changing
  them requires bumping `_version.py` and the metadata sidecar.
- Per-slice randomness is seeded with `(seed, step)` so results do not depend on which files a
  job processes or on `n_workers`. Keep that property (there is a test for serial == parallel).
- Parallel workers rebuild the pipeline from `RomanMockPipeline._worker_spec()` (config dict with the
  line set resolved); anything a worker needs must be serialisable through that dict.

## OpenCosmo (1.3.11) facts that shaped the code

- `oc.open(one_file)` returns a `Lightcone` (dict step → Dataset), no `close()`; it exposes a
  synthetic `redshift` column equal to `redshift_true`.
- `filter()` after `with_new_columns` drops the new columns' descriptions → the pipeline selects
  rows with `take_rows(int64 index)` first, then adds columns. Boolean masks are rejected.
- `with_new_columns` accepts numpy arrays or astropy Quantities; units and descriptions survive
  `oc.write` and re-open. Multi-slice lightcones write as `<step>_data` groups in one file.
- `dsps.cosmology.luminosity_distance_to_z` is scalar (wrap with `jax.vmap`).

## Working style

- Plots (notebooks) use `romanmock.plotting` (fixed categorical order, single axis, thin marks).
- Notebooks live in `notebooks/`; rebuild/execute them with `python notebooks/build_notebooks.py`
  so outputs stay in sync with the code.

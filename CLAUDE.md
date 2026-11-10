# CLAUDE.md — romanmock

Roman-like emission-line spectroscopic mock built from a diffsky lightcone (version 0.4.0). The design
follows `/global/u2/w/willlake/Projects/Roman/RomanMock/Roman_Mock_Plan_and_Comments.pdf`
(read it before changing the physics). README.md documents the user-facing behaviour; keep both files
in step with the code.

## Environment

- Conda env: `Roman-dev` (`/pscratch/sd/w/willlake/desi/conda/envs/Roman-dev`, Python 3.13).
  Package is installed editable: `pip install -e .`. There is no pandas in the env; use astropy Tables.
- Always `export JAX_PLATFORMS=cpu` on login nodes (diffsky imports JAX and probes CUDA).
  The CLI sets this itself; scripts and notebooks must do it before importing diffsky.
- Reading the mock needs `import hdf5plugin` (done in `romanmock.io`; opencosmo does it too).
- Tests: `python -m pytest -q tests` (138 tests, ~10 min on a login node: most of the time is OpenCosmo writing files in the
  real-catalog tests). Tests on the real mock skip when it is absent; point `ROMANMOCK_TEST_CATALOG`
  elsewhere to use another mock. `python -m pytest -q tests -k "not real and not two_step and not slicewise"`
  is the quick array-level subset.
- NERSC login nodes cap each user at 30 GB (user-slice cgroup) and 12.5% CPU; `suggest_n_workers()` in
  `io.py` reads that cap. Anything that reads the input mock (`run`, `base`, `line_census`) belongs on a CPU
  node (`sbatch`, account m4943, `-C cpu -q debug`); `assign`, `select`, `summary`, `compress-index` run on a
  login node. Scripts that use a process pool need an `if __name__ == "__main__":` guard (spawn context).
- A PreToolUse hook blocks shell globs such as `du *.hdf5` on Perlmutter; measure sizes with Python.

## Data

- Input mock (production since 2026-10-29): `/global/cfs/cdirs/hacc/OpenCosmo/LastJourney/synthetic_galaxies/sdss_feniks_hizels_261013_10_14_2026/`,
  the OpenCosmo version with ray-traced lensing (67 slices, 1.9 TB, 263 columns). Each file has `cores/` (412.7 M
  galaxies, up to 13 M rows per slice) and `synth_cores/` (777 M galaxies in Monte Carlo host haloes of log Mhalo
  11.0–11.5, all centrals, `core_tag` -1, uniform random positions in each patch: unclustered), which OpenCosmo loads
  only with `synth_cores=True`; `DEFAULT_OPEN_KWARGS` keeps them out. `io.galaxy_group` finds `data` or `cores/data`.
  The line columns are lensed (`<line>` = `<line>_unlensed` x `magnification`, median 0.989, 16–84 % 0.96–1.04);
  `ra`/`dec` are unlensed (NFW-repositioned satellites), `ra_obs`/`dec_obs` lensed (deflection ~0.8'); also
  `kappa`, `shear1/2`, `Pa_alpha_1p8751um` (18756.4 Å, in the bandpass at z < 0.029 only; feature `PaA` = bit 9 of
  the preset `roman_transitions_paa`, which production uses; `DIFFSKY_ADDED_LINE_WAVELENGTHS` holds it so that
  `all_transitions` still matches the earlier mock). Paα/Hα median 0.149, Paα/[S III] 9532 0.42 at z < 0.05.
  Same cores as the mock of 2026-09 (same `core_tag`/`redshift_true`; a core appears in several box replicas, so match
  on core_tag + redshift), new diffsky fit (`sdss_feniks_hizels_261013`): per galaxy A_V median 0.83 -> 0.15, sSFR
  +0.6 dex, Hα +0.44 dex; Hα emitters above 1e-16 x1.48 at z 1.23, x0.93 at z 1.9; [O III] x2.2 at z 2.9.
  The synthetic cores would add 42 % (z 1.23) / 21 % (z 1.9) Hα emitters above 1e-16 and no [O III] at z 2.9.
  The raw per-patch version (`lc_cores-<step>.<patch>.diffsky_gals[.synthetic_halos].hdf5`, not OpenCosmo, no
  lensing, no `gal_id`, `ra`/`dec` = core positions) is in `/global/cfs/cdirs/m4943/Simulations/sdss_feniks_hizels_261013_10_14_2026/`.
- Previous mock (tests still use it): `/global/cfs/cdirs/m4943/Simulations/sdss_feniks_hizels_260710_09_07_2026/`
  (67 slices `lc_cores-<step>.diffsky_gals.hdf5`, 324 M galaxies, z 0–4, 340 GB, 174 columns; up to 10 M rows
  per file). Steps 219/224 (z ≈ 1.15–1.26) are handy for quick Hα tests; the smallest files are at z < 0.4,
  where only [S III] is inside the grism bandpass.
- Footprint: new mock 80 patches of a 1280-patch decomposition, a polar cap plus part of a ring, dec 58.7–90,
  2578.3 deg², `ra = phi`; previous mock 60 patches of a 1200-patch decomposition, dec 57.6–75.9, 2062.65 deg².
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

### Galacticus mocks (added 2026-10-15)

- `/global/cfs/cdirs/m4943/Galacticus/mockCatalogs/GP-MCMC-Calibration1_allUNIT_16sqDegMock_XMM-LSS/galaxyCatalogs/`:
  400 files `romanUNIT-d<a>_<b>_16sqDeg.hdf5` (random subsamples of the merger trees, each covering the whole 4° × 4°
  lightcone; 431 GB, 38.5 M galaxies, z 0.006–9.0, 7–135 k rows per file), read with plain h5py (`romanmock.galacticus`).
  Line datasets are contiguous and uncompressed (all 51 dust-attenuated line columns of a file read in 0.15 s); the
  cost is CFS latency per file, so `n_workers` (processes) helps little on a login node: 400 files take ~3 min for a
  few columns, 4 min for the flux distribution of 16 features with 6 workers.
- `/Lightcone/Output1/nodeData` (dust-free, `wavelength` attribute = air wavelength, `luminosity...` in erg/s) and
  `/Lightcone/Output1/dustAttenuatedNodeData` (same names, no attributes; the dust metadata are attributes of the
  group). A line is the sum of `luminosityEmissionLine{Disk,Spheroid,AGN}:<line>`; AGN and spheroid are usually
  negligible (median 1e-5 and 2e-4 of Hα) but dominate in 0.7 % of the galaxies. 70 % of the galaxies have no line
  attenuation at all (the dust model depends on stellar mass: median ratio 1.0 below 1e9 Msun, 0.47 at 1e9–1e10).
  That is a high-z average: the model (`gb10_generalised`, redshift and mass terms) attenuates almost everything at low z
  (Hα unattenuated in 1 % at z < 0.6, 34 % at z 1–1.5, 80 % at z 2–3; Paβ transmission 0.55 at z < 0.01, 0.75 at z = 0.5;
  attenuated Paβ/Hα 0.13 at z < 0.05 against the dust-free 0.054).
- `nodeSubsamplingWeight` is 1 everywhere (no weights); `angularWeight` is a constant 15.99 (meaning unclear, unused).
  The sky area is `angularSize**2` = 16 deg² (`Parameters/geometryLightcone`; RA 33.4–38.0, Dec −7.1 to −2.4, roll
  −10°); the Readme's `rightAscension`/`declination` are in `nodeData`.
- Cosmology `Parameters/cosmologyParameters`: H0 67.74, Ω_m 0.3089, Ω_b 0.0462, T_CMB 2.72548. The positions follow
  `FlatLambdaCDM(H0=67.74, Om0=0.3089)` *without* radiation (2e-5), as the Readme says; with `Tcmb0` the distances are
  off by 2e-4. `cosmology_from_header` therefore leaves `Tcmb0` at 0.
- `GALACTICUS_TRANSITIONS` (`galacticus_transitions`) is a `LineSet` whose components are Galacticus line names; its
  first seven features carry the `roman_transitions` names so `plotting.FEATURE_COLORS` / `FEATURE_LABELS` apply.
  `KNOWN_LINE_WAVELENGTHS` (diffsky + Galacticus names) is what `LineSet` uses for component wavelengths.
- The Galacticus mock is much deeper than diffsky (median M* 5e7 Msun, median Hα flux at z 1–1.5 of 2.5e-18 against
  3–5e-17): fractions above a threshold are not comparable between the mocks, counts per deg² are.
- Products go to `/global/cfs/cdirs/m4943/Simulations/RomanMocks/GP-MCMC-Calibration1_allUNIT_16sqDegMock_XMM-LSS/`
  (`census/flux_distribution.npz`, z 0–4 in 0.01, from `romanmock galacticus-flux-distribution`). Notebook 07.
- `GalacticusMock.line_fractions(flux_limit, secondary_flux_limit, ...)` (2026-10-16) builds the `LineFractionTable`
  of `BaseCatalog.line_fractions` from the 400 files (all 16 `galacticus_transitions` features count as other lines,
  [N II] included; S/N route only with a fixed `size`); ~8 min with 4 workers on a login node. 16 deg² give only
  ~1e4–1e5 single-line galaxies, so the diffsky binning (250 Å, 500 per flux bin) leaves one flux bin per wavelength
  bin: use coarser wavelength bins / fewer galaxies per bin. Notebook `Projects/Roman/RomanMock/notebooks/10_galacticus_line_fractions.ipynb`.
- `GalacticusMock.single_lines(flux_limit, secondary_flux_limit, ..., columns=[...])` (2026-10-16) returns the scan behind
  `line_fractions` as arrays: `wave`, `flux`, `line` (index in the line set) of every single-line galaxy plus any `columns` of
  those galaxies (e.g. the gas metal masses, to cut afterwards). Notebook `Projects/Roman/RomanMock/notebooks/11_line_flux_vs_wavelength_single_lines.ipynb`.
- Paβ at low z (2026-10-16, notebook `Projects/Roman/RomanMock/notebooks/12_paschen_beta_low_redshift.ipynb`, cache
  `census/pab_z0p51.npz`, 617 k galaxies at z < 0.51): the steep rise of the Paβ flux quantiles at the blue end (median
  above 1.5e-16 below 13 100 Å) is 1/D_L² at fixed luminosity (median Paβ L 2e38 erg/s at every z < 0.1). The bins of
  0.01 at z < 0.05 hold 3, 27, 160, 121, 80 galaxies (17 000 Mpc³, half the density of 0.05–0.15: local structure,
  group at z 0.021, wall at 0.055–0.06). The lowest galaxy is at z = 0.0062.
- Artefact (found 2026-10-16): 20 % of the galaxies have a negative gas metal mass (`spheroidAbundancesGasMetals` 18 %,
  `diskAbundancesGasMetals` 3 %, down to -1e14 Msun); the line model then uses its lowest metallicity ([N II]/Hα = 2.7e-4,
  [S III]/Hα ≈ 0.01) while Hα/Paβ stay bright. They are 21 % of the Hα emitters above 1.5e-16 and 59 % of the single-line
  galaxies at (1.5e-16, 6e-17). `line_fractions(cuts={"diskAbundancesGasMetals": (0, None), "spheroidAbundancesGasMetals": (0, None)})`
  removes them (and 6.7 % of the other emitters).

### Line rescaling (added 2026-11-09)

- `rescaling.py` (`LineFactors`, `GalacticusLike`, `build_rescaling`, `galaxy_normal`, `calzetti_k`): log10 F' = log10 F +
  offset + sigma * eps per feature, eps ~ N(0, 1) a splitmix64 hash of `(gal_id, seed)` (default seed 20261109), one
  deviate per galaxy shared by all its scattered lines; zero/negative/NaN fluxes untouched. Wired into `make_catalog`,
  `line_fractions` and `make_proposal` (2026-11-09, `rescaling=` argument, default None = stored fluxes; `make_proposal`
  defaults to `"galacticus_like"`): `BaseCatalog._flux_models` validates it (gal_id stored for a scatter, Ha/PaA in the
  line set for `galaxy_dust`: the earlier mock's `roman_transitions` has no PaA and needs `galaxy_dust=()`; line sets with
  separated multi-component features refused) and warns when the thresholds are less than 10^max_log_boost times the
  base limit, times 1 + 1/ratio when a doublet rule merges (`_check_rescaling_depth(model, rule, sens)`, called for the
  doublet rule alone too since 2026-11-09); the workers apply it in `_chunk_fluxes` to the read dict before stacking, so the
  catalog's flux columns are the rescaled ones; its config is in `cat.settings["rescaling"]` and the table settings, and
  `_check_table` compares it. NOT in the YAML pipeline route (assign/select).
- `GalacticusLike` is the recipe of `Projects/Roman/Fullshape/Catalog/diffsky_line_rescaling.ipynb`: Hα fixed (diffsky's
  Hα is fitted to HiZELS, `param_utils/sdss_feniks_hizels_calibrations/README.txt` of diffsky); the user's rule
  (2026-11-09): rescale every diffsky feature that Galacticus also has (names shared with `galacticus_transitions`), leave
  [O III] λ4363 and [S III] λ6312 alone. Dust term at each feature's own wavelength (so [S III] 9532/9069 changes as
  Galacticus's dust does), delta and sigma shared within a doublet. Blue lines: constant dust term (da_ha 0.5 mag; their
  delta were derived with it). `galaxy_dust` lines (default [S III] λλ9069, 9532, Paα; added 2026-11-09 after a constant
  term put the floor of [S III]/Hα at 0.40 and emptied the single-line galaxies below z = 1): diffsky's own reddening
  A_D (r_X - 1) (A_D from the galaxy's Paα/Hα, `galaxy_a_ha`: intrinsic -0.879, r_PaA 0.115; `DIFFSKY_DUST_RATIO`) is
  swapped for the Calzetti reddening of the Galacticus A(Hα) at the same quantile (`DUST_MAP_*`, 41 knots from the
  figure-1 galaxies of both mocks). Needs Ha and PaA in the line set / the `apply_columns` dict (`required_features`).
  Factors (`typical_offsets`, per-galaxy lines at the median dust 0.35 mag): Hβ 0.92, [O III] 1.32 (both), [O II] 1.65,
  [S III] 9532 1.19 / 9069 1.17, Paα 1.31; sigma [O III] 0.19, [O II] 0.11. `galaxy_dust=()` = the first, constant
  version (Paα 1.29, [S III] 1.20 / 1.17). Rejected forms: A_D x 2.62 (dust ratios twice too broad), the map for every
  line (halves the 1 < z < 1.94 single-line candidates; delta would need re-deriving). `max_log_boost(n_sigma=5)` = 1.07
  dex (11.8x; 15.8x with the merged [O III] doublet): a 1e-17 base is complete for primary thresholds >= 1.6e-16 only,
  and the S/N route (3.06e-17 at best, point source, S/N 5, 1 h, 2 x zodi) warned "3.0 times deeper" on it; hence the
  production base at 1e-18 since 2026-11-09 (ratio 30). Tests: `tests/test_rescaling.py` (9; checks the
  uint64 hash against plain-Python splitmix64 and that dust-free galaxies keep their dust-free ratios);
  `test_catalog.py::test_rescaling_and_doublet_rule` (the catalog's fluxes = the stored ones rescaled, galaxy by galaxy).

### [O III] doublet rule (added 2026-11-09)

- `doublets.py` (`DoubletRule`, `build_doublets`, `doublet_profile`, `doublet_dip`, `arcsec_per_kpc_interpolated`):
  whether [O III] λ4959 is a feature of its own. Line image = face-on exponential disk of half-light radius r50
  projected on the dispersion (`|x| K1(|x|/h)`, h = r50/1.678, 10.85 Å per 0.11" pixel) ⊗ Gaussian of 2 px; resolved
  when the profile dips between the lines by >= `min_dip` (0.2) of the λ4959 peak (a dip, not a FWHM: the noiseless
  cusp always leaves a tiny maximum, ~1e-3, for blended galaxies). `resolved_radius(z)`: 0.354/0.551/0.738" at z
  1.0/1.94/2.85 from a table on z 0-6 (0.05) computed once per process (~2 s, `lru_cache`). Modes `per_galaxy` (size
  from a `SizeSpec`, default `r50_disk_2d` in kpc at `redshift_true`; kpc -> arcsec interpolated, 2e-6, 6x faster than
  astropy; invalid sizes count as blended), `merged`, `separated`. `apply` acts on the stacked feature fluxes (in
  `_chunk_fluxes`, after the rescaling) only where both lines are strictly inside the bandpass (as the detection tests
  it): weak flux added to the strong column, weak set to 0, strong keeps its rest wavelength; returns the merged mask
  (`doublet_merged`, a derived column, all False without a rule). `make_catalog(doublets=None)` default = separated;
  `make_proposal` defaults to `"per_galaxy"`. Validated in `_flux_models` (both features in the line set, size columns
  in `base.description["columns"]`; the cosmology goes to the workers as `doublet_cosmology`). Config in
  `cat.settings["doublets"]` and the table settings; `_check_table` compares it. Physics and caveats:
  `Projects/Roman/Diffsky_line_rescaling_and_OIII_doublet.md`. Tests: `tests/test_doublets.py`.

### Proposal -> catalog (added 2026-11-09, the user's two-step routine for catalogs with interlopers)

- `BaseCatalog.make_proposal(z_range, line="Ha", <thresholds>, rescaling="galacticus_like", doublets="per_galaxy",
  dz=0.01, min_count=100, min_share=1e-3)` -> `proposal.Proposal`. Thresholds must be given (flux limits, S/N or a
  sensitivity). One pass (worker `_proposal_slice`, stream `(seed, 50000, step)`) over the slices whose z (with
  `RSD_MARGIN`) meets the range or a window interval; `proposal_counts` (array level) histograms in true z (grid
  `proposal_grid`: the range's ends are edges, bins continued to the lightcone's end): `n_det`, `n_line_det[z, i]`,
  `n_single[z, i]`, `n_window_det` / `n_window_single[z, i, k]` (line i inside the window of target k = observed
  wavelengths of the target for the range, inclusive), `n_assigned[z, i, k]` (those singles binned at the redshift they
  would get as target k; a target's own column lands on itself and is dropped wherever interlopers are summed),
  `n_doublet_both/merged`. Settings keep `targets` and `windows` (one per target).
  "Inside" = bins of the range; target sample `n_sample` = target detected inside. Minor lines: count < min_count or
  share < min_share (of the detected galaxies inside / of n_sample), left out of `plot()` and `confusion_template()`
  (`{interloper: {target: p}}` + `{target: {interloper: p}}`).
- `line="all"` (`proposal.ALL_LINES`, added 2026-11-09 with the hint, at the user's request): targets = every line in the
  bandpass for the range (`target_windows`); `n_sample` = every detected galaxy inside; interlopers are pairs
  (line, taken for) restricted to `Proposal.assignable` = targets that are `major_inside` (a line never single, e.g.
  [S III] 6312, is not taken for, although by wavelength it could be); `interlopers()` -> `{taken for: {line: ...}}`,
  `major_pairs`; summed "at most" can count a galaxy twice (wide ranges: an upper bound); template entries that are both
  kinds get `p_interloper`; loss entries toward non-assignable lines are dropped (a matrix is wavelength-free: Ha->OII_3727 meant
  to lose Ha at 1.6-1.94 also made 2,597 'Ha_as_OII_3727' interlopers at 1.68-2.0 in notebook 02 before the fix). `make_catalog`
  then uses `lines=None`; default name `all_z<lo>_<hi>`.
- `make_proposal(verbose=True)` prints `proposal.line_hint(line, names, rest)` first: what `line="Ha"` /
  `"OIII_5007"` / `"all"` mean and the line names.
- Step 2 `Proposal.make_catalog(confusion, out_dir, name=None, columns=CATALOG_COLUMNS, n_randoms=0)`: calls
  `base.make_catalog(sensitivity=<config>, detection, rescaling, doublets, confusion, lines=[line] (None for "all"),
  z_range, z_column="redshift_assigned")` and writes `<name>.hdf5`, `<name>_summary.json`, `<name>_proposal.npz`
  (`Proposal.load`), `<name>_random.hdf5` (n_randoms). `confusion` or `fractions` is required (`{}` = clean, `"empirical"`
  builds the table with the same choices). Tests: `tests/test_proposal.py` (array level, incl. `line="all"` and the
  hint), `test_catalog.py::test_proposal_then_catalog` (the target sample = the clean catalog; with every potential
  interloper taken for the target, the catalog's interlopers = the proposal's `n_assigned`) and
  `::test_proposal_of_every_line_then_catalog` (same for `line="all"`). Notebook:
  `Projects/Roman/Fullshape/Catalog/contaminated_catalog_routine.ipynb`.
- Catalogs by design (2026-11-10, the user asked for "a designer approach": `make_proposal` takes the range, line,
  primary threshold and an expected total `interloper_fraction` and shows the possible interloper fraction against the
  secondary limit for each contribution and in total; `make_catalog` takes `{taken for: {line: fraction}}`, e.g.
  `{"Ha": {"OIII_5007": 0.005, "SIII_9532": 0.005}}`, "the fraction of OIII_5007 and SIII as interlopers to Halpha
  selected galaxy sample").
  - Scan: every sensitivity model puts the secondary limit at a fixed fraction q of the primary one at each wavelength
    (F2/F1, snr2/snr1, `secondary_ratio`), so `det.snr / proposal.primary_snr(model)` (`snr_threshold` of noisy detection,
    else `snr_at_limit`) decides a galaxy's lines at every q: single at q iff its second line <= q. `proposal_counts(
    primary_snr=, q_edges=, z_range=)` adds `scan_window[q bin, line, target, inside]` (detected galaxies whose brightest
    line falls in the target window, binned in their second line; inside = true z in the range's bins) and
    `scan_target[q bin, target]` (detected galaxies inside, binned in the target's own signal: the line-mode sample at q
    is those above q). Grid `proposal.SCAN` (q 0.01–1 in 0.005 dex; bin 0 <= first edge, last above 1;
    `searchsorted(side="left")`): exact on the edges, linear in log q between them (`Proposal._interp`; 33 of 280,912
    off at q = 0.8 in the test base). Settings `scan` and `interloper_fraction`; proposals saved before have no scan
    (`has_scan`, the methods raise). `Proposal.thresholds` ("flux" / "snr" / "ratio"), `primary`, `secondary` (units of
    the thresholds), `SECONDARY_KEYS`, `_secondary_q(secondary_flux_limit=, secondary_snr=)`.
  - Fractions are shares of the catalog: for `line="Ha"` the Hα sample with its interlopers, for `line="all"` every
    galaxy of the catalog (the line report's share² stays per taken-for line). "At most" (`interlopers()`,
    `contamination_at_most`, the scan) is the composition if every potential interloper were taken,
    `n / (n_sample + sum n_out)`: changed from `n / n_sample` on 2026-11-10, so that the pairs add up to the total and
    the summary's two tables agree. `n_out = n` for a target line; for `line="all"` only the single lines whose true z is
    outside the range (those inside are in the sample already and only move).
  - `Proposal.design(fractions, secondary_*)`: `N_cat = N / (1 - sum f n_out / n)`, `p = f N_cat / n`; refuses `p > 1`
    naming the lowest secondary limit that meets the table (`_feasible` on the edges, true from there up to the primary
    limit). `fraction_template(total)` splits in proportion to n (one common p, major pairs); `secondary_for(total)`;
    `interloper_scan(secondary)`; `plot_secondary()`; the summary's section "Interlopers against the secondary limit"
    (rows = pairs, columns = round secondary limits, the proposal's own marked ⁴).
  - `make_catalog(confusion=None, out_dir=None, ..., *, fractions=, secondary_flux_limit=, secondary_snr=)`: exactly one
    of confusion / fractions; any secondary limit of the scan (also with a confusion table). The fractions become
    `confusion={"type": "windowed", "pairs": [...], "meta": {...}}` (`confusion.WindowedConfusion`: p only inside the
    window of the pair, elsewhere the line is kept, so nothing of the sample is lost; one uniform per single-line galaxy
    as `ConfusionMatrix`; overlapping windows of one line must not sum above 1). Prints the design before the scan and
    "asked and made" after; the summary JSON gets `design` (with `made`).
  - Tests: `test_proposal.py` (`_scanned`: `test_secondary_scan`, `test_design_by_interloper_fractions`,
    `test_windowed_confusion`), `test_catalog.py::test_catalog_by_design` (the non-interlopers are exactly the clean
    catalog; interlopers within 5 sigma of the expectation; another secondary limit) and `::test_every_line_by_design`
    (`len(cat) - n_sample` = interlopers from outside the range).
  - Scan grid (2026-11-10): 0.001 dex (2000 bins) plus the proposal's own q as an extra edge (`scan["extra"]`), so the
    scan equals the tables against redshift at the proposal's secondary limit. The summary table and `plot_secondary`
    start at the last secondary limit where no interloper is possible (`_shown_low`): at F1 1.5e-16 no [S III] λ9532
    galaxy is single below 5.8e-17, because the mock's fixed λ9532/λ9069 ratio (2.57 after rescaling) keeps λ9069 above
    the secondary limit; then at most 9 round values up to the primary limit. No "none" column (no secondary limit) any
    more since 2026-11-10: catalogs with one line per galaxy have no interlopers, so the most a catalog can hold is at
    the primary limit (a single tier); `_numbers`/`_interp` take finite q only and `scan_preferred` is gone.
- Single-line clean mode (2026-11-10, the user: "make_proposal((1.0, 1.2), line='Ha', flux_limit=1.5e-16) and ignore
  the secondary flux_limit ... print the single line fraction of Ha as true single line galaxies as a function of z, for
  3 secondary flux limits, 0.3, 0.4, 0.5 times primary ... if [make_catalog's] secondary flux limit is none, then all Ha
  galaxies above the primary flux limit are viewed as single-line galaxies; if input, only those without secondary
  lines above it"; then, on the first notebook 05, which made interlopers in this mode: "not what I want ... no
  interlopers in this mode"). `make_proposal` with no secondary limit (not given or None, no sensitivity model) AND no
  `interloper_fraction` is single-line: single tier plus `detection["single_line"] = line` (True for `line="all"`), one
  line per galaxy, applied by `detect_lines` (`detection.one_line_per_galaxy(primary, snr, prefer)`; the models only
  carry the attribute): the target line wherever it is above the primary limit, else the brightest. Target first, not
  brightest only: [O III] outshines Hα in 42.5 % of the Hα-above-primary galaxies at 1.0–1.2 (rescaling + merged
  doublet), and brightest-only kept 7.25 M of 12.6 M. No interlopers: `make_catalog` takes no `fractions` and only
  `confusion=None`/`{}` (anything else raises), and `_require_interlopers()` makes `interloper_scan`, `secondary_for`,
  `design`, `fraction_template`, `plot_secondary`, `confusion_template` (and `_numbers`) raise. With a secondary limit,
  `_detection_at(q)` gives two tiers plus `detection["primary_line"] = line` (`ThresholdDetection` /
  `GaussianNoiseDetection` attribute, applied in `detect_lines`: only galaxies with that line above the primary limit
  are detected), so the catalog holds the same galaxies as without one, single-line (`id_outcome` 2) those without
  another line above it, the rest multi-line (1); without `primary_line` a two-tier Hα catalog would also hold galaxies
  whose Hα is only a secondary detection (14.96 M vs 12.62 M at 6e-17). The single-line count equals
  `nq_single[j][inside, line].sum()` exactly (321,887 at 7.5e-17 = 0.5 × primary). `secondary_ratios` (default 0.3,
  0.4, 0.5, setting) -> `n_primary[z, i]`, `nq_single[j, z, i]`, `nq_window_single[j, z, i, k]`;
  `Proposal.single_line_fraction(ratio)`, `fraction_interloper(..., ratio=)`. The summary of a single-line proposal is
  the fields (interlopers: "none: a single-line clean catalog") plus "Hα single-line galaxies against redshift" and
  `next:` hints; `plot()` is one panel (the three ratios). `make_proposal(..., interloper_fraction=...)` without a
  secondary limit is a designer proposal with a single tier of its own (secondary = primary, the ⁴ column at the right
  end of the scan; notebook 04), not single-line. Tests: `test_proposal.py::test_single_line_mode`,
  `test_catalog.py::test_single_line_catalog` (same gal_id with and without a secondary limit, single count = nq_single,
  refusals, the designer proposal without a secondary limit is single-tier).
- Tests: `pytest.approx` keeps an absolute tolerance of 1e-12 unless `abs=` is given, so comparisons of fluxes
  (1e-17) pass whatever the value: use `abs=0` (found 2026-11-10 when a wrong `secondary_for` passed).

## Layout

| Module | Role |
|---|---|
| `roman.py` | defaults: grism range 10000–19300 Å, primary limit 1.0e-16, secondary limit 4.0e-17, S/N at limit 6.5, redshift bins `DEFAULT_REDSHIFT_BINS` (Spezzati, Wang & Hearin 2026, Table I) |
| `lines.py` | `LineFeature`/`LineSet`, presets, bit ordering (bit i = feature i, uint32) |
| `cosmology.py`, `fluxes.py` | erg/s → erg/s/cm² via the OpenCosmo header cosmology (astropy or dsps backend) |
| `sensitivity.py` | `SensitivityModel` + `DetectionModel` protocols and registries; two-tier rule (`apply_two_tier`); `RomanGrismSensitivity` (`type: roman_grism`, S/N thresholds, WFI grism tables vs wavelength and size); `limits_contained`, `nominal_flux_limit` |
| `sizes.py` | `SizeSpec`: angular size of a galaxy, a number in arcsec or a column in proper kpc converted at `redshift_true` |
| `detection.py` | `detect_lines`, `pack_bitmask`, `popcount`, `DetectionResult` |
| `confusion.py` | `ConfusionMatrix` (fail column), `WavelengthBinnedConfusion`, `WindowedConfusion` (`type: windowed`: a line taken for another only inside a wavelength window; catalogs by design), `LineFractionTable` (`type: empirical`: the line mix of the single-line galaxies per observed wavelength and flux rank), `DEFAULT_CONFUSION_ROWS`, `build_confusion` |
| `redshift.py` | `assign_redshifts`, `IdOutcome` codes, `MultiLineModel` hook |
| `selection.py` | footprints, `SampleDefinition`, `sample_mask`, `TargetDNDZ`, thinning helpers, `sky_area_deg2` |
| `samples.py` | slice-wise stage B: `process_slice`, `run_pass`, `thinning_keep`, `sample_keep` |
| `pipeline.py` | `RomanMockPipeline`: `process_arrays`/`process_fluxes` (pure numpy) → `run` (stage A, from the mock or from the base catalog), `build_base`, `build_samples` |
| `footprint.py` | `PatchFootprint`: union of the theta-phi sky patches of the lightcone (`from_mock`, `sample`, `contains`, `area_deg2`) |
| `randoms.py` | `make_random(data, footprint, n_times, ...)`: the random catalog of one data catalog (any `Catalog`: whole, `subset`, mask), uniform in the footprint, redshifts drawn from the data, stream keyed on a hash of the data; `sample_sky`, `data_key` |
| `catalog.py` | `BaseCatalog.make_catalog` -> `Catalog`: step 2 in memory from the base catalog (the everyday route), `Catalog.write/read`; `BaseCatalog.line_fractions` scans it for the `LineFractionTable`; `BaseCatalog.make_proposal` -> `Proposal`; `_chunk_fluxes` (rescaling + doublet rule) shared by the three workers |
| `reference_dndz.py` | Wang et al. (2022) Tables 1–2 with uncertainties |
| `fluxdist.py` | `flux_distribution` / `FluxDistribution`: log-flux histograms of every galaxy per line and redshift bin, no cut; quantiles, fraction above a threshold, `rebin(step)` to merge redshift bins |
| `galacticus.py` | `GalacticusMock`: the Galacticus lightcone mocks of the GRS-PIT (`read`, `iter_files`, `read_fluxes`, `flux_distribution`, `single_lines`, `line_fractions`), plain h5py, no OpenCosmo; `resolve_column`, `read_file`, `air_to_vacuum` |
| `census.py` | `line_census` / `LineCensus`: dN/dz of emitters above flux limits, single/multi-line, no confusion |
| `rescaling.py` | `LineFactors` / `GalacticusLike` (`type: line_factors / galacticus_like`), `build_rescaling`: per-galaxy rescaling of the line fluxes (constant + shared log-normal deviate keyed on `gal_id`); `make_catalog(rescaling=)` |
| `doublets.py` | `DoubletRule` (`per_galaxy` / `merged` / `separated`), `build_doublets`: [O III] λ4959 a line of its own where resolved for the galaxy's size; `make_catalog(doublets=)` |
| `proposal.py` | `Proposal` (step 1 of a catalog with interlopers: single-line fractions inside the range and of the potential interlopers, the scan of the secondary limit, `design` / `fraction_template`; `make_catalog(fractions= or confusion=, out_dir)` = step 2), `proposal_counts` (array level) |
| `report.py` | layout of every printed report (2026-11-09, the user asked for prettier, more organised output): `title` (━━), `section` (──), `fields` (aligned key/value), `table` (rule under the header), `notes` (wrapped, footnote marks ¹ ² ³), `Progress` (`[romanmock] … done in N s`), `zfmt`, `keep` (non-breaking), WIDTH 100. Used by `Proposal.summary`, `line_hint`, `Catalog.summary`, `format_line_report`, `LineFractionTable.summary`; tables keep line-set names, prose uses display names |
| `diagnostics.py`, `plotting.py` | checks (confusion, contamination, completeness, dN/dz); notebook plot style |
| `config.py`, `cli.py`, `io.py` | YAML config dataclasses, CLI, OpenCosmo I/O, manifests, index compression, memory budget |

CLI commands: `info`, `lines`, `example-config`, `run`, `base`, `census`, `flux-distribution`, `galacticus-flux-distribution`,
`catalog`, `line-fractions`, `random`, `assign`, `summary`, `select`, `compress-index`.
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
- Empirical identification (`type: empirical`, added 2026-10-09 at the user's request, the confusion matrix kept as
  an option): `LineFractionTable` in `confusion.py`. The single-line galaxies of the mock are binned in the observed
  wavelength of their line (`wave_bin` 250 Å by default: the grism resolves 21–24 Å, R 475–900 per 2 pixels, so the
  bin is set by statistics, not resolution) and, per wavelength bin, in flux rank: as many equal-count bins as hold
  `galaxies_per_bin` (500, the user's choice on 2026-10-09; 20 fixed bins diluted the rare lines) galaxies, so the table is ragged
  (`flux_edges` padded with inf before the last edge, `counts` zero there, `valid_cells` / `n_flux_bins_per_wave`), or
  a fixed `n_flux_bins`, or `flux_quantiles`; `fraction[w, f, j]` = share of line j in the cell; lines with fewer than `min_count` (100)
  single-line galaxies in all are dropped and the shares renormalised. `assign` ignores the true line: `mode`
  `draw` (each line with the share of the cell, one uniform per galaxy like `ConfusionMatrix`) or `max`; uniform
  `p_fail`. Empty cells take the mix of their wavelength bin, empty wavelength bins the mix of the whole table;
  wavelengths/fluxes outside the edges go to the nearest bin. `expected_confusion()` gives the implied
  `ConfusionMatrix` (used as `P` in the line report, `probability_kind: expected`). The table is built by
  `BaseCatalog.line_fractions` (worker `_single_lines_slice`, stream `(seed, 40000, step)`, matters for noisy detection
  only) at the limits of the catalog it is for; `make_catalog(confusion="empirical" | {"type": "empirical", options})`
  builds it on the fly (one more pass), a `LineFractionTable` or `{"type": "empirical", "file": ...}` reuses one;
  `_check_table` warns when its limits differ from the catalog's. `to_config()` is compact (hash of the content,
  not the numbers): it goes into `cat.settings` and so into HDF5 attributes (64 KB limit) and into
  `scientific_config()`; `save`/`load` (`.npz`) and `to_dict` hold the numbers. The pipeline route needs `file:`
  (`romanmock line-fractions` writes it); it does not build the table itself.

Detection (two-tier since 0.3.0):

- Two ways of fixing the thresholds (the user's decision, 2026-10-15): flux limits (`type: constant`,
  `make_catalog(flux_limit, secondary_flux_limit)`, kept as is) or S/N (`type: roman_grism`,
  `make_catalog(snr=5, secondary_snr=2, size=...)`). In the S/N route a galaxy needs one line with S/N >= 5, its other
  lines are detected at S/N >= 2; one detected line = single-line galaxy, primary + secondary lines = multi-line.
  Noise: the 5-sigma line limits of the NASA WFI technical page (grism, v1.4; `roman.py`) for a point source and
  r50 = 0.3" at ten wavelengths, 1 h, 2 x min zodi; `F_5sigma(lambda, r) = F_pt(lambda) (1 + (r/r_c)^2)^p (t/1h)^-1/2 (b/2)^1/2`,
  `r_c(lambda)` fixed by the two rows (0.12-0.14", an empirical transition scale, not the PSF), `p = size_exponent`
  0.5 (2-D broadening: noise ∝ r, the large-size scaling the WFI page quotes; other exponents are sensitivity tests);
  S/N = 5 F / F_5sigma. `snr=6.5, size=0.3` gives 1.05e-16 at 1.5 um, the old baseline.
- Sizes (`sizes.SizeSpec`): a number (arcsec, 0 = point source) or a column (`r50_disk_2d`, proper kpc, converted with
  `arcsec_per_kpc_proper(redshift_true)`; `scale`, `floor`; a non-finite or negative size is an error unless
  `fallback` [arcsec] is set, `n_invalid` counts them, `make_catalog` reports `info["n_fallback_size"]`, the pipeline
  warns). `detect_lines(size=)` broadcasts an `(n,)`
  array against `(n, n_features)`; `primary_flux_limit` / `secondary_flux_limit` forward sizes only to a model that
  takes them from a column (`needs_size`), so a fixed-size model keeps its size (matters for `limits_contained`).
  `RomanMockPipeline.base_columns` = `base.columns` + the size columns: a base built with a size column stores it;
  `make_catalog(size="r50_disk_2d")` needs `r50_disk_2d` in `base.columns` (production.yaml lists it since
  2026-10-15; the production base carries it since the rebuild of 2026-10-15). Workers get the `SizeSpec` and the
  cosmology (`BaseCatalog.cosmology`, from the first base file) in their spec.
- Containment of a selection in the base catalog is `sensitivity.limits_contained(requested, base)`: primary limits
  on a wavelength grid and, when a model takes sizes from the catalog, on a grid of raw sizes (0-30") of the same
  galaxy, each model's `SizeSpec.transform` (scale, floor) applied, so that two specs are compared galaxy by galaxy;
  specs reading different quantities (`SizeSpec.same_source`) and a requested `size_exponent` below the base one
  (limits cross beyond any grid) are refused (review of 2026-10-16). A noisy detection model warns in both entry
  points when the base is less than `NOISY_BASE_MARGIN` (1.5) times deeper. `cat.settings` carries
  `sensitivity` (the config) plus nominal `flux_limit` / `secondary_flux_limit` (1.5 um, size 0.3" for a column);
  the Wang et al. lookup then needs an explicit `flux_limit` in `downsample`. `_check_table` compares the
  sensitivity configs. Note of the implementation: `Projects/Roman/RomanMock/RomanMock_SNR_Sensitivity_Note.md`.

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
- Randoms (refactored 2026-11-10 at the user's request, "too complicated ... input the data catalog only"):
  `cat.make_random(n_times, seed, z_column, write_to, footprint)` / `randoms.make_random(data, footprint, ...)` take one
  data catalog, any `Catalog` (whole, `cat.subset(which)`, `cat[mask]`), and nothing else: `n_times x len(data)` points
  uniform inside the footprint (and the data's `cuts` on ra/dec), redshifts drawn with replacement from the data. Blocks
  of `len(data)` points, block k from the stream `(seed, 30000, data_key, k)`, `data_key` = 63-bit blake2b of the data's
  finite redshifts in order: the same data give the same randoms, a larger `n_times` keeps the first blocks, two parts of
  a catalog get independent randoms with one seed (the old per-kind stream indices, `which=`, "components", `first=` and
  the per-index files are gone). `write_to` .hdf5 is written block by block into one file (h5py, attributes from
  `catalog.catalog_attributes`, which `Catalog.write` uses too); .npz/.fits in memory; a directory gets `random.hdf5`.
  `settings["random"] = {n_times, n_data, seed, data_key}`. Footprint = the 60 patches in `lc_patch`, rectangles from
  `lc_cores-decomposition.txt` in the mock directory; `dec = 90 - theta`, `ra = phi - 180` (offset measured in
  `PatchFootprint.from_mock`, cached in `<base>/<tag>_footprint.json`). Exact area 2062.65 deg², which is `base.area_deg2` and
  `cat.area_deg2` (the HEALPix estimate, 2063.0, is the fallback and what `select` uses). 0.006 % of the galaxies lie outside (satellites).
  The parts of a catalog are a catalog feature: `catalog.COMPONENTS` (all, multi_line, single_correct, galaxies,
  interlopers), `PAIR_SEP` `"<true>_as_<assigned>"` (`parse_pair`, `component_mask`; needs `line_true`,
  `line_assigned`), `Catalog.subset(which)`, `Catalog.components()` (`{which: size}`, pairs most numerous first).
  `Proposal.make_catalog(n_randoms=)` writes `<name>_random.hdf5` of the whole catalog; CLI `romanmock random CAT -n N
  [--subset WHICH]`. Tests `test_randoms.py::test_random_catalog_of_a_data_catalog`, `::test_parts_of_a_catalog`.
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

- `sdss_feniks_hizels_261013_10_14_2026/` (current, `configs/production.yaml`): `base/` rebuilt 2026-11-09 at limit
  1.0e-18 (job 61070403, `romanmock base configs/production.yaml --workers 32` alone, 1 min 52 s on a CPU node): lensed
  fluxes, line set `roman_transitions_paa`, 403.1 M galaxies (97.7 % of the 412.7 M cores), 67 slices, 61 GB;
  `base.columns` add `ra_obs`, `dec_obs`, `magnification`. Verified a strict superset of the 1.0e-17 base of
  2026-10-29 (job 60526856; 344.5 M galaxies, 83.5 %, 52.5 GB): same galaxies above 1e-17, every column identical but
  `line_bitmask_base`. The old base was deleted at the user's request after that check (2026-11-09). Why 1e-18: see the
  rescaling section (the S/N route with rescaling + doublet). Note `make_catalog()` without limits now selects at 1e-18.
  `census/` from the job of 2026-10-29 (it reads the mock, not the base); `census/dndz_census_z0p5_3p0_with_synth.npz`
  includes the synthetic cores (`Projects/Roman/RomanMock/scripts/census_with_synth.sh`, 8 workers: 24 ran out of memory).
- `sdss_feniks_hizels_260710_09_07_2026/` (previous mock):
  `base/` (limit 1.0e-17, 222.0 M galaxies, 25.9 GB; rebuilt 2026-09-30 by `scripts/production.sh`, job 59129636, 2 min
  on a CPU node; the 1.0e-16 base of 2026-09-28 had 64.6 M galaxies, 7.6 GB), `census/dndz_census_z0p5_3p0.npz`,
  `census/flux_distribution.npz` (job 59160266, 1 min), `catalogs/Every_z1p5_1p9.hdf5` (the former notebook 02),
  `randoms/Every_z1p5_1p9_<which>_<index>.hdf5` (the former notebook 03).
- `$PSCRATCH/romanmock_demo` holds the outputs of the earlier tutorial notebooks (scratch, purged in time).

## Working style

- Plots (notebooks) use `romanmock.plotting` (fixed categorical order, single axis, thin marks).
- `notebooks/` holds five examples (the user cleared the folder on 2026-11-09 and asked for 01-03; on 2026-11-10 for two
  more on the new methods, keeping 01/02 as they were): `01_catalog_z1p0_1p2.ipynb` (Hα 1.0-1.2: flux / S/N / "all"
  proposals, matrix and empirical catalogs, components, files), `02_catalog_z1p6_2p0.ipynb` (the same for every galaxy at
  1.6-2.0, `line="all"`), `03_randoms.ipynb` (randoms of the empirical catalog of 01 and of its parts with the
  data-catalog-only API, checks, reproducible / extensible / independent), `04_catalog_by_design.ipynb` (interloper
  fractions: the scan, `design`, a refused table, `fraction_template`, S/N scan, `line="all"` at 1.6-2.0) and
  `05_single_line_clean_catalogs.ipynb` (renamed from `05_single_line_catalogs` on 2026-11-10, the user: "no
  interlopers in this mode": the single-line table, the clean catalog with one line per galaxy and with a secondary
  limit of 7.5e-17, checks against the proposal, S/N). They write to `$PSCRATCH/RomanMocks/examples/`
  and run with `sbatch scripts/run_notebook.sh <notebooks...>` (`scripts/run_notebook.py`, nbclient, kernel
  `$NOTEBOOK_KERNEL` or `roman`; in order, a failure stops the run). Built by a generator script kept outside the repo,
  so edit them as notebooks. The seven earlier production notebooks (01 dN/dz, ..., 07 Galacticus flux distribution)
  are in git history only. The ten earlier tutorials were moved by the user to `~/Projects/Roman/RomanMock/Notebooks/`
  (2026-09-28). Keep notebooks minimal: only what is needed for the task.
- `configs/roman_grism_baseline.yaml` is the commented template for every config block; keep it complete (it was
  found truncated after the sensitivity block on 2026-09-28 and restored).
- The user commits; leave changes uncommitted in the working tree unless asked.

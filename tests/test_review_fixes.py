"""Tests for the items of RomanMock_Bug_Fixes_and_Methodology.md."""
import numpy as np
import pytest

from romanmock import RomanMockConfig, RomanMockPipeline, ConstantFluxLimit, ThresholdDetection, detect_lines
from romanmock.diagnostics import contamination_fraction, detection_efficiency, sample_completeness
from romanmock.fluxes import component_fluxes, feature_fluxes
from romanmock.lines import PLAN_SPECIES, ROMAN_TRANSITIONS, LineFeature, LineSet
from romanmock.selection import SampleDefinition, sample_mask, tracer_membership

from conftest import CATALOG_DIR, catalog_available


# ---------------------------------------------------------------------- B3
def _one_galaxy(z, f9069, f9532):
    """Arrays for one galaxy with only [S III] fluxes (erg/s/cm^2) at redshift z."""
    from astropy.cosmology import FlatLambdaCDM
    from romanmock.cosmology import flux_conversion_factor

    cosmo = FlatLambdaCDM(H0=67.66, Om0=0.31)
    fac = flux_conversion_factor(np.array([z]), cosmo)[0]
    data = {c: np.zeros(1) for c in ROMAN_TRANSITIONS.required_columns}
    data["S_III_9069"] = np.array([f9069 / fac])
    data["S_III_9532"] = np.array([f9532 / fac])
    zz = np.array([z])
    return data, zz, cosmo


@pytest.mark.parametrize("z, f9069, f9532, expect_species, expect_9069, expect_9532", [
    (0.075, 0.4e-16, 0.98e-16, False, False, False),   # review: summed flux must not detect; 9069 is out of band
    (1.05, 2.0e-16, 4.9e-16, True, True, False),        # review: 9069 in band must be detected although 9532 is out
])
def test_plan_species_detects_separated_transitions(z, f9069, f9532, expect_species, expect_9069, expect_9532):
    data, zz, cosmo = _one_galaxy(z, f9069, f9532)
    model = ThresholdDetection(ConstantFluxLimit(1e-16, (10000.0, 19300.0)))
    flux = feature_fluxes(data, PLAN_SPECIES, zz, cosmo)
    comp = component_fluxes(data, PLAN_SPECIES, zz, cosmo)
    det = detect_lines(flux, zz, PLAN_SPECIES, model, component_flux=comp)
    assert bool(det.detected[0, PLAN_SPECIES.bit("SIII")]) is expect_species
    # transition-level reference
    fluxt = feature_fluxes(data, ROMAN_TRANSITIONS, zz, cosmo)
    dett = detect_lines(fluxt, zz, ROMAN_TRANSITIONS, model)
    assert bool(dett.detected[0, ROMAN_TRANSITIONS.bit("SIII_9069")]) is expect_9069
    assert bool(dett.detected[0, ROMAN_TRANSITIONS.bit("SIII_9532")]) is expect_9532
    with pytest.raises(ValueError, match="component_flux"):
        detect_lines(flux, zz, PLAN_SPECIES, model)


def test_blend_flag_roundtrip_and_component_waves():
    assert PLAN_SPECIES["OII"].is_blend and not PLAN_SPECIES["OIII"].is_blend
    assert ROMAN_TRANSITIONS["OII_3727"].is_blend and not ROMAN_TRANSITIONS.has_separated_components
    ls = LineSet.from_config(PLAN_SPECIES.to_config())
    assert [f.blend for f in ls] == [f.blend for f in PLAN_SPECIES]
    assert PLAN_SPECIES.component_wave("O_III_4959") == 4960.370
    f = LineFeature.from_dict({"name": "S", "components": ["S_III_9069", "S_III_9532"], "rest_wave": 9533.4})
    assert not f.blend


def test_pipeline_runs_plan_species(synthetic, cosmo):
    cfg = RomanMockConfig.from_dict({"line_set": "plan_species"})
    out = RomanMockPipeline(cfg).process_arrays(synthetic, cosmo, np.random.default_rng(0))
    assert out["line_bitmask"].max() < 2 ** len(PLAN_SPECIES)


# ---------------------------------------------------------------------- M2
def test_tracer_membership_modes():
    data = {
        "line_assigned": np.array([2, 0, 0, -1], dtype=np.int8),    # brightest: OIII, Ha, Ha, none
        "line_bitmask": np.array([1 | 4, 1, 1, 0], dtype=np.uint32),  # galaxy 0 detects Ha and OIII
        "n_det": np.array([2, 1, 1, 0], dtype=np.uint8),
    }
    assert tracer_membership(data, ROMAN_TRANSITIONS, ["Ha"], "assigned").tolist() == [False, True, True, False]
    assert tracer_membership(data, ROMAN_TRANSITIONS, ["Ha"], "detected").tolist() == [True, True, True, False]
    assert tracer_membership(data, ROMAN_TRANSITIONS, ["OIII_5007"], "detected").tolist() == [True, False, False, False]
    data.update({"redshift_assigned": np.array([1.2, 1.2, 1.2, np.nan]), "id_outcome": np.array([1, 2, 3, 0], dtype=np.int8),
                 "ra": np.zeros(4), "dec": np.zeros(4)})
    m = sample_mask(data, ROMAN_TRANSITIONS, SampleDefinition(assigned_lines=["Ha"], membership="detected"))
    assert m.tolist() == [True, True, True, False]
    with pytest.raises(ValueError):
        tracer_membership(data, ROMAN_TRANSITIONS, ["Ha"], "bogus")


# ---------------------------------------------------------------------- M3
def test_m3_diagnostics_split():
    z = np.array([1.1, 1.1, 1.1, 1.1, 1.15, 1.15])
    data = {
        "redshift_rsd": z, "redshift_assigned": z,
        "line_bitmask": np.array([1, 1, 1, 0, 1, 1], dtype=np.uint32),
        "n_det": np.array([1, 1, 1, 0, 1, 1], dtype=np.uint8),
        "line_true": np.array([0, 0, 0, -1, 0, 0], dtype=np.int8),
        "line_assigned": np.array([0, 0, 2, -1, 0, -1], dtype=np.int8),
        "id_outcome": np.array([2, 2, 3, 0, 2, 4], dtype=np.int8),
    }
    edges = np.array([1.0, 1.2])
    d = detection_efficiency(data, ROMAN_TRANSITIONS, edges, "Ha")
    assert d["efficiency"][0] == pytest.approx(5 / 6)
    d = detection_efficiency(data, ROMAN_TRANSITIONS, edges, "Ha", reference_mask=z < 1.12)
    assert d["efficiency"][0] == pytest.approx(3 / 4)
    from romanmock.diagnostics import completeness
    assert completeness(data, ROMAN_TRANSITIONS, edges, "Ha")["completeness"][0] == pytest.approx(3 / 5)
    in_sample = np.array([True, False, False, False, True, False])
    assert sample_completeness(data, ROMAN_TRANSITIONS, edges, "Ha", in_sample)["completeness"][0] == pytest.approx(2 / 5)
    c = contamination_fraction(data, ROMAN_TRANSITIONS, edges, "Ha")
    assert c["n_sample"][0] == 3 and c["fraction"][0][0] == pytest.approx(1.0)


# ---------------------------------------------------------------------- B1 / B2 (real slice)
@pytest.mark.skipif(not catalog_available(), reason="diffsky mock not available")
def test_provenance_manifest_and_stale_samples(tmp_path):
    import opencosmo as oc
    import yaml

    from romanmock.io import list_catalog_files, read_manifest, read_stamp, step_from_filename

    files = sorted(list_catalog_files(CATALOG_DIR), key=lambda p: p.stat().st_size)[:2]
    steps = [step_from_filename(f) for f in files]
    base = {
        "catalog": {"path": CATALOG_DIR, "steps": steps, "rest_waves_from_catalog": False},
        "output": {"dir": str(tmp_path), "keep_undetected": True,
                   "passthrough_columns": ["gal_id", "ra", "dec", "redshift_true"]},
        "sensitivity": {"type": "constant", "flux_limit": 1e-17, "wave_range": [3000.0, 25000.0]},
        "confusion": {"type": "identity", "p_fail": 0.1},
    }
    pipe = RomanMockPipeline(RomanMockConfig.from_dict(base), aux_data=None)
    out1 = pipe.run(files=files[:1], verbose=False)
    man = read_manifest(tmp_path, "romanmock")
    assert man["config_hash"] == pipe.config_hash() and list(man["slices"]) == [steps[0]]
    assert read_stamp(out1[0])["romanmock_config_hash"] == pipe.config_hash()
    assert oc.open(out1[0]).columns  # stamping did not break OpenCosmo

    # partial parent is refused unless allowed
    with pytest.raises(RuntimeError, match="incomplete"):
        pipe.parent_files()
    assert pipe.parent_files(allow_partial=True) == out1

    # compatible rerun resumes: the first slice is reused, the second is processed
    out2 = pipe.run(verbose=False)
    assert len(out2) == 2 and sorted(read_manifest(tmp_path, "romanmock")["slices"]) == sorted(steps)
    meta = yaml.safe_load((tmp_path / "romanmock_metadata.yaml").read_text())
    assert meta["config_hash"] == pipe.config_hash() and len(meta["input_files"]) == 2

    # changed configuration cannot relabel existing outputs
    changed = RomanMockPipeline(RomanMockConfig.from_dict({**base, "confusion": {"type": "identity", "p_fail": 0.5}}), aux_data=None)
    with pytest.raises(RuntimeError, match="different configuration|configuration hash"):
        changed.run(verbose=False)
    with pytest.raises(RuntimeError):
        changed.parent_files()
    assert read_manifest(tmp_path, "romanmock")["config_hash"] == pipe.config_hash()  # untouched

    # B2: an empty rerun of a formerly non-empty sample removes the stale file
    from romanmock.config import SampleConfig

    pipe.config.output.overwrite = True
    s_full = SampleConfig(name="s", outcomes=["MULTI_LINE", "SINGLE_CORRECT", "SINGLE_MISID"])
    # samples are written slice by slice: a directory per sample (one file per slice) by default
    from romanmock.io import open_sample

    assert pipe.build_samples(samples=[s_full], verbose=False) == [tmp_path / "samples" / "s"]
    n_slices = len(open_sample(tmp_path / "samples" / "s"))
    s_empty = SampleConfig(name="s", z_min=9.0, z_max=10.0)
    with pytest.warns(UserWarning, match="empty"):
        assert pipe.build_samples(samples=[s_empty], verbose=False) == []
    assert not (tmp_path / "samples" / "s").exists()
    assert yaml.safe_load((tmp_path / "samples" / "s_metadata.yaml").read_text())["empty"] is True
    # the single-file layout is still available and holds the same rows
    pipe.config.output.sample_layout = "single"
    assert pipe.build_samples(samples=[s_full], verbose=False) == [tmp_path / "samples" / "s.hdf5"]
    assert len(oc.open(tmp_path / "samples" / "s.hdf5")) == n_slices
    with pytest.warns(UserWarning, match="empty"):
        assert pipe.build_samples(samples=[s_empty], verbose=False) == []
    assert not (tmp_path / "samples" / "s.hdf5").exists() and not (tmp_path / "samples" / ".s_slices").exists()
    pipe.config.output.sample_layout = "slices"

    # overwrite with a changed configuration discards the old outputs and starts over
    changed.config.output.overwrite = True
    out3 = changed.run(verbose=False)
    assert len(out3) == 2 and read_manifest(tmp_path, "romanmock")["config_hash"] == changed.config_hash()

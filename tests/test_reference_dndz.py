import numpy as np
import pytest

from romanmock.config import RomanMockConfig
from romanmock.lines import ROMAN_TRANSITIONS
from romanmock.pipeline import RomanMockPipeline
from romanmock.reference_dndz import WANG2022_FLUX_LIMITS, WANG2022_TABLES, wang2022_bins, wang2022_target
from romanmock.selection import TargetDNDZ, downsample_to_dndz, population_thinning_mask


def test_transcription_matches_summary_rows():
    """The paper's summary rows (mean dN/dz over 1<z<2 etc.) check the transcribed bins."""
    for key, tab in WANG2022_TABLES.items():
        edges = tab["edges"]
        for av in (1.65, 1.92):
            for (lo, hi), row in tab["summary"][av].items():
                sel = (edges[:-1] >= lo - 1e-9) & (edges[1:] <= hi + 1e-9)
                mean = tab[av][sel].mean(axis=0)
                np.testing.assert_allclose(mean, row, rtol=6e-3, atol=1.0, err_msg=f"{key} AV={av} {lo}-{hi}")  # integer rounding
        # densities decrease with the flux limit in every bin
        for av in (1.65, 1.92):
            assert np.all(np.diff(tab[av], axis=1) < 0)


def test_wang2022_target_step_function():
    edges, dndz = wang2022_bins("Ha", 1e-16, 1.65)
    t = wang2022_target("Ha", 1e-16)
    assert t(np.array([1.15]))[0] == dndz[6]            # bin 1.1-1.2
    assert t(np.array([2.0]))[0] == dndz[-1]            # closed upper edge
    assert t(np.array([0.3, 2.5])).tolist() == [0.0, 0.0]
    assert wang2022_target("OIII_5007", 1e-16, 1.92)(np.array([2.05]))[0] == 555
    with pytest.raises(ValueError):
        wang2022_target("Ha", 1.2e-16)
    with pytest.raises(KeyError):
        wang2022_target("Hb")


def test_downsample_closed_upper_edge_and_empty_bin_warning():
    rng = np.random.default_rng(0)
    keep, p = downsample_to_dndz(np.array([1.0, 1.5, 2.0]), TargetDNDZ([0, 3], [1e9, 1e9]), 1.0,
                                 np.array([1.0, 2.0]), rng)
    assert keep.tolist() == [True, True, True]
    with pytest.warns(RuntimeWarning, match="2 of 2 bins"):
        downsample_to_dndz(np.array([1.1, 1.2]), TargetDNDZ([0, 3], [10.0, 10.0]), 1.0,
                           np.array([1.0, 1.5, 2.0]), rng)


def test_population_thinning_mask_matches_target():
    rng = np.random.default_rng(1)
    n = 400_000
    area = 10.0
    z = rng.uniform(0.5, 2.5, n)                       # ~6700 per unit z per deg^2 per population
    line_true = rng.choice([0, 2, 4], size=n)          # Ha, OIII_5007, OII_3727 in equal numbers
    data = {"line_true": line_true, "redshift_obs": z}
    targets = {"Ha": TargetDNDZ.from_bins([1.0, 1.5, 2.0], [2000.0, 1000.0])}
    keep, info = population_thinning_mask(data, ROMAN_TRANSITIONS, targets, area, rng)
    ha = line_true == 0
    n_bin1 = np.sum(keep & ha & (z >= 1.0) & (z < 1.5)) / (0.5 * area)
    n_bin2 = np.sum(keep & ha & (z >= 1.5) & (z <= 2.0)) / (0.5 * area)
    assert abs(n_bin1 / 2000.0 - 1) < 0.05 and abs(n_bin2 / 1000.0 - 1) < 0.05
    assert keep[ha & (z < 1.0)].all()                  # outside the target range: kept
    assert keep[~ha].all()                             # populations without a target: kept
    keep2, _ = population_thinning_mask(data, ROMAN_TRANSITIONS, targets, area, rng, outside_range="drop")
    assert not keep2[ha & (z < 1.0)].any() and not keep2[~ha].any()


def test_pipeline_resolves_reference_targets():
    cfg = RomanMockConfig.from_dict({
        "sensitivity": {"type": "constant", "flux_limit": 1.5e-16, "wave_range": [10000.0, 19300.0]},
        "population_thinning": {"targets": {"Ha": {}, "OIII_5007": {"dust_av": 1.92}}},
    })
    pipe = RomanMockPipeline(cfg)
    t = pipe._target({}, default_line="Ha")
    assert t(np.array([1.15]))[0] == 7422                       # Ha, 1.5e-16, AV 1.65 at z 1.1-1.2
    t = pipe._target(cfg.population_thinning.targets["OIII_5007"], default_line="OIII_5007")
    assert t(np.array([2.05]))[0] == 317                        # OIII, 1.5e-16, AV 1.92 at z 2.0-2.1
    t = pipe._target({"z": [0, 1], "dndz": [5, 5]})
    assert t(np.array([0.5]))[0] == 5
    assert cfg.to_dict()["population_thinning"]["targets"]["Ha"] == {}

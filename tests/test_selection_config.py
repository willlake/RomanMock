import numpy as np
import pytest

from romanmock.config import RomanMockConfig, SampleConfig
from romanmock.lines import ROMAN_TRANSITIONS
from romanmock.redshift import IdOutcome
from romanmock.selection import (
    BoxFootprint,
    SampleDefinition,
    TargetDNDZ,
    downsample_to_dndz,
    mock_dndz,
    sample_mask,
)


def _parent(n=50000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.5, 2.5, n)
    outcome = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.1, 0.4, 0.3, 0.1, 0.1]).astype(np.int8)
    z[outcome == 0] = np.nan
    z[outcome == 4] = np.nan
    return {
        "redshift_assigned": z,
        "id_outcome": outcome,
        "line_assigned": rng.choice([-1, 0, 2], size=n).astype(np.int8),
        "n_det": rng.integers(0, 4, n).astype(np.uint8),
        "ra": rng.uniform(0, 90, n),
        "dec": rng.uniform(50, 70, n),
        "logsm_obs": rng.normal(10, 0.5, n),
    }


def test_sample_mask_cuts():
    d = _parent()
    s = SampleDefinition(z_min=1.0, z_max=2.0, assigned_lines=["Ha"], footprint=BoxFootprint(0, 45, 55, 65),
                         extra_cuts={"logsm_obs": [9.5, None]})
    m = sample_mask(d, ROMAN_TRANSITIONS, s)
    assert m.any()
    assert np.all((d["redshift_assigned"][m] >= 1.0) & (d["redshift_assigned"][m] <= 2.0))
    assert np.all(d["line_assigned"][m] == 0)
    assert np.all(np.isin(d["id_outcome"][m], [1, 2, 3]))
    assert np.all(d["ra"][m] <= 45) and np.all(d["logsm_obs"][m] >= 9.5)


def test_outcome_names_accepted():
    d = _parent()
    s = SampleDefinition(outcomes=["MULTI_LINE"])
    m = sample_mask(d, ROMAN_TRANSITIONS, s)
    assert np.all(d["id_outcome"][m] == IdOutcome.MULTI_LINE)


def test_downsample_to_target():
    rng = np.random.default_rng(0)
    area = 100.0
    z = rng.uniform(1.0, 2.0, 200000)  # mock dN/dz = 2000 per unit z per deg^2
    edges = np.linspace(1.0, 2.0, 11)
    target = TargetDNDZ([1.0, 2.0], [500.0, 500.0])
    keep, p = downsample_to_dndz(z, target, area, edges, rng)
    np.testing.assert_allclose(p, 0.25, atol=0.02)
    assert abs(keep.mean() - 0.25) < 0.01


def test_downsample_warns_on_shortage():
    rng = np.random.default_rng(0)
    z = rng.uniform(1.0, 2.0, 1000)
    edges = np.linspace(1.0, 2.0, 5)
    target = TargetDNDZ([1.0, 2.0], [1e6, 1e6])
    with pytest.warns(RuntimeWarning):
        keep, p = downsample_to_dndz(z, target, 1.0, edges, rng)
    assert keep.all() and np.all(p == 1.0)


def test_config_roundtrip(tmp_path):
    cfg = RomanMockConfig.from_dict({
        "catalog": {"path": "/x"}, "output": {"dir": "/y"},
        "confusion": {"type": "matrix", "rows": {"Ha": {"OIII_5007": 0.1}}},
        "samples": [{"name": "s", "z_min": 1.0, "assigned_lines": ["Ha"], "downsample": {"nbins": 5, "target": {"z": [0, 4], "dndz": [1, 1]}}}],
    })
    p = tmp_path / "c.yaml"
    cfg.to_yaml(p)
    back = RomanMockConfig.from_yaml(p)
    assert back.to_dict() == cfg.to_dict()
    assert isinstance(back.samples[0], SampleConfig) and back.samples[0].downsample.nbins == 5
    with pytest.raises(KeyError):
        RomanMockConfig.from_dict({"output": {"bogus": 1}})


def test_box_footprint_area_and_healpix_intersection():
    import healpy as hp

    box = BoxFootprint(0.0, 90.0, 0.0, 90.0)  # one octant
    assert abs(box.area_deg2() - 41252.96 / 8) < 1.0

    class FakeCatalog:
        def get_pixels(self, nside):
            return np.arange(hp.nside2npix(nside))  # full sky

    from romanmock.selection import sky_area_deg2

    full = sky_area_deg2(FakeCatalog(), 64)
    assert abs(full - 41252.96) < 1.0
    octant = sky_area_deg2(FakeCatalog(), 64, footprint=box)
    assert abs(octant / full - 0.125) < 0.01

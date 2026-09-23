import numpy as np
import pytest

from romanmock.lines import (
    ALL_TRANSITIONS,
    DIFFSKY_LINE_WAVELENGTHS,
    PLAN_SPECIES,
    ROMAN_TRANSITIONS,
    LineFeature,
    LineSet,
    get_line_set,
)


def test_presets_have_documented_bit_order():
    assert ROMAN_TRANSITIONS.names[:2] == ["Ha", "Hb"]
    assert ROMAN_TRANSITIONS.bit("Ha") == 0 and ROMAN_TRANSITIONS.bit_value("OIII_5007") == 4
    assert PLAN_SPECIES.names == ["Ha", "Hb", "OIII", "OII", "SIII"]
    assert len(ALL_TRANSITIONS) == len(DIFFSKY_LINE_WAVELENGTHS)


def test_required_columns_cover_components():
    cols = ROMAN_TRANSITIONS.required_columns
    assert set(cols) == set(DIFFSKY_LINE_WAVELENGTHS)
    assert "O_II_3726" in ROMAN_TRANSITIONS["OII_3727"].components


def test_config_roundtrip():
    cfg = ROMAN_TRANSITIONS.to_config()
    ls = LineSet.from_config(cfg, name="rt")
    assert ls.names == ROMAN_TRANSITIONS.names
    np.testing.assert_allclose(ls.rest_waves, ROMAN_TRANSITIONS.rest_waves)
    assert get_line_set("plan_species") is PLAN_SPECIES
    assert get_line_set(ls) is ls


def test_feature_from_dict_defaults_rest_wave():
    f = LineFeature.from_dict({"name": "Ha", "components": "Ba_alpha_6563"})
    assert f.rest_wave == DIFFSKY_LINE_WAVELENGTHS["Ba_alpha_6563"]
    with pytest.raises(ValueError):
        LineFeature.from_dict({"name": "blend", "components": ["O_II_3726", "O_II_3729"]})


def test_duplicate_names_rejected():
    f = ROMAN_TRANSITIONS["Ha"]
    with pytest.raises(ValueError):
        LineSet([f, f])


def test_with_rest_waves_from_ssp_data():
    from collections import namedtuple

    W = namedtuple("W", ["Ba_alpha_6563"])
    S = namedtuple("S", ["ssp_emline_wave"])
    ls = ROMAN_TRANSITIONS.with_rest_waves_from(S(W(6000.0)))
    assert ls["Ha"].rest_wave == 6000.0
    assert ls["Hb"].rest_wave == ROMAN_TRANSITIONS["Hb"].rest_wave

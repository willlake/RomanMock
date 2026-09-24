"""Emission-line features and the bitmask ordering.

A *feature* is one detectable spectral feature. It can be a single transition
(``Ba_alpha_6563``) or a blend of several diffsky luminosity columns that Roman
cannot separate (the [O II] 3726, 3729 doublet). Bit ``i`` of the detection bitmask
corresponds to feature ``i`` of a :class:`LineSet`; the ordering is fixed by the
LineSet and written into the output header, as requested by the plan's
implementation note.

Rest wavelengths are vacuum wavelengths in Angstrom. The default values are those
stored in the diffsky mock's ``ssp_data`` file (``ssp_emline_wave``); use
:meth:`LineSet.with_rest_waves_from` to take them from the catalog itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

# Vacuum rest-frame wavelengths (Angstrom) of the diffsky "roman_grs_pit" line set,
# as stored in diffsky_<mock>_ssp_data.hdf5 / ssp_emline_wave.
DIFFSKY_LINE_WAVELENGTHS: dict[str, float] = {
    "Ba_alpha_6563": 6564.723,
    "Ba_beta_4861": 4862.763,
    "O_III_4363": 4364.294,
    "O_III_4959": 4960.370,
    "O_III_5007": 5008.314,
    "O_II_3726": 3727.118,
    "O_II_3729": 3730.119,
    "S_III_6312": 6313.902,
    "S_III_9069": 9071.246,
    "S_III_9532": 9533.378,
}

MAX_FEATURES = 32  # bitmask is stored as uint32


@dataclass(frozen=True)
class LineFeature:
    """One detectable spectral feature.

    Parameters
    ----------
    name : str
        Short identifier used in output column names (e.g. ``"Ha"``).
    rest_wave : float
        Vacuum rest wavelength in Angstrom used for ``lambda_obs = rest_wave (1+z)``.
        For a blend this is a representative wavelength (see presets).
    components : tuple of str
        diffsky luminosity columns whose luminosities are summed to form the feature.
    label : str
        Human readable description.
    blend : bool
        Only meaningful for multi-component features. ``True``: the components are an
        unresolved blend (e.g. the [O II] doublet) and are detected as one feature at
        ``rest_wave``. ``False`` (default for multi-component features): the components are
        separated transitions, each detected independently at its own rest wavelength with
        its own bandpass test; the feature is detected when any component is. Summing
        separated lines before the bandpass test would let an out-of-band component count
        towards a detection, or discard an in-band one (review item B3).
    """

    name: str
    rest_wave: float
    components: tuple[str, ...]
    label: str = ""
    blend: bool = False

    def __post_init__(self):
        if not self.components:
            raise ValueError(f"feature {self.name!r} needs at least one component column")
        if self.rest_wave <= 0:
            raise ValueError(f"feature {self.name!r} has non-positive rest wavelength")

    @property
    def is_blend(self) -> bool:
        """Detected as one feature (single transition or unresolved blend)."""
        return len(self.components) == 1 or self.blend

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "rest_wave": float(self.rest_wave),
            "components": list(self.components),
            "label": self.label,
            "blend": bool(self.blend),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "LineFeature":
        comps = d.get("components", [d["name"]])
        if isinstance(comps, str):
            comps = [comps]
        rest_wave = d.get("rest_wave")
        if rest_wave is None:
            if len(comps) != 1 or comps[0] not in DIFFSKY_LINE_WAVELENGTHS:
                raise ValueError(f"rest_wave must be given for feature {d['name']!r}")
            rest_wave = DIFFSKY_LINE_WAVELENGTHS[comps[0]]
        return cls(d["name"], float(rest_wave), tuple(comps), d.get("label", ""), bool(d.get("blend", False)))


class LineSet:
    """An ordered collection of :class:`LineFeature` defining the bitmask.

    ``line_set[i]`` / ``line_set.bit("Ha")`` give the position of a feature; bit ``i``
    of the bitmask has value ``2**i``.
    """

    def __init__(self, features: Sequence[LineFeature], name: str = "custom"):
        features = tuple(features)
        if not features:
            raise ValueError("LineSet needs at least one feature")
        if len(features) > MAX_FEATURES:
            raise ValueError(f"at most {MAX_FEATURES} features are supported")
        names = [f.name for f in features]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate feature names in LineSet: {names}")
        self._features = features
        self._index = {f.name: i for i, f in enumerate(features)}
        self.name = name
        # rest wavelength of every component column (needed to detect separated transitions)
        self._component_waves = {}
        for f in features:
            for c in f.components:
                w = DIFFSKY_LINE_WAVELENGTHS.get(c, f.rest_wave if len(f.components) == 1 else None)
                if w is None:
                    raise ValueError(f"unknown rest wavelength for component {c!r} of feature {f.name!r}")
                self._component_waves[c] = float(w)

    # ---- basic container behaviour -------------------------------------------------
    def __len__(self) -> int:
        return len(self._features)

    def __iter__(self):
        return iter(self._features)

    def __getitem__(self, key) -> LineFeature:
        if isinstance(key, str):
            return self._features[self._index[key]]
        return self._features[key]

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def __repr__(self) -> str:
        return f"LineSet({self.name!r}, {self.names})"

    # ---- properties ----------------------------------------------------------------
    @property
    def features(self) -> tuple[LineFeature, ...]:
        return self._features

    @property
    def names(self) -> list[str]:
        return [f.name for f in self._features]

    @property
    def rest_waves(self) -> np.ndarray:
        return np.array([f.rest_wave for f in self._features], dtype=np.float64)

    @property
    def required_columns(self) -> list[str]:
        """All diffsky luminosity columns needed to build every feature."""
        cols: list[str] = []
        for f in self._features:
            for c in f.components:
                if c not in cols:
                    cols.append(c)
        return cols

    def component_wave(self, column: str) -> float:
        """Vacuum rest wavelength [A] of a component luminosity column."""
        return self._component_waves[column]

    @property
    def has_separated_components(self) -> bool:
        return any(not f.is_blend for f in self._features)

    def bit(self, name: str) -> int:
        """Bit position of a feature."""
        return self._index[name]

    def bit_value(self, name: str) -> int:
        """Integer value (2**bit) of a feature in the bitmask."""
        return 1 << self._index[name]

    def index_of(self, names: Iterable[str]) -> np.ndarray:
        return np.array([self._index[n] for n in names], dtype=np.int64)

    # ---- derived line sets ---------------------------------------------------------
    def with_rest_waves_from(self, ssp_data) -> "LineSet":
        """Return a copy whose single-component features take their rest wavelength
        from ``ssp_data.ssp_emline_wave`` (the diffsky SSP metadata of the catalog).
        Blended features keep their configured wavelength."""
        wave_tab = ssp_data.ssp_emline_wave
        table = dict(zip(wave_tab._fields, np.array(wave_tab, dtype=float)))
        new = []
        for f in self._features:
            if len(f.components) == 1 and f.components[0] in table:
                new.append(LineFeature(f.name, float(table[f.components[0]]), f.components, f.label, f.blend))
            else:
                new.append(f)
        ls = LineSet(new, name=self.name)
        for c, w in table.items():
            if c in ls._component_waves:
                ls._component_waves[c] = float(w)
        return ls

    def subset(self, names: Sequence[str], name: str | None = None) -> "LineSet":
        return LineSet([self[n] for n in names], name=name or f"{self.name}_subset")

    # ---- serialization -------------------------------------------------------------
    def to_config(self) -> list[dict]:
        return [f.to_dict() for f in self._features]

    @classmethod
    def from_config(cls, cfg, name: str = "custom") -> "LineSet":
        """Build from a preset name, a list of feature dicts, or a LineSet."""
        if isinstance(cfg, LineSet):
            return cfg
        if isinstance(cfg, str):
            return get_line_set(cfg)
        return cls([LineFeature.from_dict(d) for d in cfg], name=name)

    def describe(self) -> str:
        """Human readable table of the bit ordering (stored in output headers)."""
        lines = [f"LineSet {self.name!r}: bit -> feature (rest wavelength [A]; components)"]
        for i, f in enumerate(self._features):
            lines.append(
                f"  bit {i:2d} (value {1 << i:>10d}): {f.name:<10s} "
                f"{f.rest_wave:9.3f}  {'+'.join(f.components)}  {f.label}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------------
def _w(col):
    return DIFFSKY_LINE_WAVELENGTHS[col]


_OII_BLEND_WAVE = 0.5 * (_w("O_II_3726") + _w("O_II_3729"))

#: Individual transitions resolvable by the Roman grism, [O II] doublet blended.
#: This is the default: it follows comment 2 of the plan (specify transitions, not
#: species, before building the bitmask) while not pretending Roman resolves [O II].
ROMAN_TRANSITIONS = LineSet(
    [
        LineFeature("Ha", _w("Ba_alpha_6563"), ("Ba_alpha_6563",), "H-alpha 6563"),
        LineFeature("Hb", _w("Ba_beta_4861"), ("Ba_beta_4861",), "H-beta 4861"),
        LineFeature("OIII_5007", _w("O_III_5007"), ("O_III_5007",), "[O III] 5007"),
        LineFeature("OIII_4959", _w("O_III_4959"), ("O_III_4959",), "[O III] 4959"),
        LineFeature("OII_3727", _OII_BLEND_WAVE, ("O_II_3726", "O_II_3729"), "[O II] 3726+3729 blend", blend=True),
        LineFeature("SIII_9532", _w("S_III_9532"), ("S_III_9532",), "[S III] 9532"),
        LineFeature("SIII_9069", _w("S_III_9069"), ("S_III_9069",), "[S III] 9069"),
        LineFeature("OIII_4363", _w("O_III_4363"), ("O_III_4363",), "[O III] 4363 (auroral)"),
        LineFeature("SIII_6312", _w("S_III_6312"), ("S_III_6312",), "[S III] 6312 (auroral)"),
    ],
    name="roman_transitions",
)

#: The five species of the plan's section 2 (bits 0..4 in the order written there).
#: [O III] and [S III] are *separated* transitions: each component is detected on its own
#: (own wavelength and bandpass test) and the species bit is set when any component is
#: detected; the species flux column is the sum. [O II] is an unresolved blend. The
#: representative rest wavelength of a species is that of its strongest component.
PLAN_SPECIES = LineSet(
    [
        LineFeature("Ha", _w("Ba_alpha_6563"), ("Ba_alpha_6563",), "H-alpha"),
        LineFeature("Hb", _w("Ba_beta_4861"), ("Ba_beta_4861",), "H-beta"),
        LineFeature("OIII", _w("O_III_5007"), ("O_III_4959", "O_III_5007"), "[O III] 4959+5007 (separated)", blend=False),
        LineFeature("OII", _OII_BLEND_WAVE, ("O_II_3726", "O_II_3729"), "[O II] 3726+3729 blend", blend=True),
        LineFeature("SIII", _w("S_III_9532"), ("S_III_9069", "S_III_9532"), "[S III] 9069+9532 (separated)", blend=False),
    ],
    name="plan_species",
)

#: Every diffsky transition as its own bit, in catalog order.
ALL_TRANSITIONS = LineSet(
    [LineFeature(col, w, (col,), col) for col, w in DIFFSKY_LINE_WAVELENGTHS.items()],
    name="all_transitions",
)

LINE_SET_PRESETS: dict[str, LineSet] = {
    "roman_transitions": ROMAN_TRANSITIONS,
    "plan_species": PLAN_SPECIES,
    "all_transitions": ALL_TRANSITIONS,
}

DEFAULT_LINE_SET = "roman_transitions"


def get_line_set(spec="roman_transitions") -> LineSet:
    """Resolve a preset name, a list of feature dicts, or a LineSet into a LineSet."""
    if isinstance(spec, LineSet):
        return spec
    if isinstance(spec, str):
        try:
            return LINE_SET_PRESETS[spec]
        except KeyError:
            raise KeyError(f"unknown line set preset {spec!r}; known: {sorted(LINE_SET_PRESETS)}")
    return LineSet.from_config(spec)

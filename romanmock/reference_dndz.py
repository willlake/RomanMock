"""Reference redshift distributions from the literature, usable as downsampling targets.

Currently: Wang et al. (2022), "The High Latitude Spectroscopic Survey on the Nancy Grace
Roman Space Telescope", ApJ 928, 1 (arXiv:2110.01829), Tables 1 and 2. They list ``dN/dz``
per square degree of H-alpha and [O III] emitting galaxies in bins of width 0.1 in
redshift, predicted from the Galacticus semi-analytic model on the UNIT lightcone, for six
line-flux limits and two dust models (A_V = 1.65 and 1.92). Central values only; the
tabulated uncertainties (jackknife + cosmic variance, typically 5-10 %) are not reproduced.

Important: the tables describe *true emitters above the flux limit*. Per the table caption,
"the observational efficiency is not included", i.e. the HLSS completeness (60 %) and purity
(90 %) requirements are not applied. They are therefore directly comparable to the mock's
detected-line population (galaxies with the line above the flux limit), not to the final
assigned-redshift sample.
"""

from __future__ import annotations

import numpy as np

from .selection import TargetDNDZ

WANG2022_CITATION = "Wang et al. 2022, ApJ 928, 1 (arXiv:2110.01829), Tables 1-2"
WANG2022_FLUX_LIMITS = (5e-17, 7e-17, 8.5e-17, 1e-16, 1.5e-16, 2e-16)   # erg/s/cm^2, table columns
WANG2022_DUST_AV = (1.65, 1.92)

# rows: redshift bins of width 0.1 starting at the given z_min; columns: WANG2022_FLUX_LIMITS
_HA_EDGES = np.round(np.arange(0.5, 2.0 + 1e-9, 0.1), 2)
_HA_165 = np.array([
    [34725, 26812, 22912, 19912, 13662, 9820],
    [42560, 32052, 26952, 23110, 14995, 10420],
    [34850, 26302, 22022, 18567, 11675, 7802],
    [35750, 26277, 21670, 17942, 10555, 6897],
    [32662, 23500, 18935, 15570, 8752, 5527],
    [36517, 25317, 20057, 16235, 8780, 5340],
    [34552, 23292, 17970, 14312, 7422, 4260],
    [26615, 17502, 13287, 10377, 5280, 3055],
    [31170, 20030, 15107, 11615, 5552, 3025],
    [26380, 16642, 12372, 9270, 4267, 2247],
    [22125, 13555, 9770, 7190, 2985, 1355],
    [19727, 11752, 8432, 6075, 2342, 1050],
    [17817, 9895, 6795, 4705, 1740, 832],
    [11732, 6157, 4092, 2792, 955, 477],
    [9955, 5070, 3340, 2232, 820, 390],
], dtype=float)
_HA_192 = np.array([
    [29787, 22795, 19327, 16642, 10935, 7732],
    [36100, 26830, 22317, 18820, 11657, 7820],
    [29610, 21915, 17890, 14985, 8875, 5815],
    [29992, 21562, 17205, 14072, 7952, 4920],
    [27007, 18835, 14780, 11812, 6445, 3960],
    [29597, 19932, 15470, 12120, 6282, 3667],
    [27627, 17837, 13595, 10620, 5042, 2865],
    [20987, 13202, 9777, 7537, 3625, 1990],
    [24215, 14995, 10900, 8217, 3727, 1835],
    [20185, 12272, 8692, 6410, 2765, 1327],
    [16742, 9675, 6625, 4790, 1785, 800],
    [14670, 8350, 5582, 3887, 1352, 667],
    [12645, 6732, 4345, 2997, 1072, 542],
    [8095, 4047, 2562, 1640, 575, 310],
    [6792, 3305, 2027, 1367, 502, 255],
], dtype=float)
# summary rows of Table 1 (z 1.0-2.0 and 0.5-2.0), kept for the transcription self-test
_HA_SUMMARY = {
    1.65: {(1.0, 2.0): [23659, 14921, 11122, 8480, 4014, 2203], (0.5, 2.0): [27809, 18944, 14914, 11993, 6652, 4166]},
    1.92: {(1.0, 2.0): [18155, 11035, 7957, 5958, 2673, 1426], (0.5, 2.0): [22270, 14819, 11406, 9061, 4839, 2967]},
}

_O3_EDGES = np.round(np.arange(1.0, 3.0 + 1e-9, 0.1), 2)
_O3_165 = np.array([
    [10437, 6160, 4447, 3440, 1610, 977],
    [10212, 5907, 4225, 3167, 1550, 922],
    [8360, 4797, 3380, 2560, 1155, 697],
    [10027, 5727, 4125, 3095, 1540, 875],
    [8987, 5147, 3690, 2732, 1292, 752],
    [7900, 4530, 3217, 2355, 1015, 562],
    [8365, 4792, 3267, 2355, 962, 490],
    [6957, 3870, 2690, 1930, 812, 455],
    [5412, 2857, 1937, 1305, 480, 262],
    [4442, 2277, 1532, 1052, 445, 275],
    [3055, 1725, 1185, 875, 430, 302],
    [2265, 1215, 817, 597, 282, 190],
    [3210, 1642, 1102, 715, 307, 227],
    [3170, 1615, 1097, 820, 440, 302],
    [2722, 1305, 817, 520, 212, 122],
    [2360, 1087, 677, 477, 237, 162],
    [2090, 850, 480, 312, 117, 67],
    [1932, 795, 435, 295, 105, 60],
    [1942, 847, 477, 337, 155, 105],
    [1337, 522, 345, 222, 100, 67],
], dtype=float)
_O3_192 = np.array([
    [6880, 3920, 2892, 2117, 1032, 610],
    [6590, 3707, 2657, 1942, 957, 557],
    [5362, 2992, 2145, 1562, 727, 427],
    [6442, 3627, 2610, 1967, 935, 572],
    [5752, 3215, 2267, 1677, 790, 482],
    [5155, 2780, 1857, 1350, 585, 340],
    [5377, 2840, 1872, 1315, 505, 290],
    [4362, 2340, 1555, 1110, 477, 310],
    [3252, 1640, 1050, 682, 277, 167],
    [2680, 1305, 845, 605, 282, 180],
    [1962, 1047, 715, 555, 317, 235],
    [1422, 710, 490, 345, 200, 137],
    [1872, 937, 567, 395, 230, 175],
    [1845, 980, 687, 522, 310, 265],
    [1547, 667, 375, 267, 122, 97],
    [1297, 587, 377, 275, 175, 120],
    [1067, 395, 245, 160, 72, 30],
    [960, 372, 225, 140, 60, 35],
    [1045, 405, 265, 195, 110, 67],
    [625, 290, 167, 122, 70, 35],
], dtype=float)
_O3_SUMMARY = {
    1.65: {(1.0, 2.0): [8110, 4606, 3251, 2399, 1086, 627], (2.0, 3.0): [2408, 1160, 743, 517, 238, 160]},
    1.92: {(1.0, 2.0): [5185, 2836, 1975, 1433, 657, 393], (2.0, 3.0): [1364, 639, 411, 297, 166, 119]},
}

WANG2022_TABLES = {
    "Ha": {"edges": _HA_EDGES, 1.65: _HA_165, 1.92: _HA_192, "summary": _HA_SUMMARY, "table": 1},
    "OIII": {"edges": _O3_EDGES, 1.65: _O3_165, 1.92: _O3_192, "summary": _O3_SUMMARY, "table": 2},
}

#: feature names of the presets mapped onto the two tabulated species
_LINE_ALIASES = {"Ha": "Ha", "OIII": "OIII", "OIII_5007": "OIII", "O_III_5007": "OIII", "Ba_alpha_6563": "Ha"}


def wang2022_bins(line: str, flux_limit: float = 1e-16, dust_av: float = 1.65) -> tuple[np.ndarray, np.ndarray]:
    """Bin edges and ``dN/dz`` per deg^2 of Table 1 (H-alpha) or Table 2 ([O III])."""
    key = _LINE_ALIASES.get(line)
    if key is None:
        raise KeyError(f"no Wang et al. (2022) table for line {line!r}; available: Ha, OIII (5007)")
    limits = np.array(WANG2022_FLUX_LIMITS)
    j = np.where(np.isclose(limits, flux_limit, rtol=1e-3, atol=0.0))[0]
    if j.size == 0:
        raise ValueError(f"flux limit {flux_limit:.3g} is not tabulated; choose one of {WANG2022_FLUX_LIMITS}")
    if dust_av not in WANG2022_DUST_AV:
        raise ValueError(f"dust_av must be one of {WANG2022_DUST_AV}")
    tab = WANG2022_TABLES[key]
    return tab["edges"].copy(), tab[dust_av][:, j[0]].copy()


def wang2022_target(line: str, flux_limit: float = 1e-16, dust_av: float = 1.65) -> TargetDNDZ:
    """A piecewise-constant :class:`TargetDNDZ` from the Wang et al. (2022) tables."""
    edges, dndz = wang2022_bins(line, flux_limit, dust_av)
    return TargetDNDZ.from_bins(edges, dndz)


REFERENCE_TARGETS = {"wang2022": wang2022_target}


def reference_target(name: str, **kwargs) -> TargetDNDZ:
    """Look up a named reference (currently ``"wang2022"``) with its keyword arguments."""
    try:
        return REFERENCE_TARGETS[name](**kwargs)
    except KeyError:
        raise KeyError(f"unknown reference dN/dz {name!r}; known: {sorted(REFERENCE_TARGETS)}")

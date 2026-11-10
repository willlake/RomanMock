"""Rescaling of the emission-line fluxes of the mock, galaxy by galaxy.

A rescaling multiplies the flux of each feature of a galaxy by a constant and, if asked, by a
log-normal factor drawn once per galaxy:

    log10 F'_X = log10 F_X + offset_X + sigma_X * eps,    eps ~ N(0, 1)

``eps`` is a fixed function of the galaxy's ``gal_id`` and of ``seed`` (:func:`galaxy_normal`): a galaxy
gets the same deviate in every slice, chunk, rerun and subsample, and all its lines with a scatter move
together. Features without an offset or scatter, and fluxes that are zero, negative or NaN, are left as
they are. The rescaling acts on fluxes or luminosities alike (it is a ratio).

Models are selected by ``type`` in a config, like the sensitivity, detection and confusion models
(:data:`RESCALING_MODELS`, :func:`build_rescaling`):

* ``line_factors`` (:class:`LineFactors`): explicit offsets and scatters per feature name.
* ``galacticus_like`` (:class:`GalacticusLike`): the diffsky lines rescaled toward the Galacticus mock
  of the GRS-PIT, the recipe of ``Projects/Roman/Fullshape/Catalog/diffsky_line_rescaling.ipynb``
  (2026-11-09). Only the features that Galacticus also has are rescaled; the dust of [S III] and Paα is
  swapped galaxy by galaxy, which needs the Hα and Paα fluxes of each galaxy.

The rescaling is array-level (dicts or arrays of fluxes); it is not wired into ``make_catalog`` or the
pipeline. Apply it to the fluxes read from the base catalog (``base.read_slices``), then select.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.special import ndtri

from .lines import LineSet, get_line_set

DEFAULT_SEED = 20261109  #: seed of the per-galaxy deviate unless a model is given another

_U64 = np.uint64
_GOLDEN, _MIX1, _MIX2 = _U64(0x9E3779B97F4A7C15), _U64(0xBF58476D1CE4E5B9), _U64(0x94D049BB133111EB)


def calzetti_k(wave_aa) -> np.ndarray:
    """Calzetti et al. (2000) attenuation curve k(λ) = A(λ) / E(B−V), R_V = 4.05; λ in Å (valid 1200–22000 Å)."""
    x = np.asarray(wave_aa, dtype=np.float64) / 1e4
    return np.where(x < 0.63, 2.659 * (-2.156 + 1.509 / x - 0.198 / x**2 + 0.011 / x**3) + 4.05,
                    2.659 * (-1.857 + 1.040 / x) + 4.05)


def galaxy_normal(gal_id, seed: int = DEFAULT_SEED) -> np.ndarray:
    """One N(0, 1) deviate per galaxy, a fixed function of ``(gal_id, seed)``.

    The id is hashed with splitmix64 (offset by ``seed`` times the golden-ratio constant), the top 53 bits
    become a uniform deviate in (0, 1) and the inverse normal CDF turns it into a normal one. Deviates of
    different ids are independent for practical purposes; the same id always gets the same deviate.
    """
    x = np.asarray(gal_id).astype(np.uint64)
    with np.errstate(over="ignore"):
        x = x + _U64(int(seed) & 0xFFFFFFFFFFFFFFFF) * _GOLDEN + _GOLDEN
        x = (x ^ (x >> _U64(30))) * _MIX1
        x = (x ^ (x >> _U64(27))) * _MIX2
        x = x ^ (x >> _U64(31))
    return ndtri(((x >> _U64(11)).astype(np.float64) + 0.5) * 2.0**-53)


def _line_set(line_set) -> LineSet:
    return line_set if isinstance(line_set, LineSet) else get_line_set(line_set)


class LineRescaling:
    """Common part of the rescaling models: subclasses define :meth:`offsets` and :meth:`sigmas` (dex, one
    value per feature of a line set), optionally a per-galaxy part :meth:`galaxy_offsets`, and :meth:`to_config`."""

    type_name = "base"

    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed = int(seed)

    # ---- per-feature parameters --------------------------------------------------------
    def offsets(self, line_set) -> np.ndarray:
        """The constant part of the log10 factor of each feature (dex)."""
        raise NotImplementedError

    def sigmas(self, line_set) -> np.ndarray:
        """The scatter of each feature (dex), one shared deviate per galaxy."""
        raise NotImplementedError

    def galaxy_features(self, line_set) -> list[str]:
        """The features with a per-galaxy part (none by default)."""
        return []

    def required_features(self, line_set) -> list[str]:
        """The features whose fluxes the per-galaxy part reads (none by default)."""
        return []

    def galaxy_offsets(self, line_set, names: Sequence[str], flux_of) -> dict[str, np.ndarray]:
        """The per-galaxy part of the log10 factor of the features ``names`` (dex, one value per galaxy);
        ``flux_of(name)`` returns the flux of a feature for the same galaxies."""
        return {}

    def has_scatter(self, line_set) -> bool:
        return bool(np.any(self.sigmas(line_set) > 0))

    def max_log_boost(self, line_set, n_sigma: float = 5.0) -> float:
        """Largest brightening of any feature, in dex, for a deviate of ``n_sigma``: how much deeper than a
        catalog's threshold its source must be for the rescaled selection to be complete."""
        return float(np.max(self.offsets(line_set) + n_sigma * self.sigmas(line_set)))

    def factors(self, line_set) -> dict[str, float]:
        """The constant factor of each feature (the scatter and any per-galaxy part left out)."""
        ls = _line_set(line_set)
        return dict(zip(ls.names, (10.0 ** self.offsets(ls)).tolist()))

    # ---- application ----------------------------------------------------------------------
    def _log_factors(self, ls: LineSet, names: Sequence[str], gal_id, flux_of):
        off, sig = self.offsets(ls), self.sigmas(ls)
        eps = None
        if any(sig[ls.names.index(n)] > 0 for n in names):
            if gal_id is None:
                raise ValueError("this rescaling has a per-galaxy scatter: give the gal_id of the galaxies")
            eps = galaxy_normal(gal_id, self.seed)
        per_galaxy = self.galaxy_offsets(ls, [n for n in names if n in self.galaxy_features(ls)], flux_of)
        for n in names:
            j = ls.names.index(n)
            log_factor = off[j] + (sig[j] * eps if sig[j] > 0 else 0.0) + per_galaxy.get(n, 0.0)
            if np.ndim(log_factor) or log_factor != 0:
                yield n, log_factor

    def apply(self, flux, line_set, gal_id=None) -> np.ndarray:
        """Rescaled copy of ``flux``, shape ``(n, n_features)`` in the order of ``line_set``."""
        ls = _line_set(line_set)
        flux = np.asarray(flux, dtype=np.float64)
        if flux.ndim != 2 or flux.shape[1] != len(ls):
            raise ValueError(f"flux must have shape (n, {len(ls)}) in the order {ls.names}, not {flux.shape}")
        if gal_id is not None and np.shape(gal_id) != (flux.shape[0],):
            raise ValueError(f"gal_id must have shape ({flux.shape[0]},), not {np.shape(gal_id)}")
        if self.galaxy_features(ls):
            missing = [n for n in self.required_features(ls) if n not in ls.names]
            if missing:
                raise ValueError(f"the per-galaxy part of this rescaling reads the fluxes of {missing}, which the line set lacks")
        out = flux.copy()
        for n, log_factor in self._log_factors(ls, ls.names, gal_id, lambda name: flux[:, ls.names.index(name)]):
            j = ls.names.index(n)
            pos = flux[:, j] > 0
            out[pos, j] = flux[pos, j] * 10.0 ** (log_factor[pos] if np.ndim(log_factor) else log_factor)
        return out

    def apply_columns(self, data: Mapping[str, np.ndarray], line_set, suffix: str = "_flux",
                      gal_id: str = "gal_id") -> dict:
        """Copy of a dict of columns (``base.read_slices`` output) with every ``<feature><suffix>`` column it
        holds rescaled; the other columns are passed through. ``data[gal_id]`` is needed for a scatter, and the
        columns of :meth:`required_features` for a per-galaxy part (for ``GalacticusLike``: Hα and Paα)."""
        ls = _line_set(line_set)
        present = [n for n in ls.names if f"{n}{suffix}" in data]
        if any(n in self.galaxy_features(ls) for n in present):
            missing = [f"{n}{suffix}" for n in self.required_features(ls) if f"{n}{suffix}" not in data]
            if missing:
                raise KeyError(f"the per-galaxy part of this rescaling reads {missing}: read those columns too")
        out = dict(data)
        ids = data.get(gal_id) if self.has_scatter(ls) else None
        flux_of = lambda name: np.asarray(data[f"{name}{suffix}"], dtype=np.float64)
        for n, log_factor in self._log_factors(ls, present, ids, flux_of):
            f = flux_of(n)
            out[f"{n}{suffix}"] = np.where(f > 0, f * 10.0 ** log_factor, f)
        return out

    # ---- description ----------------------------------------------------------------------
    def to_config(self) -> dict:
        raise NotImplementedError

    def describe(self, line_set=None) -> str:
        if line_set is None:
            return f"{self.type_name} rescaling"
        ls = _line_set(line_set)
        off, sig = self.offsets(ls), self.sigmas(ls)
        parts = [f"{n} x{10 ** o:.3f}" + (f" ±{s:g} dex" if s > 0 else "") for n, o, s in zip(ls.names, off, sig) if o or s]
        return f"{self.type_name} rescaling: " + (", ".join(parts) if parts else "no change")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.to_config()})"


class LineFactors(LineRescaling):
    """Explicit rescaling: ``log_offsets`` (dex) and ``scatter`` (dex, one shared deviate) per feature name.

    ``LineFactors({"OII_3727": 0.2}, {"OII_3727": 0.1})`` brightens [O II] by 0.2 dex with a scatter of 0.1 dex.
    """

    type_name = "line_factors"

    def __init__(self, log_offsets: Mapping[str, float] | None = None, scatter: Mapping[str, float] | None = None,
                 seed: int = DEFAULT_SEED):
        super().__init__(seed)
        self.log_offsets = {str(k): float(v) for k, v in (log_offsets or {}).items()}
        self.scatter = {str(k): float(v) for k, v in (scatter or {}).items()}
        if any(v < 0 for v in self.scatter.values()):
            raise ValueError("a scatter must not be negative")

    def _check(self, ls: LineSet):
        unknown = sorted((set(self.log_offsets) | set(self.scatter)) - set(ls.names))
        if unknown:
            raise KeyError(f"rescaling of unknown features {unknown}; the line set has {ls.names}")

    def offsets(self, line_set) -> np.ndarray:
        ls = _line_set(line_set)
        self._check(ls)
        return np.array([self.log_offsets.get(n, 0.0) for n in ls.names])

    def sigmas(self, line_set) -> np.ndarray:
        ls = _line_set(line_set)
        self._check(ls)
        return np.array([self.scatter.get(n, 0.0) for n in ls.names])

    def to_config(self) -> dict:
        return {"type": self.type_name, "log_offsets": dict(self.log_offsets), "scatter": dict(self.scatter), "seed": self.seed}


class GalacticusLike(LineRescaling):
    """The diffsky lines rescaled toward the Galacticus mock of the GRS-PIT.

    Every feature X of the line set that Galacticus also has (the names of the line set ``counterparts``, by
    default ``galacticus_transitions``; in ``roman_transitions_paa`` all but [O III] λ4363 and [S III] λ6312) gets

        log10 F'_X = log10 F_X + dust_X + delta_X + sigma_X eps

    Hα is the anchor and never changes: diffsky's attenuated Hα is fitted to the HiZELS Hα luminosity functions.
    The terms (``Projects/Roman/Fullshape/Catalog/diffsky_line_rescaling.ipynb``, measured on the galaxies with a
    line above 1.5e-16 at 0.5 < z < 3):

    * dust. Galacticus attenuates its lines with the Calzetti curve and an A(Hα) that grows with stellar mass
      (median 0.89 mag); diffsky with its steeper stellar curve (median 0.34 mag).
      - For the lines in ``galaxy_dust`` (default [S III] λλ9069, 9532 and Paα, the lines redward of Hα whose ratio
        to Hα has a floor at its dust-free value) the dust is swapped galaxy by galaxy: diffsky's own reddening of
        the line relative to Hα, A_D (r_X - 1) with A_D the galaxy's A(Hα) read from its Paα/Hα and r_X diffsky's
        curve ratio A_X / A(Hα), is replaced by the Calzetti reddening of A' = the Galacticus A(Hα) at the same
        quantile of the dust distribution (``dust_map``): dust_X = -0.4 [A' (k_X / k_Hα - 1) - A_D (r_X - 1)].
        Galaxies without dust keep the dust-free ratio, as in Galacticus. (A constant term put the floor of
        [S III]/Hα at 0.40 and left almost no single-line galaxy below z = 1 for the pairs with a 0.4 ratio of
        thresholds: Fullshape/Catalog/single_line_fraction_rescaled.ipynb.)
      - For the other lines, dust_X = -0.4 da_ha [k_X / k_Hα - 1], a constant with ``da_ha`` = 0.5 mag; their
        ``delta`` were derived with it.
    * ``delta`` (dex), nebular model and AGN: [O II] +0.37 (diffsky's single log U = -2, solar-gas model gives an
      intrinsic [O II]/Hβ of 1.04, Galacticus 2.09), [O III] +0.19 (Galacticus's AGN, a median 13 % of its
      [O III], and its nebular model), Hβ +0.04 and Paα -0.055 (case-B recombination ratios). Members of a
      doublet share it (their ratio is atomic physics).
    * ``sigma`` (dex), the galaxy-to-galaxy scatter of the excitation that diffsky's fixed U and gas metallicity
      leave out: [O III] 0.19, [O II] 0.11, one deviate shared by both (in Galacticus the two ratios to Hα
      correlate at 0.80).

    Factors at diffsky's median dust in ``roman_transitions_paa``: Hβ 0.92, [O III] 5007 and 4959 1.32,
    [O II] 1.65, [S III] 9532 1.19 and 9069 1.17, Paα 1.31. ``galaxy_dust=()`` gives the constant recipe of the
    first version (2026-11-09 morning): [S III] 1.20 / 1.17 and Paα 1.29 for every galaxy.
    """

    type_name = "galacticus_like"
    DEFAULT_DELTA = {"Hb": 0.04, "OIII_5007": 0.19, "OIII_4959": 0.19, "OII_3727": 0.37, "PaA": -0.055}
    DEFAULT_SIGMA = {"OIII_5007": 0.19, "OIII_4959": 0.19, "OII_3727": 0.11}
    DEFAULT_GALAXY_DUST = ("SIII_9532", "SIII_9069", "PaA")
    #: diffsky's line attenuation relative to Hα, A_X / A(Hα): weighted medians over its galaxies with a line above
    #: 1.5e-16, from the attenuation rebuilt from the stored dust parameters (diffsky_line_rescaling.ipynb, 1b)
    DIFFSKY_DUST_RATIO = {"SIII_9532": 0.531, "SIII_9069": 0.583, "PaA": 0.115}
    PAA_HA_INTRINSIC = -0.879        #: log10 Paα/Hα of diffsky's SSP line table (constant SFR, solar Z), dust-free
    #: A(Hα) [mag] of the galaxies with a line above 1.5e-16 at 0.5 < z < 3 at the quantiles 0, 0.025, ..., 1:
    #: diffsky (read from Paα/Hα, 383 k galaxies) and Galacticus (136 874 galaxies; 8 % have no dust)
    DUST_MAP_DIFFSKY = (0.0000, 0.0209, 0.0322, 0.0425, 0.0527, 0.0630, 0.0737, 0.0851, 0.0971, 0.1103, 0.1244, 0.1392, 0.1561,
                        0.1738, 0.1931, 0.2141, 0.2367, 0.2613, 0.2875, 0.3152, 0.3455, 0.3775, 0.4115, 0.4477, 0.4853, 0.5255,
                        0.5667, 0.6106, 0.6564, 0.7037, 0.7547, 0.8092, 0.8677, 0.9304, 0.9989, 1.0762, 1.1657, 1.2728, 1.4158,
                        1.6409, 3.7108)
    DUST_MAP_GALACTICUS = (0.0000, 0.0000, 0.0000, 0.0000, 0.0770, 0.1603, 0.2332, 0.2983, 0.3598, 0.4134, 0.4652, 0.5151, 0.5643,
                           0.6094, 0.6522, 0.6943, 0.7357, 0.7764, 0.8157, 0.8540, 0.8928, 0.9302, 0.9675, 1.0055, 1.0425, 1.0805,
                           1.1178, 1.1555, 1.1946, 1.2345, 1.2749, 1.3179, 1.3622, 1.4077, 1.4601, 1.5141, 1.5766, 1.6467, 1.7369,
                           1.8716, 2.8916)

    def __init__(self, da_ha: float = 0.5, delta: Mapping[str, float] | None = None, sigma: Mapping[str, float] | None = None,
                 seed: int = DEFAULT_SEED, counterparts: str | Sequence[str] = "galacticus_transitions",
                 galaxy_dust: Sequence[str] = DEFAULT_GALAXY_DUST, dust_map: Sequence[Sequence[float]] | None = None,
                 diffsky_dust_ratio: Mapping[str, float] | None = None, paa_ha_intrinsic: float = PAA_HA_INTRINSIC):
        super().__init__(seed)
        self.da_ha = float(da_ha)
        self.delta = {str(k): float(v) for k, v in (self.DEFAULT_DELTA if delta is None else delta).items()}
        self.sigma = {str(k): float(v) for k, v in (self.DEFAULT_SIGMA if sigma is None else sigma).items()}
        self.counterparts = counterparts if isinstance(counterparts, str) else [str(c) for c in counterparts]
        self.galaxy_dust = [str(n) for n in galaxy_dust]
        x, y = (self.DUST_MAP_DIFFSKY, self.DUST_MAP_GALACTICUS) if dust_map is None else dust_map
        self.dust_map = (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        self.diffsky_dust_ratio = {str(k): float(v) for k, v in
                                   (self.DIFFSKY_DUST_RATIO if diffsky_dust_ratio is None else diffsky_dust_ratio).items()}
        self.paa_ha_intrinsic = float(paa_ha_intrinsic)
        names = self._counterpart_names()
        unknown = sorted((set(self.delta) | set(self.sigma) | set(self.galaxy_dust)) - set(names))
        if unknown:
            raise KeyError(f"delta/sigma/galaxy_dust for {unknown}, which have no counterpart in {self.counterparts!r} ({names})")
        if any(v < 0 for v in self.sigma.values()):
            raise ValueError("a scatter must not be negative")
        if "Ha" in self.galaxy_dust:
            raise ValueError("Hα is the anchor of the rescaling; it cannot be in galaxy_dust")
        no_ratio = sorted(set(self.galaxy_dust) - set(self.diffsky_dust_ratio))
        if no_ratio:
            raise KeyError(f"galaxy_dust lines {no_ratio} need diffsky's curve ratio A_X / A(Hα) in diffsky_dust_ratio")
        if self.galaxy_dust and "PaA" not in self.diffsky_dust_ratio:
            raise KeyError("the per-galaxy dust reads A(Hα) from Paα/Hα and needs diffsky_dust_ratio['PaA']")
        if self.dust_map[0].shape != self.dust_map[1].shape or np.any(np.diff(self.dust_map[0]) <= 0) or np.any(np.diff(self.dust_map[1]) < 0):
            raise ValueError("dust_map is (diffsky A(Hα), Galacticus A(Hα)) at the same quantiles, both increasing")

    def _counterpart_names(self) -> list[str]:
        if isinstance(self.counterparts, str):
            return list(get_line_set(self.counterparts).names)
        return list(self.counterparts)

    def rescaled_features(self, line_set) -> list[str]:
        """The features of ``line_set`` that are rescaled: those Galacticus also has."""
        names = set(self._counterpart_names())
        return [n for n in _line_set(line_set).names if n in names]

    def galaxy_features(self, line_set) -> list[str]:
        rescaled = set(self.rescaled_features(line_set))
        return [n for n in _line_set(line_set).names if n in self.galaxy_dust and n in rescaled]

    def required_features(self, line_set) -> list[str]:
        return ["Ha", "PaA"] if self.galaxy_features(line_set) else []

    def _k_ratios(self, ls: LineSet) -> dict[str, float]:
        if "Ha" not in ls.names:
            raise ValueError(f"the galacticus_like rescaling is anchored on Hα; the line set has {ls.names}")
        k = calzetti_k(np.asarray(ls.rest_waves, dtype=np.float64))
        return dict(zip(ls.names, k / k[ls.names.index("Ha")]))

    def offsets(self, line_set) -> np.ndarray:
        ls = _line_set(line_set)
        kr = self._k_ratios(ls)
        rescaled, per_galaxy = set(self.rescaled_features(ls)), set(self.galaxy_features(ls))
        return np.array([0.0 if n not in rescaled else
                         self.delta.get(n, 0.0) + (0.0 if n in per_galaxy else -0.4 * self.da_ha * (kr[n] - 1.0))
                         for n in ls.names])

    def sigmas(self, line_set) -> np.ndarray:
        ls = _line_set(line_set)
        rescaled = set(self.rescaled_features(ls))
        return np.array([self.sigma.get(n, 0.0) if n in rescaled else 0.0 for n in ls.names])

    def galaxy_a_ha(self, flux_ha, flux_paa) -> np.ndarray:
        """A(Hα) [mag] of each diffsky galaxy from its Paα/Hα: Paα is barely attenuated and the dust-free ratio is
        fixed, so A = 2.5 [log10 (Paα/Hα) - intrinsic] / (1 - A(Paα)/A(Hα)), at least 0 (0 when a flux is not positive)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log10(np.asarray(flux_paa, dtype=np.float64) / np.asarray(flux_ha, dtype=np.float64))
            a = 2.5 * (r - self.paa_ha_intrinsic) / (1.0 - self.diffsky_dust_ratio["PaA"])
        return np.clip(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)

    def galacticus_a_ha(self, a_diffsky) -> np.ndarray:
        """The Galacticus A(Hα) at the quantile of the dust distribution where diffsky has ``a_diffsky``."""
        return np.interp(a_diffsky, *self.dust_map)

    def _dust_term(self, n: str, kr: Mapping[str, float], a_d):
        return -0.4 * (self.galacticus_a_ha(a_d) * (kr[n] - 1.0) - a_d * (self.diffsky_dust_ratio[n] - 1.0))

    def galaxy_offsets(self, line_set, names: Sequence[str], flux_of) -> dict[str, np.ndarray]:
        ls = _line_set(line_set)
        names = [n for n in names if n in self.galaxy_features(ls)]
        if not names:
            return {}
        kr = self._k_ratios(ls)
        a_d = self.galaxy_a_ha(flux_of("Ha"), flux_of("PaA"))
        return {n: self._dust_term(n, kr, a_d) for n in names}

    def typical_offsets(self, line_set, a_ha: float | None = None) -> np.ndarray:
        """The log10 factor of each feature for a galaxy of diffsky A(Hα) ``a_ha`` (default: the median of the
        galaxies of the dust map, 0.35 mag), scatter left out."""
        ls = _line_set(line_set)
        a = float(np.median(self.dust_map[0])) if a_ha is None else float(a_ha)
        off = self.offsets(ls).copy()
        kr = self._k_ratios(ls) if self.galaxy_features(ls) else {}
        for n in self.galaxy_features(ls):
            off[ls.names.index(n)] += self._dust_term(n, kr, a)
        return off

    def max_log_boost(self, line_set, n_sigma: float = 5.0) -> float:
        ls = _line_set(line_set)
        boost = self.offsets(ls) + n_sigma * self.sigmas(ls)
        if self.galaxy_features(ls):
            kr = self._k_ratios(ls)
            grid = np.linspace(0.0, float(self.dust_map[0][-1]), 400)
            for n in self.galaxy_features(ls):
                j = ls.names.index(n)
                boost[j] += float(np.max(self._dust_term(n, kr, grid)))
        return float(np.max(boost))

    def describe(self, line_set=None) -> str:
        if line_set is None:
            return f"{self.type_name} rescaling"
        ls = _line_set(line_set)
        off, sig, per = self.typical_offsets(ls), self.sigmas(ls), set(self.galaxy_features(ls))
        parts = [f"{n} x{10 ** o:.3f}" + (" (dust per galaxy; at the median dust)" if n in per else "")
                 + (f" ±{s:g} dex" if s > 0 else "") for n, o, s in zip(ls.names, off, sig) if o or s or n in per]
        return f"{self.type_name} rescaling: " + (", ".join(parts) if parts else "no change")

    def to_config(self) -> dict:
        return {"type": self.type_name, "da_ha": self.da_ha, "delta": dict(self.delta), "sigma": dict(self.sigma),
                "seed": self.seed, "counterparts": self.counterparts, "galaxy_dust": list(self.galaxy_dust),
                "dust_map": [self.dust_map[0].tolist(), self.dust_map[1].tolist()],
                "diffsky_dust_ratio": dict(self.diffsky_dust_ratio), "paa_ha_intrinsic": self.paa_ha_intrinsic}


RESCALING_MODELS = {"line_factors": LineFactors, "galacticus_like": GalacticusLike}


def build_rescaling(cfg) -> LineRescaling | None:
    """A rescaling model from a config (``{"type": "galacticus_like", ...}``), a type name, a model, or
    ``None`` (no rescaling)."""
    if cfg is None or cfg is False:
        return None
    if isinstance(cfg, LineRescaling):
        return cfg
    if isinstance(cfg, str):
        cfg = {"type": cfg}
    cfg = dict(cfg)
    kind = cfg.pop("type", "line_factors")
    try:
        model = RESCALING_MODELS[kind]
    except KeyError:
        raise KeyError(f"unknown rescaling model {kind!r}; known: {sorted(RESCALING_MODELS)}") from None
    return model(**cfg)

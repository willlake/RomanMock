"""Configuration dataclasses and YAML I/O for the pipeline.

Every model choice of the plan is a small dictionary with a ``type`` key resolved by a
factory (``build_sensitivity``, ``build_detection``, ``build_confusion``,
``build_footprint``), so new models can be selected from YAML without code changes
in the pipeline.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .lines import DEFAULT_LINE_SET
from .roman import DEFAULT_LINE_FLUX_LIMIT, GRISM_WAVE_RANGE_ANGSTROM

DEFAULT_PASSTHROUGH = ["ra", "dec", "redshift_true"]
#: columns written to analysis-sample files by default (stage B); None = every parent column
DEFAULT_SAMPLE_COLUMNS = ["ra", "dec", "redshift_true", "redshift_rsd", "redshift_assigned", "id_outcome"]


@dataclass
class CatalogConfig:
    path: str = ""
    files: list[str] | None = None      # explicit slice files (relative to path or absolute)
    steps: list[int] | None = None      # restrict to these HACC step numbers
    z_min: float | None = None          # restrict to slices overlapping [z_min, z_max]
    z_max: float | None = None
    open_kwargs: dict = field(default_factory=lambda: {"synth_cores": False, "keep_top_host": True})
    rest_waves_from_catalog: bool = True  # take line wavelengths from the diffsky ssp_data file


@dataclass
class FluxConfig:
    redshift_column: str = "redshift_true"  # redshift used for the luminosity distance
    backend: str = "astropy"                # or "dsps"
    suffix: str = "_flux"


@dataclass
class RedshiftConfig:
    rsd_column: str = "redshift_obs"   # z including peculiar velocity (lambda_obs and z assignment)
    sigma_z: float = 0.0               # optional Gaussian error sigma_z (1+z) for multi-line redshifts
    context_columns: list[str] = field(default_factory=list)  # passthrough columns handed to the confusion model


@dataclass
class OutputConfig:
    dir: str = ""
    tag: str = "romanmock"
    overwrite: bool = False
    keep_undetected: bool = False        # keep galaxies with bitmask == 0 in the parent catalog
    passthrough_columns: list[str] = field(default_factory=lambda: list(DEFAULT_PASSTHROUGH))
    # The parent catalog always contains redshift_rsd (copy of redshift.rsd_column),
    # redshift_assigned and id_outcome. The identification columns (line_bitmask, n_det,
    # line_true, line_assigned) are needed by stage B and the diagnostics.
    store_identification: bool = True
    store_fluxes: bool = False
    store_wave_obs: bool = False
    store_snr: bool = False
    store_measured_flux: bool = False    # only differs from flux for noisy detection models
    sample_columns: list[str] | None = field(default_factory=lambda: list(DEFAULT_SAMPLE_COLUMNS))


@dataclass
class DownsampleConfig:
    """Thin one analysis sample to a target dN/dz of *assigned* redshift.

    ``target`` is a target spec: ``{"reference": "wang2022", "line": "Ha", "flux_limit": 1.5e-16,
    "dust_av": 1.65}`` (line defaults to the sample's first assigned line, flux_limit to the
    run's constant flux limit), ``{"file": "dndz.txt"}`` (two columns z, dN/dz per deg^2) or
    ``{"z": [...], "dndz": [...]}``.
    """

    target_file: str | None = None   # shorthand for target: {"file": ...}
    target: dict | None = None
    z_edges: list[float] | None = None
    nbins: int = 40
    area_deg2: float | None = None   # if None, estimated from the HEALPix index (inside the footprint)
    nside: int = 256
    smooth_bins: float = 1.0
    poly_degree: int | None = None   # smooth selection function: polynomial fit to log dN/dz of the mock


@dataclass
class PopulationThinningConfig:
    """Thin the detected-line populations to reference dN/dz in bins of true redshift, before
    sample selection (what forecast tables such as Wang et al. 2022 describe).

    ``targets`` maps feature names to target specs (see :class:`DownsampleConfig`); a spec
    may be ``{}`` to mean the default reference at the run's flux limit.
    """

    targets: dict = field(default_factory=lambda: {"Ha": {"reference": "wang2022"}, "OIII_5007": {"reference": "wang2022"}})
    by: str = "line_true"            # population membership column
    z_column: str = "redshift_rsd"   # true redshift (with peculiar velocity) used for the comparison
    bin_width: float = 0.1           # used only for targets without their own bins
    area_deg2: float | None = None
    nside: int = 256
    smooth_bins: float = 0.0
    poly_degree: int | None = None
    outside_range: str = "keep"      # galaxies with no target / outside the target's z range


@dataclass
class SampleConfig:
    name: str = "sample"
    z_min: float | None = None
    z_max: float | None = None
    outcomes: list = field(default_factory=lambda: ["MULTI_LINE", "SINGLE_CORRECT", "SINGLE_MISID"])
    assigned_lines: list[str] | None = None
    membership: str = "assigned"     # "assigned": identity from the redshift procedure (disjoint samples);
                                     # "detected": multi-line galaxies belong to every sample whose line they detect
    n_det_min: int | None = None
    footprint: dict | None = None
    extra_cuts: dict = field(default_factory=dict)
    downsample: DownsampleConfig | None = None


@dataclass
class RomanMockConfig:
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    line_set: Any = DEFAULT_LINE_SET   # preset name or list of feature dicts
    flux: FluxConfig = field(default_factory=FluxConfig)
    sensitivity: dict = field(default_factory=lambda: {
        "type": "constant", "flux_limit": DEFAULT_LINE_FLUX_LIMIT, "wave_range": list(GRISM_WAVE_RANGE_ANGSTROM)})
    detection: dict = field(default_factory=lambda: {"type": "threshold"})
    confusion: dict = field(default_factory=lambda: {"type": "identity", "p_fail": 0.0})
    redshift: RedshiftConfig = field(default_factory=RedshiftConfig)
    population_thinning: PopulationThinningConfig | None = None
    samples: list[SampleConfig] = field(default_factory=list)
    seed: int = 12345

    # ---- (de)serialisation --------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "RomanMockConfig":
        d = dict(d or {})
        samples = [_dc(SampleConfig, s, downsample=DownsampleConfig) for s in d.pop("samples", []) or []]
        thinning = d.pop("population_thinning", None)
        return cls(
            catalog=_dc(CatalogConfig, d.pop("catalog", {})),
            output=_dc(OutputConfig, d.pop("output", {})),
            flux=_dc(FluxConfig, d.pop("flux", {})),
            redshift=_dc(RedshiftConfig, d.pop("redshift", {})),
            population_thinning=None if thinning is None else _dc(PopulationThinningConfig, thinning),
            samples=samples,
            **d,
        )

    @classmethod
    def from_yaml(cls, path) -> "RomanMockConfig":
        with open(Path(path)) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path=None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path is not None:
            Path(path).write_text(text)
        return text


def _dc(cls, d, **nested):
    """Instantiate dataclass ``cls`` from dict ``d``; ``nested`` maps field -> dataclass."""
    if d is None:
        return cls()
    if isinstance(d, cls):
        return d
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise KeyError(f"unknown option(s) for {cls.__name__}: {sorted(unknown)}; known: {sorted(known)}")
    kw = dict(d)
    for name, sub in nested.items():
        if kw.get(name) is not None and not isinstance(kw[name], sub):
            kw[name] = _dc(sub, kw[name])
    return cls(**kw)

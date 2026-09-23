import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from romanmock.lines import ROMAN_TRANSITIONS

CATALOG_DIR = os.environ.get(
    "ROMANMOCK_TEST_CATALOG",
    "/global/cfs/cdirs/m4943/Simulations/sdss_feniks_hizels_260710_09_07_2026",
)


@pytest.fixture(scope="session")
def cosmo():
    return FlatLambdaCDM(H0=67.66, Om0=0.3096446816186967, Ob0=0.04897468161869667)


@pytest.fixture
def synthetic(cosmo):
    """Dict of arrays mimicking the diffsky columns for 20000 galaxies."""
    rng = np.random.default_rng(42)
    n = 20000
    z = rng.uniform(0.3, 3.0, n)
    data = {c: 10 ** rng.normal(42.0, 0.7, n) for c in ROMAN_TRANSITIONS.required_columns}
    data["redshift_true"] = z
    data["redshift_obs"] = z + 1e-3 * rng.standard_normal(n)
    data["gal_id"] = np.arange(n)
    data["ra"] = rng.uniform(0, 90, n)
    data["dec"] = rng.uniform(50, 70, n)
    return data


def catalog_available():
    return os.path.isdir(CATALOG_DIR)

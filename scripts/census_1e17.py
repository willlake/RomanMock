"""Census at 1e-17 (plus 1e-16, 1.5e-16) over 0.5 < z < 3.0: distinct-galaxy counts for cost estimates."""
import os, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
from romanmock.census import line_census
from romanmock.io import suggest_n_workers

CAT = "/global/cfs/cdirs/m4943/Simulations/sdss_feniks_hizels_260710_09_07_2026"
OUT = os.path.join(os.environ["PSCRATCH"], "romanmock_demo", "census_z0p5_3p0_1e17.npz")
if __name__ == "__main__":      # required: workers are spawned and re-import this script
    edges = np.round(np.arange(0.5, 3.0 + 1e-9, 0.1), 2)
    t0 = time.time()
    c = line_census(CAT, [1e-17, 1e-16, 1.5e-16], edges, n_workers=suggest_n_workers(max_workers=32, per_worker_gb=4.0), verbose=False)
    c.save(OUT)
    print(f"{len(c.steps)} slices in {time.time() - t0:.0f} s -> {OUT}")
    print(f"galaxies 0.5<z<3.0: {c.n_all.sum():,}")
    for lim in c.flux_limits:
        g = c.galaxies(lim); s = c.galaxies(lim, "single")
        print(f"F > {lim:.1e}: distinct galaxies with a line {g.sum():>12,}  single {s.sum():>12,}  multi {g.sum() - s.sum():>12,}  "
              f"Ha {c.hist('Ha', lim).sum():>12,}  OIII_5007 {c.hist('OIII_5007', lim).sum():>12,}")

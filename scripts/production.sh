#!/bin/bash
# Production, step 1: base catalog, dN/dz census and line-flux distribution on one CPU node (about 5 minutes).
# Usage (from the package directory, with the Roman-dev environment active):
#   sbatch scripts/production.sh [config.yaml]
#SBATCH -A m4943
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J romanmock
#SBATCH -o logs/production_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
CONFIG=${1:-configs/production.yaml}
export JAX_PLATFORMS=cpu
read -r MOCK ROOT LINES < <(python -c "
import sys, yaml
from pathlib import Path
c = yaml.safe_load(open(sys.argv[1]))
print(c['catalog']['path'], Path(c['base']['dir']).parent, c.get('line_set', 'roman_transitions'))" "$CONFIG")
mkdir -p "$ROOT/census"

romanmock base "$CONFIG" --workers 32 --quiet
romanmock census "$MOCK" -o "$ROOT/census/dndz_census_z0p5_3p0.npz" --flux-limits 1.0e-16 1.5e-16 \
          --secondary-flux-limits 0 8.0e-17 6.0e-17 4.0e-17 --z-range 0.5 3.0 --dz 0.01 --line-set "$LINES" --workers 32 --quiet
romanmock flux-distribution "$MOCK" -o "$ROOT/census/flux_distribution.npz" --z-range 0.0 4.0 --dz 0.01 --line-set "$LINES" --workers 32 --quiet
chmod -R g+rX "$ROOT"

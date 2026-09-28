#!/bin/bash
#SBATCH -A m4943
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH -J romanmock-census
#SBATCH -o logs/census_1e17_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export JAX_PLATFORMS=cpu
python scripts/census_1e17.py

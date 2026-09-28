#!/bin/bash
# Two-step workflow, step 1: build the base flux catalog on one CPU node.
# Usage (from the package directory): sbatch scripts/run_base.sh <config.yaml>
#SBATCH -A m4943
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J romanmock-base
#SBATCH -o logs/base_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
python -c "import romanmock" || { echo "romanmock not importable: activate the Roman-dev env before sbatch"; exit 1; }
export JAX_PLATFORMS=cpu
/usr/bin/time -v romanmock base "$1" --workers 32 --quiet

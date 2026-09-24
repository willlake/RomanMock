#!/bin/bash
# Execute the tutorial notebooks on one CPU node (needed for 05_full_run, which processes the
# whole mock with ~30 parallel workers and 20+ GB in stage B; login nodes cap users at 30 GB).
# Usage (from the package directory): sbatch scripts/execute_notebooks.sh [notebook prefixes, default: 05]
#SBATCH -A m4943
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J romanmock-nb
#SBATCH -o logs/notebooks_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"   # sbatch runs a spooled copy of this script
python -c "import romanmock" || { echo "romanmock not importable: activate the Roman-dev env before sbatch"; exit 1; }
mkdir -p logs
export JAX_PLATFORMS=cpu
python notebooks/build_notebooks.py --execute --timeout 7200 --only "${@:-05}"

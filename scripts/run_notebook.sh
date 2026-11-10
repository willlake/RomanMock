#!/bin/bash
# Execute notebooks on one CPU node, in order, saving their outputs into them. The example notebooks read the
# production base catalog several times and use a whole node (64 workers) when they see SLURM_JOB_ID.
# Usage (from the package directory, with the Roman-dev environment active):
#   sbatch scripts/run_notebook.sh notebooks/01_catalog_z1p0_1p2.ipynb [notebooks/02_... ...]
#SBATCH -A m4943
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH -J notebook
#SBATCH -o logs/notebook_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
echo "node $(hostname), ${SLURM_CPUS_ON_NODE:-?} CPUs, start $(date)"
export JAX_PLATFORMS=cpu          # romanmock imports diffsky, which imports JAX
export OMP_NUM_THREADS=1          # one thread per worker process
python scripts/run_notebook.py "$@"
echo "end $(date)"

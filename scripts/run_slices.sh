#!/bin/bash
# Process a diffsky mock slice-by-slice as a Slurm job array on Perlmutter.
# Usage: sbatch --array=0-66 scripts/run_slices.sh configs/roman_grism_baseline.yaml
#SBATCH -A m4943
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -J romanmock
#SBATCH -o logs/romanmock_%A_%a.out
set -euo pipefail
CONFIG=${1:?config yaml}
export JAX_PLATFORMS=cpu
CATALOG=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['catalog']['path'])" "$CONFIG")
FILES=($(ls "$CATALOG"/lc_cores-*.diffsky_gals.hdf5 | sort -t- -k2 -n))
FILE=${FILES[${SLURM_ARRAY_TASK_ID:-0}]}
echo "task ${SLURM_ARRAY_TASK_ID:-0}: $FILE"
romanmock run "$CONFIG" --files "$FILE"

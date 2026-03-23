#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --output=result.txt

set -e

module purge
module load Python/3.10.8-GCCcore-12.2.0

# Activate venv
source .venv/bin/activate

python python_scripts/01_build_fm_synth_params.py

echo "finished"

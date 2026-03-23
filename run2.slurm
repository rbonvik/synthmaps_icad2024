#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --output=result.txt

set -e

module purge
module load Python/3.10.8-GCCcore-12.2.0

# Activate venv
source .venv/bin/activate

python python_scripts/02_build_perceptual_ds.py

echo "finished"

#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=16G
#SBATCH --output=result3.txt
#SBATCH --error=error3.txt

set -e

module purge
module load Python/3.10.8-GCCcore-12.2.0

# Activate venv
source .venv/bin/activate

python python_scripts/03_build_spectral_ds.py

echo "finished"
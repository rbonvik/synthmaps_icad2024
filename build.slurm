#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --output=result.txt

echo "Starting job on $(hostname)"

# Load Python module
module purge
module load Python/3.10.8-GCCcore-12.2.0

# Create virtual environment
python -m venv .venv

# Activate venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
# Install dependencies
pip install \
  numpy==1.23.5 \
  scipy==1.9.3 \
  pandas==1.5.3 \
  scikit-learn==1.1.3 \
  librosa==0.9.2 \
  numba==0.56.4 \
  timbral_models \
  torch \
  tqdm \
  pytimbre==0.6.10

echo "Build finished"
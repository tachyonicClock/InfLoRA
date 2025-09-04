#!/bin/bash -e
#SBATCH --job-name=domainnet_l2p
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --qos=debug
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=logs/domainnet_l2p.out
#SBATCH --error=logs/domainnet_l2p.err

module load Miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$NESI_PYVENV/inflora"

set -x # Echo commands to stdout
set -e # Exit on error

python main.py --device 0 --config configs/domainnet_l2p.yaml

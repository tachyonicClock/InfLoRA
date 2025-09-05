#!/bin/bash -e
#SBATCH --job-name=domainnet_l2p
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=logs/domainnet_l2p_%a.log
#SBATCH --array=0-0

export PATH=$NESI_PYVENV/inflora/bin:$PATH

set -x # Echo commands to stdout
set -e # Exit on error

python main.py --device 0 --config configs/domainnet_l2p.yaml --seed "$SLURM_ARRAY_TASK_ID"

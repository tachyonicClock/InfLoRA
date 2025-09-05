#!/bin/bash -e
#SBATCH --job-name=imagenetr_l2p
#SBATCH --time=13:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=logs/imagenetr_l2p_%a.log
#SBATCH --array=0-0

export PATH=$NESI_PYVENV/inflora/bin:$PATH

set -x # Echo commands to stdout
set -e # Exit on error

python main.py --device 0 --config configs/imagenetr_l2p.yaml --seed "$SLURM_ARRAY_TASK_ID"

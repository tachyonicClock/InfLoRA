#!/bin/bash -e
#SBATCH --job-name=imagenetr_l2p
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --qos=debug
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=logs/imagenetr_l2p.out
#SBATCH --error=logs/imagenetr_l2p.err

set -x # Echo commands to stdout
set -e # Exit on error

export PATH=$NESI_PYVENV/inflora/bin:$PATH
python main.py --device 0 --config configs/imagenetr_l2p.yaml

#!/bin/bash -e
#SBATCH --job-name=imagenetr_inflora
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --qos=debug
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=logs/imagenetr_inflora.out
#SBATCH --error=logs/imagenetr_inflora.err

set -x # Echo commands to stdout
set -e # Exit on error

export PATH=$NESI_PYVENV/inflora/bin:$PATH
python main.py --device 0 --config configs/imagenetr_inflora.yaml

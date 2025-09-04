#!/bin/bash -e
#SBATCH --job-name=domainnet_inflora
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=logs/domainnet_inflora.log

export PATH=$NESI_PYVENV/inflora/bin:$PATH

set -x # Echo commands to stdout
set -e # Exit on error

python main.py --device 0 --config configs/domainnet_inflora.yaml

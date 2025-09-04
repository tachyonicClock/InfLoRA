#!/bin/bash -e
#SBATCH --job-name=cifar100_l2p
#SBATCH --time=03:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=logs/cifar100_l2p.log

export PATH=$NESI_PYVENV/inflora/bin:$PATH

set -x # Echo commands to stdout
set -e # Exit on error

python main.py --device 0 --config configs/cifar100_l2p.yaml

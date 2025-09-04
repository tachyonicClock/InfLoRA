#!/bin/bash -e
#SBATCH --job-name=cifar100_dualprompt
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --qos=debug
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=logs/cifar100_dualprompt.log

export PATH=$NESI_PYVENV/inflora/bin:$PATH

set -x # Echo commands to stdout
set -e # Exit on error

python main.py --device 0 --config configs/cifar100_dualprompt.yaml

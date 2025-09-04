#!/bin/bash -e
#SBATCH --job-name=cifar100_codap
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --qos=debug
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=logs/cifar100_codap.out
#SBATCH --error=logs/cifar100_codap.err

module load Miniconda3
conda activate "$NESI_PYVENV/inflora"

set -x # Echo commands to stdout
set -e # Exit on error

python main.py --device 0 --config configs/cifar100_codap.yaml

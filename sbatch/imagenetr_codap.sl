#!/bin/bash -e
#SBATCH --job-name=imagenetr_codap
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --qos=debug
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=logs/imagenetr_codap.out
#SBATCH --error=logs/imagenetr_codap.err

module load Miniconda3
export PATH=${HOME}/nobackup/pyvenv/inflora/bin:${PATH}

set -x # Echo commands to stdout
set -e # Exit on error

python main.py --device 0 --config configs/imagenetr_codap.yaml

#!/bin/bash -e
#SBATCH --job-name=domainnet_codap
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --qos=debug
#SBATCH --gpus-per-node=L4:1
#SBATCH --output=logs/domainnet_codap.out
#SBATCH --error=logs/domainnet_codap.err

module load Miniconda3
export PATH=${HOME}/nobackup/pyvenv/inflora/bin:${PATH}

set -x # Echo commands to stdout
set -e # Exit on error

python main.py --device 0 --config configs/domainnet_codap.yaml

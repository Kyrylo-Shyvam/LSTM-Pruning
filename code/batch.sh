#!/bin/bash

#SBATCH -A kyrylo
#SBATCH -n 9
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=0-04:00:00
#SBATCH --mail-type=END
#SBATCH -w gnode067


source ~/cpu1/bin/activate

which python3
echo I started
./scripts/train.sh


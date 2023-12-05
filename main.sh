#!/bin/bash

# FILEPATH: /home/ari/learning/main.sh
module purge
module load miniconda3
source activate /vast/home/ajherman/miniconda3/envs/pytorch
# Allocate 4 nodes using srun and run main.py
srun -N 4 python main.py --version original --filepath original_csv >> log_original.out &&
srun -N 4 python main.py --version alternate --filepath alternate_csv >> log_alternate.out &&

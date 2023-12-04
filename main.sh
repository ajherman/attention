#!/bin/bash

# FILEPATH: /home/ari/learning/main.sh

# Allocate 4 nodes using srun and run main.py
srun -N 1 -c 10 python main.py --version original >> log_original.out &&
srun -N 1 -c 10 python main.py --version alternate >> log_alternate.out &&

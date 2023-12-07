
#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 4
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch

srun -N 1 -n 1 -c 6 -o original.out --open-mode=append ./main_wrapper.sh --version original --filepath original.csv
srun -N 1 -n 1 -c 6 -o alternate.out --open-mode=append ./main_wrapper.sh --version alternate --filepath alternate.csv



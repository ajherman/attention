
#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 2
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch
cores=20
srun -N 1 -n 1 -c $cores -o original.out --open-mode=append ./main_wrapper.sh --version 0 --filepath original.csv &
#srun -N 1 -n 1 -c $cores -o mix.out --open-mode=append ./main_wrapper.sh --version 2 --filepath mix.csv &
#srun -N 1 -n 1 -c $cores -o rms.out --open-mode=append ./main_wrapper.sh --version 3 --filepath rms.csv &




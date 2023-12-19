
#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 2
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch
cores=20

srun -N 1 -n 1 -c $cores -o regular.out --open-mode=append ./main_wrapper.sh --block-type 3 --filepath regular.csv &
srun -N 1 -n 1 -c $cores -o rectified.out --open-mode=append ./main_wrapper.sh --block-type 5 --filepath rectified.csv &

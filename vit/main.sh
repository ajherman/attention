
#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 3
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch
cores=20

name=results

srun -N 1 -n 1 -c $cores -o $name.out --open-mode=append ./main_wrapper.sh & 

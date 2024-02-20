
#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 1
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch
cores=20

name=results

#srun -N 1 -n 1 -c $cores -o $name.out --open-mode=append ./main_wrapper.sh --patch-size 4 --dm 256 --h 8 --N 4 --lr 3e-4 &
srun -N 1 -n 1 -c 20 -o $name.out --open-mode=append ./main_wrapper.sh --post-norm 0 --patch-size 4 --dm 256 --h 8 --N 6 --lr 3e-4 &

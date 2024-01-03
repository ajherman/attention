
#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 2
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch
cores=20

# srun -N 1 -n 1 -c $cores -o regular.out --open-mode=append ./main_wrapper.sh --batch-size 32 --block-type 3 --filepath regular.csv &
srun -N 1 -n 1 -c $cores -o rectified.out --open-mode=append ./main_wrapper.sh --batch-size 32 --block-type 5 --eval-interval 200 --filepath rectified.csv &
# srun -N 1 -n 1 -c $cores -o log.out --open-mode=append ./main_wrapper.sh --batch-size 32 --block-type 6 --filepath log.csv &
# srun -N 1 -n 1 -c $cores -o mine.out --open-mode=append ./main_wrapper.sh --batch-size 32 --block-type 7 --filepath mine.csv &
srun -N 1 -n 1 -c $cores -o base.out --open-mode=append ./main_wrapper.sh --batch-size 32 --block-type 0 --eval-interval 200 --filepath base.csv &

# For TinyStories version
# srun -N 1 -n 1 -c $cores -o base.out --open-mode=append ./main_wrapper.sh --block-type 0 --block-size 100 --dm 384 --h 8 --lr 5e-4 --filepath base.csv #--vocab-size 50258 &
# srun -N 1 -n 1 -c $cores -o base.out --open-mode=append ./main_wrapper.sh --block-type 0 --block-size 256 --dm 768 --N 8 --h 16 --filepath base.csv #--vocab-size 50258 &

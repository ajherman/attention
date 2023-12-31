
#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 3
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch
cores=20

# srun -N 1 -n 1 -c $cores -o regular.out --open-mode=append ./main_wrapper.sh --batch-size 32 --block-type 3 --filepath regular.csv &
# srun -N 1 -n 1 -c $cores -o rectified.out --open-mode=append ./main_wrapper.sh --batch-size 32 --block-type 5 --filepath rectified.csv &
# srun -N 1 -n 1 -c $cores -o log.out --open-mode=append ./main_wrapper.sh --batch-size 32 --block-type 6 --filepath log.csv &
# srun -N 1 -n 1 -c $cores -o mine.out --open-mode=append ./main_wrapper.sh --batch-size 32 --block-type 7 --filepath mine.csv &
#srun -N 1 -n 1 -c $cores -o base.out --open-mode=append ./main_wrapper.sh --batch-size 32 --block-type 0 --filepath base.csv &

# For TinyStories version
#<<<<<<< HEAD
#srun -N 1 -n 1 -c $cores -o base.out --open-mode=append ./main_wrapper.sh --block-type 0 --block-size 128 --eval-interval 50 --batch-size 64 --dm 512 --h 8 --N 6 --lr 5e-4 --dataset stories --filepath base.csv & #--vocab-size 50258 &
#srun -N 1 -n 1 -c $cores -o base.out --open-mode=append ./main_wrapper.sh --block-type 3 --block-size 128 --eval-interval 50 --batch-size 64 --dm 512 --h 8 --N 6 --lr 5e-4 --dataset stories --filepath base.csv & #--vocab-size 50258 &
#srun -N 1 -n 1 -c $cores -o base.out --open-mode=append ./main_wrapper.sh --block-type 5 --block-size 128 --eval-interval 50 --batch-size 64 --dm 512 --h 8 --N 6 --lr 5e-4 --dataset stories --filepath base.csv & #--vocab-size 50258 &
#=======
srun -N 1 -n 1 -c $cores -o base.out --open-mode=append ./main_wrapper.sh --block-type 0 --block-size 128 --eval-interval 50 --batch-size 64 --dm 512 --h 8 --N 6 --lr 5e-4 --dataset stories --filepath base.csv & #--vocab-size 50258 &
srun -N 1 -n 1 -c $cores -o rms.out --open-mode=append ./main_wrapper.sh --block-type 3 --block-size 128 --eval-interval 50 --batch-size 64 --dm 512 --h 8 --N 6 --lr 5e-4 --dataset stories --filepath rms.csv & #--vocab-size 50258 &
srun -N 1 -n 1 -c $cores -o rectified.out --open-mode=append ./main_wrapper.sh --block-type 5 --block-size 128 --eval-interval 50 --batch-size 64 --dm 512 --h 8 --N 6 --lr 5e-4 --dataset stories --filepath rectified.csv & #--vocab-size 50258 &
#>>>>>>> db8e2a946b083589f5bf037b33a90139ddaa25d9

#srun -N 1 -n 1 -c $cores -o base.out --open-mode=append ./main_wrapper.sh --block-type 0 --block-size 100 --dm 384 --h 8 --lr 5e-4 --filepath base.csv #--vocab-size 50258 &

# srun -N 1 -n 1 -c $cores -o base.out --open-mode=append ./main_wrapper.sh --block-type 0 --block-size 256 --dm 768 --N 8 --h 16 --filepath base.csv #--vocab-size 50258 &

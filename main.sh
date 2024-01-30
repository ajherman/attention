
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
# srun -N 1 -n 1 -c $cores -o base.out --open-mode=append ./main_wrapper.sh --block-type 0 --block-size 128 --eval-interval 50 --batch-size 64 --dm 512 --h 8 --N 6 --lr 5e-4 --dataset stories --filepath base.csv & #--vocab-size 50258 &
# srun -N 1 -n 1 -c $cores -o rms.out --open-mode=append ./main_wrapper.sh --block-type 3 --block-size 128 --eval-interval 50 --batch-size 64 --dm 512 --h 8 --N 6 --lr 5e-4 --dataset stories --filepath rms.csv & #--vocab-size 50258 &
# srun -N 1 -n 1 -c $cores -o rectified.out --open-mode=append ./main_wrapper.sh --block-type 5 --block-size 128 --eval-interval 50 --batch-size 64 --dm 512 --h 8 --N 6 --lr 5e-4 --dataset stories --filepath rectified.csv & #--vocab-size 50258 &
# srun -N 1 -n 1 -c $cores -o new.out --open-mode=append ./main_wrapper.sh --block-type 8 --block-size 128 --eval-interval 50 --batch-size 64 --dm 512 --h 8 --N 6 --lr 5e-4 --dataset stories --filepath new.csv & #--vocab-size 50258 &




# srun -N 1 -n 1 -c $cores -o base.out --open-mode=append ./main_wrapper.sh --block-size 128 --eval-interval 50 --filepath base.csv & #--vocab-size 50258 &
# srun -N 1 -n 1 -c $cores -o rms.out --open-mode=append ./main_wrapper.sh --block-size 128 --eval-interval 50 --norm-type rms --filepath rms.csv & #--vocab-size 50258 &
# srun -N 1 -n 1 -c $cores -o rectified.out --open-mode=append ./main_wrapper.sh --block-size 128 --eval-interval 50 --norm-type rms --rectify True --filepath rectified.csv & #--vocab-size 50258 &

version=0

for norm_type in {layer,rms}
do
for rectify in {0,1}
do
for post_norm in {0,1}
do

name="norm_${norm_type}_rectify_${rectify}_post_norm_${post_norm}"
srun -N 1 -n 1 -c $cores -o $name.out --open-mode=append ./main_wrapper.sh --block-size 200  --version $version --eval-interval 50 --norm-type $norm_type --rectify $rectify --post-norm $post_norm --dataset stories --filepath $name.csv & 
version=$((version+1))
done
done
done


#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 3
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch
cores=20

############################################################

# for dataset in {stories,}
# do

# name="test_${dataset}"

# # srun -N 1 -n 1 -c $cores -o $name.out --open-mode=append ./main_wrapper.sh --block-size 200 --eval-interval 50 --dataset $dataset --stream-data --filepath $name.csv & 

# python main.py --batch-size 2 --block-size 200 --eval-interval 50 --dataset $dataset --dm 256 --dk 64 --dv 64 --h 4 --N 3 --n-fixed-keys 0 --filepath $name.csv > $name.out 2>&1 

# done


# version=0

# for norm_type in {layer,rms}
# do
# for rectify in {0,1}
# do
# for block_architecture in {series,parallel}
# do

# name="norm_${norm_type}_rectify_${rectify}_arch_${block_architecture}"
# srun -N 1 -n 1 -c $cores -o $name.out --open-mode=append ./main_wrapper.sh --block-size 128  --version $version --eval-interval 50 --norm-type $norm_type --rectify $rectify --block-architecture $block_architecture --dataset stories --filepath $name.csv & 
# version=$((version+1))

# done
# done
# done


norm_type=layer
rectify=0
block_architecture=series
version=0
for n_fixed_keys in {0,8,16}
do

name="test_fixed_keys_${n_fixed_keys}"
srun -N 1 -n 1 -c $cores -o $name.out --open-mode=append ./main_wrapper.sh --block-size 128  --version $version --eval-interval 50 --norm-type $norm_type --rectify $rectify --block-architecture $block_architecture --n-fixed-keys $n_fixed_keys --dataset stories --filepath $name.csv & 
version=$((version+1))

done

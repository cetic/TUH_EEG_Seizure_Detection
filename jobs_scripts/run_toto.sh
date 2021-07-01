#!/bin/bash
#SBATCH --job-name=multigpu_cnn
#SBATCH --partition=gpu 
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=5G
#SBATCH --gres="gpu:2"
#SBATCH -t 1:00:00
#SBATCH --mail-user=id@umons.ac.be
#SBATCH --mail-type=ALL 

# Loading Python 3.8.6rc1
module use "$HOME"/modulefiles/
module load python/3.8.6rc1

# Loading an Anaconda environment
# conda source tensorflow-gpu-1.8

echo "DATE : $(date)"
echo "_____________________________________________"
echo " HOSTNAME             : $HOSTNAME"
echo "_____________________________________________"
echo " CUDA_DEVICE_ORDER    : $CUDA_DEVICE_ORDER"
echo "_____________________________________________"
echo " CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "_____________________________________________"
nvidia-smi -L
echo "_____________________________________________"

# Starting the Python program and printing the time it took to complete
time python -V

du -sh ~/TUH_SZ_v1.5.2/TUH/
du -sh ~/CHBMIT/

free -h

# cat /proc/cpuinfo
lscpu

# srun --partition debug -n 2 --mem 2G --gres="gpu:3" --pty bash ~/TUH_SZ_v1.5.2/run_toto.bash
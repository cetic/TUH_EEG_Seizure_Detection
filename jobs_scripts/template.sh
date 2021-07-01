#!/bin/bash
#SBATCH --job-name={0}
#SBATCH --partition={1}
#SBATCH --ntasks={2}
#SBATCH --cpus-per-task={3}
#SBATCH --mem={4}
#SBATCH -t {5}
#SBATCH --mail-user={6}
#SBATCH --mail-type={7}
{8}

# Loading Python 3.8.6rc1
module purge
module use "$HOME"/modulefiles/
module load python/3.8.6rc1

{9}

wait

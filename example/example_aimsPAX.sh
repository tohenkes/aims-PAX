#!/bin/bash -l
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --time=06:00:00                    # time (HH:MM:SS)
#SBATCH --partition=gpu               # partition
#SBATCH --account=p200243                  # project account
#SBATCH --qos=default        # SLURM qos


module load intel
source path/to/your/conda/bin/activate my_env

aims-PAX-al

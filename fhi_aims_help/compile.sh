#!/bin/bash -l
#SBATCH --time=06:00:00
#SBATCH --account=XXX
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128

# these modules also need be loaded (step 5 in README.md)
module load env/release/2023.1
module load imkl
module load CMake
module load intel

make -j 128 >> compile.out

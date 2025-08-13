#!/bin/bash -l
#SBATCH --time=01:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p batch
#SBATCH -J aims_pax_test

source /home/users/thenkes/miniconda3/bin/activate aimspax

aims-PAX

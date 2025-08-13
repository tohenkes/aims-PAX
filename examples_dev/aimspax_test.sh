#!/bin/bash -l
#SBATCH --time=01:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH -p batch
#SBATCH -J aims_pax_test

source /home/users/thenkes/miniconda3/bin/activate aimspax
module load mpi/impi
module load numlib/imkl
ulimit -s unlimited
export OMP_NUM_THREADS=1
module load toolchain/intel


srun aims_PAX-al

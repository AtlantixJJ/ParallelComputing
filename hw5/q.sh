#!/bin/sh

#SBATCH -o ./myjob.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
mpiexec -n 4 ./prog3.5_mpi_mat_vect_col
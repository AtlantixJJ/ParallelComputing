#!/bin/bash

# set the job name to hello-mpi
#SBATCH --job-name=test

# send output to hello-mpi.out
#SBATCH --output=log/test.out

# this job requests 2 nodes
#SBATCH --nodes=2

# Run the process with mpirun. Notice -n is not required. mpirun will
# automatically figure out how many processes to run from the slurm options
mpirun build/test
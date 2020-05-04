#!/bin/bash
#SBATCH -p crill
#SBATCH -N 4
#SBATCH --job-name=mpids_unit
#SBATCH --ntasks-per-node 1
#SBATCH -t 00:03:00
#SBATCH -o mpids_unit.out

module load mpi4py;

paver clean;
#Run tests for MPInumpy and MPIscipy using 4 MPI processes
mpiexec -n 4 paver test_mpids_numpy_scipy > mpids_crill_unit.out 2>&1;
#Run tests for MPIpandas 2 MPI processes
mpiexec -n 2 paver test_mpipandas >> mpids_crill_unit.out 2>&1;

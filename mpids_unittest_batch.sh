#!/bin/bash
#SBATCH -p crill
#SBATCH -N 4
#SBATCH --job-name=mpids_unit
#SBATCH --ntasks-per-node 1
#SBATCH -t 00:03:00
#SBATCH -o mpids_unit.out

module load mpi4py;

paver clean;
mpiexec -n 4 paver test_mpinumpy > mpids_crill_unit.out 2>&1;

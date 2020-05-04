#!/bin/bash

paver clean;
#Run tests for MPInumpy and MPIscipy using 4 MPI processes
mpiexec -n 4 paver test_mpids_numpy_scipy > mpids_local_unit.out 2>&1;
#Run tests for MPIpandas 2 MPI processes
mpiexec -n 2 paver test_mpipandas >> mpids_local_unit.out 2>&1;

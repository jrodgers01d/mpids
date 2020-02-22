#!/bin/bash

paver clean;
mpiexec -n 4 paver test_mpids_numpy_scipy > mpids_local_unit.out 2>&1;

#!/bin/bash

paver clean;
mpiexec -n 4 paver test_mpinumpy > mpids_local_unit.out 2>&1;

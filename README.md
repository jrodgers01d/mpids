# This repository hosts a collection of MPI Data Science Modules

## Collection currently includes
1. MPIcollections
2. MPInumpy
3. MPIpandas
4. MPIscipy

## Software Requirements
Currently only Python3 fully supported

### For usage
- mpi4py (https://mpi4py.readthedocs.io/en/stable/)
- NumPy (https://numpy.org/)
  -- version 1.10+ for full MPInumpy usage
- nltk (https://www.nltk.org/)
- pandas (https://pandas.pydata.org/)
- petsc4py (https://pypi.org/project/petsc4py/)

### For tests
- coverage (https://coverage.readthedocs.io/en/coverage-5.1/)
- nose (https://nose.readthedocs.io/en/latest/)
- paver (https://github.com/paver/paver)

## Running tests

###Locally

> /bin/bash mpids_unittest.sh #Results saved in mpids_local_unit.out

###On a cluster

> sbatch mpids_unittest_batch.sh #Results saved in mpids_crill_unit.out

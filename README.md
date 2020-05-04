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
  - Requires working MPI implementation
  - version 3.0.0
- NumPy (https://numpy.org/)
  - version 1.10+ for full MPInumpy usage
- nltk (https://www.nltk.org/)
  - version 3.3+
- pandas (https://pandas.pydata.org/)
  - version 0.16+
- petsc4py (https://pypi.org/project/petsc4py/)
  - Requires working MPI implementation
  - version 3.12+

### For tests
- coverage (https://coverage.readthedocs.io/en/coverage-5.1/)
  - version 4.5+
- nose (https://nose.readthedocs.io/en/latest/)
  - version 1.3+
- paver (https://github.com/paver/paver)
  - version 1.3+

## Running tests

### Locally
Results saved in `mpids_local_unit.out`
> /bin/bash mpids_unittest.sh

### On a cluster
Results saved in `mpids_crill_unit.out`

Note: Script configured for UH PSTL Crill Cluster
> sbatch mpids_unittest_batch.sh

### Cleaning up
> paver clean

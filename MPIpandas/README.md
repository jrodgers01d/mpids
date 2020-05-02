#This repository contains the Parallel-Pandas library

**Pre-req: The examples are configured to be run from the mpids folder**

==============================================================
#HOW TO RUN DOCUMENT DUPLICATE DETECTION
==============================================================
##to run duplicate detection using the MPIpandas library
______________________________________________________________
mpiexec -n {processors} python3 ./MPIpandas/examples/document_duplicate_detection/driver_dup_detection_MPIpandas.py {data_folder} {threshhold} {stats_file} {results_file}

##to run serial duplicate detection
______________________________________________________________
python3 ./MPIpandas/examples/document_duplicate_detection/driver_dup_detection_serial_pandas.py {data_folder} {threshhold} {stats_file} {results_file}

##to run duplicate detection optimized by using Pandas and MPI
______________________________________________________________
mpiexec -n {processors} python3 ./MPIpandas/examples/document_duplicate_detection/driver_dup_detection_pandas_and_mpi.py {data_folder} {threshhold} {stats_file} {results_file}

==============================================================
#HOW TO RUN MICROBENCHMARKS
==============================================================
###Run the pre script (unzips the data)
./MPIpandas/examples/microbenchmarks/pre_microbenchmarks

###Run microbenchmarks
mpiexec -n {processors} python3 ./MPIpandas/examples/microbenchmarks/microbenchmarks.py --stats_file {stats_file}

###Run serial microbenchmarks
python3 ./MPIpandas/examples/microbenchmarks/microbenchmarks_serial.py --stats_file {stats_file}

###Run the post script (deletes the un-needed data)
./MPIpandas/examples/microbenchmarks/post_microbenchmarks


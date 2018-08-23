from mpi4py import MPI
import sys
import re
import os

import mpids

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if len (sys.argv) < 4:
        print("Usage: wordcount_mpi4py.py <input-directoryfile> <outputfile> <book directory>")
        comm.Abort(1)

    inputfile  = sys.argv[1]
    outputfile = sys.argv[2]
    bookpath   = sys.argv[3]

    starttime = MPI.Wtime()    
    text = mpids.ParallelIO.read_all (inputfile, bookpath )
    endreadtime = MPI.Wtime() - starttime

    starttokenizetime = MPI.Wtime()
    lower = re.sub('[^A-Za-z0-9]+', ' ', text.lower())
    tokens = re.sub(r',\s*(?![^()]*\))', '\n', lower).split()
    endtokenizetime = MPI.Wtime() - starttokenizetime

    comm.Barrier() 
    startprocesstime = MPI.Wtime()
    wcount = mpids.MPICounter.Counter_all ( tokens)
    endprocesstime = MPI.Wtime() - startprocesstime

    comm.Barrier() 
    startfilewritetime = MPI.Wtime()
    mpids.ParallelIO.write_all (wcount, outputfile)
    endtime = MPI.Wtime()

    endfilewritetime = endtime - startfilewritetime
    totaltime = endtime - starttime

    comm.reduce(totaltime, op=MPI.MAX, root=0)
    comm.reduce(endreadtime, op=MPI.MAX, root=0)
    comm.reduce(endtokenizetime, op=MPI.MAX, root=0)
    comm.reduce(endprocesstime, op=MPI.MAX, root=0)
    comm.reduce(endfilewritetime, op=MPI.MAX, root=0)
    if rank == 0:
        print ("Total time = %f seconds" % totaltime)
        print ("Read time = %f seconds" % endreadtime)
        print ("Tokenize time = %f seconds" % endtokenizetime)
        print ("Processing time = %f seconds" % endprocesstime)
        print ("File write time = %f seconds" % endfilewritetime)
        

from mpi4py import MPI
import string
import sys
import re
import os
import numpy as np

def __file_info(inputpath, comm):
        """
        Helper routine to return the number of files,
        filenames, and a set of indices assigned to each
        process.

        inputpath is assumed to be a directory name ending with a
        '/' at the end.
        """

        rank = comm.Get_rank()
        size = comm.Get_size()
    
        nr_of_files = 0
        file_list=[]
        filenames=[]
        
        if rank == 0:
                files = os.listdir(inputpath)    
                for file in files:
                        s = os.path.getsize(inputpath+file)
                        file_list.append([file, nr_of_files, s])
                        nr_of_files +=1
                filenames=sorted(file_list, key=lambda t: t[2], reverse=True)

        filenames=comm.bcast(filenames, root=0)
        nr_of_files=len(filenames)

        indices=[]
        for i in range (rank, nr_of_files, size):
            indices.append(i)

        return nr_of_files, filenames, indices

def get_num_files(inputfile, comm=MPI.COMM_WORLD):
        """
        Returns the number of files available in the 
        provided directory, or 1 if the input is 
        already a file.
        """

        rank = comm.Get_rank()
        if rank == 0:
                if os.path.isdir(inputfile):
                        files = os.listdir(inputfile)
                        num_files = len(files)
                else:
                        num_files = 1
        nfiles = np.asarray(num_files)
        comm.Bcast(nfiles, root=0)
        return int(nfiles)

def get_file_size(inputfile, comm=MPI.COMM_WORLD):
        """
        Returns the size of a file or the sum of all file sizes
        if the input is a directory.
        """
        total_size = int(0)
        if rank == 0:
                if os.path.isdir(inputfile):
                        if not inputfile.endswith("/"):
                                inputfile = inputfile+"/"

                        files = os.listdir(inputfile)
                        for file in files:
                                s = os.path.getsize(inputpath+file)
                                total_size += s
                else:
                        total_size = os.path.getsize(inputpath)
        nsize = np.asarray(total_size)
        comm.Bcast(nsize, root=0)
        return int(nsize)

def read_all (inputfile, comm=MPI.COMM_WORLD, tracing=False):
        if os.path.isdir(inputfile):
                if not inputfile.endswith("/"):
                        inputfile = inputfile+"/"

                no_of_files, filenames, indices = __file_info(inputfile, comm)
                fulltext = []
                for i in indices:
                        with open (inputfile+filenames[i][0],"rb")  as file:
                                text = file.read()
                                fulltext.append(text)
                return str(fulltext)
        else:
                print("Splitting a single file is currently not yet supported")
                return NONE

def write_all (buf, outputfile, comm=MPI.COMM_WORLD, \
               mode=MPI.MODE_CREATE|MPI.MODE_WRONLY, write_elem_per_iter=20000, \
               tracing=False, pretty_print=False):

        rank = comm.Get_rank()
        size = comm.Get_size()
        if tracing and rank == 0:
            print("Preparing for write")

        fh = MPI.File.Open(comm, outputfile, mode)

        buf_list = []
        for key,value in buf.items():
            temp = str(key) + ": " + str(value)
            buf_list.append(temp)

        max_elem = len ( buf_list)
        #s, e = __get_index(float(712), size, rank)
        #print(rank, max_elem, __startletters[s], __startletters[e-1])
        ne    = np.asarray(max_elem) 
        nemax = np.asarray(max_elem)
        comm.Allreduce ( ne, nemax, op=MPI.MAX)

        num_write_iterations = nemax // write_elem_per_iter
        if ( nemax % write_elem_per_iter):
            num_write_iterations = num_write_iterations + 1

        if tracing and rank == 0:
            print("Writing in ", num_write_iterations, " iterations")

        for write_iter in range (0, num_write_iterations):    
            first_elem = write_iter * write_elem_per_iter

            if first_elem < max_elem:
                last_elem = (write_iter+1) * write_elem_per_iter    
                if  last_elem > max_elem:
                    last_elem = max_elem

                line = str(buf_list[first_elem:last_elem])
                if pretty_print:
                        line = re.sub(r',\s*(?![^()]*\))', '\n', line)
                        line = re.sub(r'[\[\]\{\}\(\),\']', '', line)
                line = line + "\n"
            else:
                line =""

            comm.Barrier() 
            if tracing and rank == 0:
                print(write_iter, "  File write")
            
            bytebuff = line.encode('utf-8')
            fh.Write_ordered(bytebuff)

        fh.Close()




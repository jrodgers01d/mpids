from mpi4py import MPI
import string
import sys
import re
import os
import numpy as np

def __file_len(inputfile, bookpath, comm):
        booknames=[]
        indices=[]

        rank = comm.Get_rank()
        size = comm.Get_size()
    
        nr_of_books = 0
        bd=[]
        
        if rank == 0:
            for line in open(inputfile, 'r').readlines():
                book=re.sub('\n','',line)
                s = os.path.getsize(bookpath+book)
                bd.append([book, nr_of_books, s])
                nr_of_books +=1
        booknames=sorted(bd, key=lambda t: t[2], reverse=True)

        booknames=comm.bcast(booknames, root=0)
        nr_of_books=comm.bcast(nr_of_books, root=0)
        for i in range (rank, nr_of_books, size):
            indices.append(i)

        return nr_of_books, booknames, indices

def read_all (inputfile, comm=MPI.COMM_WORLD, tracing=False):

        if os.path.isdir(inputfile):
                no_of_files, booknames, indices = __file_len(inputfile, comm)

                fulltext = []
                for i in indices:
                        with open (bookpath+booknames[i][0],"rb")  as book:
                                tokens = []
                                text = book.read()
                                fulltext.append(text)
                return str(fulltext)
        else:

def write_all (buf, outputfile, comm=MPI.COMM_WORLD, \
               mode=MPI.MODE_CREATE|MPI.MODE_WRONLY, write_elem_per_iter=20000, \
               tracing=False, pretty_print=False):

        rank = comm.Get_rank()
        size = comm.Get_size()
        if tracing and rank == 0:
            print("Preparing for write")

        fh = MPI.File.Open(comm, outputfile, mode)

        buf_list = []
        for key,value in buf.iteritems():
            temp = str(key) + ": " + str(value)
            buf_list.append(temp)

        max_elem = len ( buf_list)
        #s, e = __get_index(float(712), size, rank)
        #print(rank, max_elem, __startletters[s], __startletters[e-1])
        ne    = np.asarray(max_elem) 
        nemax = np.asarray(max_elem)
        comm.Allreduce ( ne, nemax, op=MPI.MAX)

        num_write_iterations = nemax / write_elem_per_iter
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

            fh.Write_ordered("%s" % line)
        fh.Close()




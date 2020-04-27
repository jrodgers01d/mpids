from mpi4py import MPI
import string
import sys
import re
import os
import numpy as np
from collections import Counter, OrderedDict, defaultdict
from itertools import groupby


__lastrank = -1
__comm     = MPI.COMM_NULL
__startletters =[]

def __init_bib ():
        global __startletters

        for digit in range (ord('0'), ord('9') + 1 ):
            __startletters.append(chr(digit))
        
        for first in range(ord('a'), ord('z') + 1):
            __startletters.append(chr(first))
            for second in range(ord('a'), ord('z') + 1):
                __startletters.append(chr(first) + chr(second))

def __get_index(dict_len, procs, rank):
        """ return start and end index for each process"""
        num = int(dict_len/procs)
        rem = int(dict_len%procs)
        if rank < rem:
            my_len = num+1
            start_idx=rank*my_len
        else:
            my_len=num
            start_idx=rem*(num+1) + (rank-rem)*num

        end_idx = start_idx+my_len
        return start_idx, end_idx


def __groupfunction( item):
        global __comm, __lastrank 
        size = __comm.Get_size()
            

        for i in range ( __lastrank, size ):
                s, e = __get_index(float(712), size, i)
                for j in range ( s, e):
                    itemstring = str(item[0])
                    if itemstring.startswith(tuple('([{')):
                        itemstring = itemstring[2:]

                    if  len(itemstring) == 1:
                        starting = itemstring
                    elif itemstring.startswith(tuple('0123456789')):
                        starting = itemstring[0]
                    else:
                        starting = itemstring[:2]
                    if starting == __startletters[j]:
                        __lastrank=i
                        return i
        return i



def Counter_all (tokens, comm=MPI.COMM_WORLD, tokens_per_iter=100000, tracing=False):

            __init_bib()
            global __comm, __lastrank
            __comm = comm
            size = comm.Get_size()
            rank = comm.Get_rank()

            length = len(tokens)
            nm = np.asarray(length)
            nm_max = np.asarray (int(0))
            comm.Allreduce (nm, nm_max, op=MPI.MAX)

            num_iterations = nm_max // (tokens_per_iter)
            if ( nm_max % tokens_per_iter ):
                num_iterations = num_iterations + 1

            if tracing and rank == 0: 
                print(rank, "Processing documents in ", num_iterations," iterations")

            lastindex = 0  
            final_wcount={}

            for iter  in range (0, num_iterations):  
                comm.Barrier() 
                if tracing and rank == 0:
                    print("  Processing iteration", iter)

                firstindex = lastindex
                if firstindex < length:
                    lastindex = (iter+1) * tokens_per_iter
                    if lastindex > length:
                            lastindex = length
                
                partial_tokens = tokens[firstindex:lastindex]
                word_count = Counter(partial_tokens)

                od = OrderedDict(sorted(word_count.items(), key=lambda t: t[0]))
                __lastrank = 0
                odd = sorted(od.items(), key=__groupfunction)
    
                wordcount_per_reducer=defaultdict(list)
                __lastrank=0
                for ranks, words in groupby(odd, lambda s: __groupfunction(s)):
                    wordcount_per_reducer[ranks]={ranks: list(words)}

                localwordcount={}
                for step in range(0, size ):
                    found = 0
                    sendto = ( rank + step ) % size
                    recvfrom = ( rank + size - step) % size
                        
                    sreq = comm.isend (wordcount_per_reducer[sendto], dest=sendto, tag=478)
                    localwordcount[recvfrom] = comm.recv ( source=recvfrom, tag=478 )
                    sreq.wait()
                    
                for key,value  in localwordcount.items():
                    if value: 
                        for k, v in value.items():
                            for j in v:
                                if j[0] not in final_wcount:
                                    final_wcount[j[0]] = int(j[1])
                                else:
                                    final_wcount[j[0]] += int(j[1])


            return final_wcount


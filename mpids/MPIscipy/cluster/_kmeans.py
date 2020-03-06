from mpi4py import MPI
import numpy as np
import mpids.MPInumpy as mpi_np
from mpids.MPInumpy.distributions.Block import Block
from mpids.MPInumpy.distributions.Replicated import Replicated
from mpids.MPInumpy.MPIArray import MPIArray
from mpids.MPIscipy.errors import TypeError, ValueError


def kmeans(observations, k, thresh=1e-5, comm=MPI.COMM_WORLD):
    """ Distributed K-Means classification of a set of observations into
    user specified number of clusters k.

    Parameters
    ----------
    observations : array_like, ndarray or MPIArray
        1/2-Dimensional vector/matrix of observations.
        Format:
            For a vector/matrix of arbirary size MxN
            obs_i = [feature_0, feature_1, ..., feature_N]
            observations = [obs_0, obs_1, obs_2, ..., obs_M]
    k : int, ndarray or MPIArray
        The number cluster/centroids to generate from set of observations or
        initial seed/guess for centroids.
        Format:
            k = number of clusters
                    or
            k = Number of Centroids X Number of Features
            k = [[feature_0, feature_1, ..., feature_N],
                 [feature_0, feature_1, ..., feature_N],
                 ... ,
                 [feature_0, feature_1, ..., feature_N]]
    thresh : float, optional
        Centroid convergence threshold; clustering algorithm will execute
        until iteration to iteration position change of centroids is below
        specified threshold.  If none specified defaults to 1e-5.
    comm : MPI Communicator, optional
        MPI process communication object.  If none specified
        defaults to MPI.COMM_WORLD

    Returns
    -------
    centroids : Replicated MPIArray
        Array of cluster centroids generated from provided set of observations.
        Format:
            centroids[k] = [feature_0, feature_1, ..., feature_N]
    labels : Replicated MPIArray
        Array of centroid indexes that classify a given observation to its
        closest cluster centroid.
        Format:
            labels[i] = index 'k' of closest centroid for obseverations[i]
    """
    #Ensure observations are distributed and generate labels
    observations, num_features, labels  = _process_observations(observations,
                                                                comm)
    #Buffers for cluster centers
    centroids, num_centroids, temp_centroids = _process_centroids(k,
                                                                  num_features,
                                                                  observations,
                                                                  comm)
    error = np.array(np.inf)
    num_local_obs = observations.shape[0]
    #Counts number of points belonging to cluster(weights)
    counts = np.zeros(num_centroids, dtype=np.int64)

    while True:
        old_error = np.copy(error)
        error.fill(0)

        #Identify closest cluster to each point
        for i in range(num_local_obs):
            min_distance = np.inf
            for j in range(num_centroids):
                distance = np.linalg.norm(observations.local[i] - centroids[j])
                if distance < min_distance:
                    labels.local[i] = j
                    min_distance = distance
            obs_assigned_cluster = int(labels.local[i])
            #Update size and temp centroids of destination cluster
            temp_centroids[obs_assigned_cluster] += observations.local[i]
            counts[obs_assigned_cluster] += 1
            #Update standard error
            error += min_distance

        comm.Allreduce(MPI.IN_PLACE, temp_centroids, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, counts, op=MPI.SUM)
        req_error = comm.Iallreduce(MPI.IN_PLACE, error, op=MPI.SUM)

        #Update all centroids
        for j in range(num_centroids):
            centroids[j] = \
                temp_centroids[j] / counts[j] if counts[j] else temp_centroids[j]

        req_error.Wait()
        # Continue until centroid changes reach threshold
        if np.abs(error - old_error) < thresh:
            break
        #Reset previous counts/temp temp_centroids
        counts.fill(0)
        temp_centroids.fill(0)

    return centroids, labels.collect_data()


def _process_centroids(k, num_features, observations, comm):
    """ Helper method to distribute provided k if necessary and resolve whether
        or not the input is seeded.

    Returns
    -------
    centroids : Replicated MPIArray
        Array of cluster centroids generated from provided set of observations.
    num_centroids : int
        Number of centroids.
    temp_centroids : Replicated MPIArray
        Intermediate centroid locations prior to computing distributed result.
    """
    def __unsupported_type(*args):
        raise TypeError('only number of clusters(int) or ' + \
        'centroid seeds(ndarray or MPIArray) should be k.')

    __process_centroid_map = {int           : __centroids_from_int,
                              np.ndarray    : __centroids_from_ndarray,
                              Block         : __centroids_from_mpinp_block,
                              Replicated : __centroids_from_mpinp_undist}
    centroids = \
        __process_centroid_map.get(type(k), __unsupported_type)(k,
                                                                num_features,
                                                                observations,
                                                                comm)

    num_centroids = centroids.shape[0]
    if num_features != centroids.shape[-1] and centroids.ndim != 1:
        raise ValueError('expected {} '.format(num_features) + \
                         'number of features in seeded cluster centroids.')
    temp_centroids = mpi_np.zeros((num_centroids, num_features),
                                  dtype=observations.dtype,
                                  comm=comm,
                                  dist='r')

    return centroids, num_centroids, temp_centroids


def __centroids_from_int(k, num_features, observations, comm):
    centroids = mpi_np.zeros((k, num_features),
                             dtype=observations.dtype,
                             comm=comm,
                             dist='r')
    #Pick initial centroids
    num_observations = observations.globalshape[0]
    for j in range(k):
        i = j * (num_observations // k)
        centroids[j] = observations[i]

    return centroids


def __centroids_from_ndarray(k, num_features, observations, comm):
    #Duplicate ndarray on all processes
    return mpi_np.array(k, dtype=observations.dtype, comm=comm, dist='r')


def __centroids_from_mpinp_block(k, num_features, observations, comm):
    #Collect replicated copy of data
    replicated_k = k.collect_data()
    return replicated_k.astype(observations.dtype)


def __centroids_from_mpinp_undist(k, num_features, observations, comm):
    #Already in correct format
    return k.astype(observations.dtype)


def _process_observations(observations, comm):
    """ Helper method to distribute provided observations if necessary.

    Returns
    -------
    observations : Block Distributed MPIArray
        Array of cluster centroids generated from provided set of observations.
        Format
    num_features : int
        Number of features in observation vector.
    labels : Block Distributed MPIArray
        Array of centroid indexes that classify a given observation to its
        closest cluster centroid.
    """
    if not isinstance(observations, Block):
        observations = mpi_np.array(observations, comm=comm, dist='b')

    if observations.globalndim > 2:
        raise ValueError('only 1/2-Dimensional observation' +
                         'vector/matrices supported.')

    num_observations = observations.globalshape[0]
    num_features = \
        observations.globalshape[1] if observations.globalndim == 2 else 1


    labels = mpi_np.zeros(num_observations,
                          dtype=np.int64,
                          comm=comm,
                          dist=observations.dist)

    return observations, num_features, labels

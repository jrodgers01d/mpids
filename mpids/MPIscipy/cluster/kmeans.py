from mpi4py import MPI
import numpy as np
import mpids.MPInumpy as mpi_np
from mpids.MPInumpy.distributions.Block import Block

#TODO: add logic to handle seeded centroid values
def kmeans(observations, k, thresh=1e-5):
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
    k : int or ndarray
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

    Returns
    -------
    centroids : Undistributed MPIArray
        Array of cluster centroids generated from provided set of observations.
        Format:
            centroids[k] = [feature_0, feature_1, ..., feature_N]
    labels : Undistributed MPIArray
        Array of centroid indexes that classify a given observation to its
        closest cluster centroid.
        Format:
            labels[i] = index 'k' of closest centroid for obseverations[i]
    """
    observations = _process_observations(observations)
    comm = observations.comm
    num_observations = observations.globalshape[0]
    if observations.globalndim > 1:
        num_features = observations.globalshape[1]
    else:
        num_features = 1

    local_observations = observations.shape[0]
    error = np.array(np.inf)
    #Buffer for cluster centers
    centroids = \
        mpi_np.zeros((k, num_features), dtype=np.float64, comm=comm, dist='u')
    #Temp buffer for cluster centers
    temp_centroids = \
        mpi_np.zeros((k, num_features), dtype=np.float64, comm=comm, dist='u')
    #Counts number of points belonging to cluster
    counts = np.zeros(k, dtype=np.int64)
    #One label for each observation
    labels = mpi_np.zeros(num_observations,
                          dtype=np.int64,
                          comm=comm,
                          dist=observations.dist)

    #Pick initial centroids
    for j in range(k):
        i = j * (num_observations // k)
        centroids[j] = observations[i]

    while True:
        old_error = np.copy(error)
        error.fill(0)

        #Identify closest cluster to each point
        for i in range(local_observations):
            min_distance = np.inf
            for j in range(k):
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
        for j in range(k):
            centroids[j] = \
                temp_centroids[j] / counts[j] if counts[j] else temp_centroids[k]

        req_error.Wait()
        # Continue until centroid changes reach threshold
        if np.abs(error - old_error) < thresh:
            break
        #Reset previous counts/temp temp_centroids
        counts.fill(0)
        temp_centroids.fill(0)

    return centroids, labels[:]


def _process_observations(observations):
    """ Helper method to distribute provided observations if necessary """
    if isinstance(observations, Block):
        return observations
    else:
        return mpi_np.array(observations)

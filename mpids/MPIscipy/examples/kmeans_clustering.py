from mpi4py import MPI
import numpy as np
import mpids.MPInumpy as mpi_np

import mpids.MPIscipy.cluster as mpi_scipy_cluster

if __name__ == "__main__":

    #Capture default communicator and MPI process rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    #Create simulated 1D observation vector
    k, num_points, centers = 2, 10, [[-1, -0.75],
                                     [1, 1.25]]
    x0 = np.random.uniform(centers[0][0], centers[0][1], size=(num_points))
    x1 = np.random.uniform(centers[1][0], centers[1][1], size=(num_points))
    np_1D_obs_features = np.array(x0.tolist() + x1.tolist(), dtype=np.float64)

    #Distribute observations among MPI processes
    mpi_np_1D_obs_features = mpi_np.array(np_1D_obs_features, dist='b')

    #Compute K-Means Clustering Result
    centroids, labels = mpi_scipy_cluster.kmeans(mpi_np_1D_obs_features,
                                                 k,
                                                 #Below are the default kwargs
                                                 thresh=1e-5,
                                                 comm=MPI.COMM_WORLD)

    #Compute K-Means Clustering Result using Non-Distributed Input
    centroids_2, labels_2 = mpi_scipy_cluster.kmeans(np_1D_obs_features, k)

    #Check Distributed & Non-Distributed inputs generate the same result
    assert np.allclose(centroids, centroids_2)
    assert np.allclose(labels, labels_2)

    if rank == 0:
        print('Observations: {}\n'.format(np_1D_obs_features))
        print('Computed Centroids: {}\n'.format(centroids))
        print('Computed Labels: {}\n'.format(labels))

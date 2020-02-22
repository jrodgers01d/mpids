import unittest
import numpy as np
import scipy.cluster.vq as scipy_cluster
from mpi4py import MPI
import mpids.MPInumpy as mpi_np
import mpids.MPIscipy.cluster as mpi_scipy_cluster
from mpids.MPInumpy.distributions.Undistributed import Undistributed



class MPIscipyClusterKmeansTest(unittest.TestCase):

    def __create_1_feature_obs(self):
        """ Create two clusters of observations with 1 feature"""
        centers = [-1, 1]
        dist = 0.25

        np.random.seed(0)
        x0 = np.random.uniform(centers[0], centers[0] + dist, size=(50))
        x1 = np.random.uniform(centers[1], centers[1] + dist, size=(50))

        return np.array(list(x0 + x1), dtype=np.float64)


    def __create_2_feature_obs(self):
        """ Create two clusters of observations with 2 features"""
        centers = [(-0.5, 0.5), (0.5, -0.5)]
        dist = 0.25

        np.random.seed(0)
        x0 = np.random.uniform(centers[0][0], centers[0][0] + dist, size=(50,))
        y0 = np.random.normal(centers[0][1], dist, size=(50,))
        x1 = np.random.uniform(centers[1][0], centers[1][0] + dist, size=(50,))
        y1 = np.random.normal(centers[1][1], dist, size=(50,))

        return np.array(list(zip(x0,y0)) + list(zip(x1,y1)), dtype=np.float64)


    def __create_3_feature_obs(self):
        """ Create two clusters of observations with 3 features"""
        centers = [(-0.3, 0.3), (0.3, -0.3)]
        dist = 0.3

        np.random.seed(0)
        x0 = np.random.uniform(centers[0][0], centers[0][0] + dist, size=(50,))
        y0 = np.random.normal(centers[0][1], dist, size=(50,))
        z0 = np.random.normal(centers[0][1], dist, size=(50,))
        x1 = np.random.uniform(centers[1][0], centers[1][0] + dist, size=(50,))
        y1 = np.random.normal(centers[1][1], dist, size=(50,))
        z1 = np.random.normal(centers[1][1], dist, size=(50,))

        return np.array(list(zip(x0,y0,z0)) + list(zip(x1,y1,z1)), dtype=np.float64)


    def __compare_labels(self, label_1, label_2):
        if np.alltrue((label_1) == (label_2)):
            return True
        else:
            #Try flipping the values
            label_1 = 1 - label_1
            return np.alltrue((label_1) == (label_2))


    def setUp(self):
        #Number of clusters
        self.k = 2
        self.obs_1_feature = self.__create_1_feature_obs()
        self.obs_2_features = self.__create_2_feature_obs()
        self.obs_3_features = self.__create_3_feature_obs()
        self.dist_obs_1_feature = mpi_np.array(self.obs_1_feature, dist='b')
        self.dist_obs_2_features = mpi_np.array(self.obs_2_features, dist='b')
        self.dist_obs_3_features = mpi_np.array(self.obs_3_features, dist='b')


    def test_kmeans_produces_same_results_as_scipy_kmeans2_for_1_feature(self):
        scipy_centriods, scipy_labels = \
            scipy_cluster.kmeans2(self.obs_1_feature, self.k, iter=1000)
        mpids_centriods, mpids_labels = \
            mpi_scipy_cluster.kmeans(self.dist_obs_1_feature, self.k)

        self.assertTrue(self.__compare_labels(scipy_labels, mpids_labels))


    def test_kmeans_produces_same_results_as_scipy_kmeans2_for_2_features(self):
        scipy_centriods, scipy_labels = \
            scipy_cluster.kmeans2(self.obs_2_features, self.k, iter=1000)
        mpids_centriods, mpids_labels = \
            mpi_scipy_cluster.kmeans(self.dist_obs_2_features, self.k)

        self.assertTrue(self.__compare_labels(scipy_labels, mpids_labels))


    def test_kmeans_produces_same_results_as_scipy_kmeans2_for_3_features(self):
        scipy_centriods, scipy_labels = \
            scipy_cluster.kmeans2(self.obs_3_features, self.k, iter=1000)
        mpids_centriods, mpids_labels = \
            mpi_scipy_cluster.kmeans(self.dist_obs_3_features, self.k)

        self.assertTrue(self.__compare_labels(scipy_labels, mpids_labels))


if __name__ == '__main__':
    unittest.main()

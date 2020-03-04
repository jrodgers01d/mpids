import unittest
import unittest.mock as mock
import numpy as np
from mpi4py import MPI
import scipy.cluster.vq as scipy_cluster
import mpids.MPInumpy as mpi_np
import mpids.MPIscipy.cluster as mpi_scipy_cluster
from mpids.MPIscipy.cluster._kmeans import _process_centroids, _process_observations
from mpids.MPInumpy.distributions.Undistributed import Undistributed
from mpids.MPInumpy.distributions.Block import Block
from mpids.MPIscipy.errors import TypeError, ValueError


class MPIscipyClusterKmeansTest(unittest.TestCase):

    def __create_1_feature_obs(self):
        """ Create two clusters of observations with 1 feature"""
        centers = [-1, 1]
        dist = 0.25

        np.random.seed(0)
        x0 = np.random.uniform(centers[0], centers[0] + dist, size=(50))
        x1 = np.random.uniform(centers[1], centers[1] + dist, size=(50))

        return np.array(x0.tolist() + x1.tolist(), dtype=np.float64)


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


    def __compare_centroids(self, centroids_1, centroids_2):
        if centroids_1.ndim != centroids_2.ndim:
            centroids_2 = centroids_2.reshape(centroids_1.shape)

        if np.allclose(centroids_1, centroids_2, rtol=1e-05):
            return True
        else:
            #Cycle through all values
            for _ in range(centroids_1.shape[0]):
                centroids_1 = np.roll(centroids_1, 1, axis=0)
                if np.allclose(centroids_1, centroids_2, rtol=1e-05):
                    return True
            return False


    def __compare_labels(self, label_1, label_2):
        if np.alltrue((label_1) == (label_2)):
            return True
        else:
            #Try flipping the values
            label_1 = 1 - label_1
            return np.alltrue((label_1) == (label_2))


    def setUp(self):
        self.comm = MPI.COMM_WORLD
        #Number of clusters
        self.k = 2
        self.seeded_centroids = np.arange(4).reshape(2, 2)
        self.seeded_num_centroids = self.seeded_centroids.shape[0]
        self.seeded_num_features = self.seeded_centroids.shape[-1]
        self.obs_1_feature = self.__create_1_feature_obs()
        self.obs_2_features = self.__create_2_feature_obs()
        self.obs_3_features = self.__create_3_feature_obs()
        self.dist_obs_1_feature = mpi_np.array(self.obs_1_feature, dist='b')
        self.dist_obs_2_features = mpi_np.array(self.obs_2_features, dist='b')
        self.dist_obs_3_features = mpi_np.array(self.obs_3_features, dist='b')


    def test_process_observations_providing_list(self):
        observations = [0,1,2,3,4,5,6,7]
        processed_obs, num_features, labels  = \
            _process_observations(observations, self.comm)
        self.assertTrue(isinstance(processed_obs, Block))
        self.assertEqual(num_features, 1)
        self.assertTrue(isinstance(labels, Block))


    def test_process_observations_providing_numpy_array(self):
        np_observations = np.arange(8)
        processed_obs, num_features, labels  = \
            _process_observations(np_observations, self.comm)
        self.assertTrue(isinstance(processed_obs, Block))
        self.assertEqual(num_features, 1)
        self.assertTrue(isinstance(labels, Block))


    def test_process_observations_providing_mpi_np_array(self):
        #Default block distribution
        mpi_np_observations = mpi_np.arange(8, dist='b')
        processed_obs, num_features, labels  = \
            _process_observations(mpi_np_observations, self.comm)
        self.assertTrue(isinstance(processed_obs, Block))
        self.assertEqual(num_features, 1)
        self.assertTrue(isinstance(labels, Block))

        #Undistributed distribution
        mpi_np_observations = mpi_np.arange(8, dist='u')
        processed_obs, num_features, labels  = \
            _process_observations(mpi_np_observations, self.comm)
        self.assertTrue(isinstance(processed_obs, Block))
        self.assertEqual(num_features, 1)
        self.assertTrue(isinstance(labels, Block))


    def test_process_observations_providing_3D_observations_raises_ValueError(self):
        observations = np.arange(27).reshape(3,3,3)
        with self.assertRaises(ValueError):
            _process_observations(observations, self.comm)


    def test_process_observations_providing_int_1D_features(self):
        k = self.k
        num_features = 1
        obs = self.dist_obs_1_feature
        centroids, num_centroids, temp_centroids = \
            _process_centroids(k, num_features, obs, self.comm)
        self.assertTrue(isinstance(centroids, Undistributed))
        self.assertTrue(isinstance(temp_centroids, Undistributed))
        self.assertEqual(num_centroids, k)
        #Check centroid chosen from observations
        for centroid in centroids:
            self.assertTrue(centroid in self.obs_1_feature)


    def test_kmeans_calls_process_observations(self):
        obs = self.dist_obs_1_feature
        k = self.k
        processed_obs, num_features, labels = \
            _process_observations(obs, self.comm)
        with mock.patch('mpids.MPIscipy.cluster._kmeans._process_observations',
            return_value = (processed_obs, num_features, labels)) as mock_proc_obs:
            mpi_scipy_cluster.kmeans(obs, k)
        mock_proc_obs.assert_called_with(obs, self.comm)


    def test_process_observations_errors_propegated(self):
        with mock.patch('mpids.MPIscipy.cluster._kmeans._process_observations',
            side_effect = Exception('Mock Execption')) as mock_proc_obs:
            with self.assertRaises(Exception):
                mpi_scipy_cluster.kmeans(None, None)


    def test_process_centroids_providing_int_2D_features(self):
        k = self.k
        num_features = 2
        obs = self.dist_obs_2_features
        centroids, num_centroids, temp_centroids = \
            _process_centroids(k, num_features, obs, self.comm)
        self.assertTrue(isinstance(centroids, Undistributed))
        self.assertTrue(isinstance(temp_centroids, Undistributed))
        self.assertEqual(num_centroids, k)
        #Check centroid chosen from observations
        for centroid in centroids:
            self.assertTrue(centroid in self.obs_2_features)


    def test_process_centroids_providing_int_3D_features(self):
        k = self.k
        num_features = 3
        obs = self.dist_obs_3_features
        centroids, num_centroids, temp_centroids = \
            _process_centroids(k, num_features, obs, self.comm)
        self.assertTrue(isinstance(centroids, Undistributed))
        self.assertTrue(isinstance(temp_centroids, Undistributed))
        self.assertEqual(num_centroids, k)
        #Check centroid chosen from observations
        for centroid in centroids:
            self.assertTrue(centroid in self.obs_3_features)


    def test_process_centroids_providing_ndarray(self):
        k = self.seeded_centroids
        num_features = self.seeded_num_features
        obs = self.dist_obs_2_features
        centroids, num_centroids, temp_centroids = \
            _process_centroids(k, num_features, obs, self.comm)
        self.assertTrue(isinstance(centroids, Undistributed))
        self.assertTrue(isinstance(temp_centroids, Undistributed))
        self.assertEqual(num_centroids, k.shape[0])
        #Check seeded centroids returned
        self.assertTrue(np.alltrue(k == centroids))


    def test_process_centroids_providing_Undistributed_MPIArray(self):
        k = mpi_np.array(self.seeded_centroids, dist='u')
        num_features = self.seeded_num_features
        obs = self.dist_obs_2_features
        centroids, num_centroids, temp_centroids = \
            _process_centroids(k, num_features, obs, self.comm)
        self.assertTrue(isinstance(centroids, Undistributed))
        self.assertTrue(isinstance(temp_centroids, Undistributed))
        self.assertEqual(num_centroids, self.seeded_num_centroids)
        #Check seeded centroids returned
        self.assertTrue(np.alltrue(self.seeded_centroids == centroids))


    def test_process_centroids_providing_Distributed_MPIArray(self):
        k = mpi_np.array(self.seeded_centroids, dist='b')
        num_features = self.seeded_num_features
        obs = self.dist_obs_2_features
        centroids, num_centroids, temp_centroids = \
            _process_centroids(k, num_features, obs, self.comm)
        self.assertTrue(isinstance(centroids, Undistributed))
        self.assertTrue(isinstance(temp_centroids, Undistributed))
        self.assertEqual(num_centroids, self.seeded_num_centroids)
        #Check seeded centroids returned
        self.assertTrue(np.alltrue(self.seeded_centroids == centroids))


    def test_process_centroids_providing_non_int_or_array_raises_TypeError(self):
        k = 'A String'
        num_features = 2
        obs = self.dist_obs_2_features
        with self.assertRaises(TypeError):
            _process_centroids(k, num_features, obs, self.comm)


    def test_process_centroids_providing_seeded_centroids_with_too_few_features_raises_ValueError(self):
        k = self.seeded_centroids
        num_features = self.seeded_num_features - 1
        obs = self.dist_obs_2_features
        with self.assertRaises(ValueError):
            _process_centroids(k, num_features, obs, self.comm)


    def test_process_centroids_providing_seeded_centroids_with_too_many_features_raises_ValueError(self):
        k = self.seeded_centroids
        num_features = self.seeded_num_features + 1
        obs = self.dist_obs_2_features
        with self.assertRaises(ValueError):
            _process_centroids(k, num_features, obs, self.comm)


    def test_kmeans_calls_process_centroids(self):
        obs = self.dist_obs_1_feature
        k = self.k
        processed_obs, num_features, labels = \
            _process_observations(obs, self.comm)
        centroids, num_centroids, temp_centroids = \
            _process_centroids(k, num_features, obs, self.comm)
        with mock.patch('mpids.MPIscipy.cluster._kmeans._process_centroids',
            return_value = (centroids, num_centroids, temp_centroids)) as mock_proc_cents:
            mpi_scipy_cluster.kmeans(obs, k)
        mock_proc_cents.assert_called_with(k, num_features, processed_obs, self.comm)


    def test_process_centroids_errors_propegated(self):
        obs = self.dist_obs_1_feature
        k = self.k
        with mock.patch('mpids.MPIscipy.cluster._kmeans._process_centroids',
            side_effect = Exception('Mock Execption')) as mock_proc_cents:
            with self.assertRaises(Exception):
                mpi_scipy_cluster.kmeans(obs, k)


    def test_kmeans_produces_same_results_as_scipy_kmeans2_for_1_feature_no_seed(self):
        scipy_centriods, scipy_labels = \
            scipy_cluster.kmeans2(self.obs_1_feature, self.k, iter=1000)
        mpids_centriods, mpids_labels = \
            mpi_scipy_cluster.kmeans(self.dist_obs_1_feature, self.k)

        #Check results
        self.assertTrue(self.__compare_labels(scipy_labels, mpids_labels))
        self.assertTrue(self.__compare_centroids(scipy_centriods, mpids_centriods))
        #Check returned data types
        self.assertTrue(isinstance(mpids_centriods, Undistributed))
        self.assertTrue(isinstance(mpids_labels, Undistributed))
        #Check number of returned elements
        self.assertTrue(mpids_centriods.globalshape[0] == self.k)
        self.assertTrue(mpids_labels.globalshape[0] == self.obs_1_feature.shape[0])


    def test_kmeans_produces_same_results_as_scipy_kmeans2_for_2_features_no_seed(self):
        scipy_centriods, scipy_labels = \
            scipy_cluster.kmeans2(self.obs_2_features, self.k, iter=1000)
        mpids_centriods, mpids_labels = \
            mpi_scipy_cluster.kmeans(self.dist_obs_2_features, self.k)

        #Check results
        self.assertTrue(self.__compare_labels(scipy_labels, mpids_labels))
        self.assertTrue(self.__compare_centroids(scipy_centriods, mpids_centriods))
        #Check returned data types
        self.assertTrue(isinstance(mpids_centriods, Undistributed))
        self.assertTrue(isinstance(mpids_labels, Undistributed))
        #Check number of returned elements
        self.assertTrue(mpids_centriods.globalshape[0] == self.k)
        self.assertTrue(mpids_labels.globalshape[0] == self.obs_2_features.shape[0])


    def test_kmeans_produces_same_results_as_scipy_kmeans2_for_3_features_no_seed(self):
        scipy_centriods, scipy_labels = \
            scipy_cluster.kmeans2(self.obs_3_features, self.k, iter=1000)
        mpids_centriods, mpids_labels = \
            mpi_scipy_cluster.kmeans(self.dist_obs_3_features, self.k)

        #Check results
        self.assertTrue(self.__compare_labels(scipy_labels, mpids_labels))
        self.assertTrue(self.__compare_centroids(scipy_centriods, mpids_centriods))
        #Check returned data types
        self.assertTrue(isinstance(mpids_centriods, Undistributed))
        self.assertTrue(isinstance(mpids_labels, Undistributed))
        #Check number of returned elements
        self.assertTrue(mpids_centriods.globalshape[0] == self.k)
        self.assertTrue(mpids_labels.globalshape[0] == self.obs_3_features.shape[0])


    def test_kmeans_produces_same_results_as_scipy_kmeans2_for_1_feature_with_numpy_seed(self):
        k = np.array([-1, 1])
        scipy_centriods, scipy_labels = \
            scipy_cluster.kmeans2(self.obs_1_feature, k, iter=1000)
        mpids_centriods, mpids_labels = \
            mpi_scipy_cluster.kmeans(self.dist_obs_1_feature, k)

        #Check results
        self.assertTrue(self.__compare_labels(scipy_labels, mpids_labels))
        self.assertTrue(self.__compare_centroids(scipy_centriods, mpids_centriods))
        #Check returned data types
        self.assertTrue(isinstance(mpids_centriods, Undistributed))
        self.assertTrue(isinstance(mpids_labels, Undistributed))
        #Check number of returned elements
        self.assertTrue(mpids_centriods.globalshape[0] == len(k))
        self.assertTrue(mpids_labels.globalshape[0] == self.obs_1_feature.shape[0])


    def test_kmeans_produces_same_results_as_scipy_kmeans2_for_2_features_with_numpy_seed(self):
        k = np.array([[-1, -1],
                      [1, 1]])
        scipy_centriods, scipy_labels = \
            scipy_cluster.kmeans2(self.obs_2_features, k, iter=1000)
        mpids_centriods, mpids_labels = \
            mpi_scipy_cluster.kmeans(self.dist_obs_2_features, k)

        #Check results
        self.assertTrue(self.__compare_labels(scipy_labels, mpids_labels))
        self.assertTrue(self.__compare_centroids(scipy_centriods, mpids_centriods))
        #Check returned data types
        self.assertTrue(isinstance(mpids_centriods, Undistributed))
        self.assertTrue(isinstance(mpids_labels, Undistributed))
        #Check number of returned elements
        self.assertTrue(mpids_centriods.globalshape[0] == len(k))
        self.assertTrue(mpids_labels.globalshape[0] == self.obs_2_features.shape[0])


    def test_kmeans_produces_same_results_as_scipy_kmeans2_for_3_features_with_numpy_seed(self):
        k = np.array([[-1, -1, -1],
                      [1, 1, 1]])
        scipy_centriods, scipy_labels = \
            scipy_cluster.kmeans2(self.obs_3_features, k, iter=1000)
        mpids_centriods, mpids_labels = \
            mpi_scipy_cluster.kmeans(self.dist_obs_3_features, k)

        #Check results
        self.assertTrue(self.__compare_labels(scipy_labels, mpids_labels))
        self.assertTrue(self.__compare_centroids(scipy_centriods, mpids_centriods))
        #Check returned data types
        self.assertTrue(isinstance(mpids_centriods, Undistributed))
        self.assertTrue(isinstance(mpids_labels, Undistributed))
        #Check number of returned elements
        self.assertTrue(mpids_centriods.globalshape[0] ==  len(k))
        self.assertTrue(mpids_labels.globalshape[0] == self.obs_3_features.shape[0])


    def test_kmeans_produces_same_results_as_scipy_kmeans2_for_1_feature_with_Undistributed_seed(self):
        k = np.array([-1, 1])
        k_mpi_np = mpi_np.array(k, dist='u')
        scipy_centriods, scipy_labels = \
            scipy_cluster.kmeans2(self.obs_1_feature, k, iter=1000)
        mpids_centriods, mpids_labels = \
            mpi_scipy_cluster.kmeans(self.dist_obs_1_feature, k_mpi_np)

        #Check results
        self.assertTrue(self.__compare_labels(scipy_labels, mpids_labels))
        self.assertTrue(self.__compare_centroids(scipy_centriods, mpids_centriods))
        #Check returned data types
        self.assertTrue(isinstance(mpids_centriods, Undistributed))
        self.assertTrue(isinstance(mpids_labels, Undistributed))
        #Check number of returned elements
        self.assertTrue(mpids_centriods.globalshape[0] == len(k))
        self.assertTrue(mpids_labels.globalshape[0] == self.obs_1_feature.shape[0])


    def test_kmeans_produces_same_results_as_scipy_kmeans2_for_2_features_with_Undistributed_seed(self):
        k = np.array([[-1, -1],
                      [1, 1]])
        k_mpi_np = mpi_np.array(k, dist='u')
        scipy_centriods, scipy_labels = \
            scipy_cluster.kmeans2(self.obs_2_features, k, iter=1000)
        mpids_centriods, mpids_labels = \
            mpi_scipy_cluster.kmeans(self.dist_obs_2_features, k_mpi_np)

        #Check results
        self.assertTrue(self.__compare_labels(scipy_labels, mpids_labels))
        self.assertTrue(self.__compare_centroids(scipy_centriods, mpids_centriods))
        #Check returned data types
        self.assertTrue(isinstance(mpids_centriods, Undistributed))
        self.assertTrue(isinstance(mpids_labels, Undistributed))
        #Check number of returned elements
        self.assertTrue(mpids_centriods.globalshape[0] == len(k))
        self.assertTrue(mpids_labels.globalshape[0] == self.obs_2_features.shape[0])


    def test_kmeans_produces_same_results_as_scipy_kmeans2_for_3_features_with_Undistributed_seed(self):
        k = np.array([[-1, -1, -1],
                      [1, 1, 1]])
        k_mpi_np = mpi_np.array(k, dist='u')
        scipy_centriods, scipy_labels = \
            scipy_cluster.kmeans2(self.obs_3_features, k, iter=1000)
        mpids_centriods, mpids_labels = \
            mpi_scipy_cluster.kmeans(self.dist_obs_3_features,k_mpi_np)

        #Check results
        self.assertTrue(self.__compare_labels(scipy_labels, mpids_labels))
        self.assertTrue(self.__compare_centroids(scipy_centriods, mpids_centriods))
        #Check returned data types
        self.assertTrue(isinstance(mpids_centriods, Undistributed))
        self.assertTrue(isinstance(mpids_labels, Undistributed))
        #Check number of returned elements
        self.assertTrue(mpids_centriods.globalshape[0] ==  len(k))
        self.assertTrue(mpids_labels.globalshape[0] == self.obs_3_features.shape[0])


    def test_kmeans_produces_same_results_as_scipy_kmeans2_for_1_feature_with_Block_distributed_seed(self):
        k = np.array([-1, 1])
        k_mpi_np = mpi_np.array(k, dist='b')
        scipy_centriods, scipy_labels = \
            scipy_cluster.kmeans2(self.obs_1_feature, k, iter=1000)
        mpids_centriods, mpids_labels = \
            mpi_scipy_cluster.kmeans(self.dist_obs_1_feature, k_mpi_np)

        #Check results
        self.assertTrue(self.__compare_labels(scipy_labels, mpids_labels))
        self.assertTrue(self.__compare_centroids(scipy_centriods, mpids_centriods))
        #Check returned data types
        self.assertTrue(isinstance(mpids_centriods, Undistributed))
        self.assertTrue(isinstance(mpids_labels, Undistributed))
        #Check number of returned elements
        self.assertTrue(mpids_centriods.globalshape[0] == len(k))
        self.assertTrue(mpids_labels.globalshape[0] == self.obs_1_feature.shape[0])


    def test_kmeans_produces_same_results_as_scipy_kmeans2_for_2_features_with_Block_distributed_seed(self):
        k = np.array([[-1, -1],
                      [1, 1]])
        k_mpi_np = mpi_np.array(k, dist='b')
        scipy_centriods, scipy_labels = \
            scipy_cluster.kmeans2(self.obs_2_features, k, iter=1000)
        mpids_centriods, mpids_labels = \
            mpi_scipy_cluster.kmeans(self.dist_obs_2_features, k_mpi_np)

        #Check results
        self.assertTrue(self.__compare_labels(scipy_labels, mpids_labels))
        self.assertTrue(self.__compare_centroids(scipy_centriods, mpids_centriods))
        #Check returned data types
        self.assertTrue(isinstance(mpids_centriods, Undistributed))
        self.assertTrue(isinstance(mpids_labels, Undistributed))
        #Check number of returned elements
        self.assertTrue(mpids_centriods.globalshape[0] == len(k))
        self.assertTrue(mpids_labels.globalshape[0] == self.obs_2_features.shape[0])


    def test_kmeans_produces_same_results_as_scipy_kmeans2_for_3_features_with_Block_distributed_seed(self):
        k = np.array([[-1, -1, -1],
                      [1, 1, 1]])
        k_mpi_np = mpi_np.array(k, dist='b')
        scipy_centriods, scipy_labels = \
            scipy_cluster.kmeans2(self.obs_3_features, k, iter=1000)
        mpids_centriods, mpids_labels = \
            mpi_scipy_cluster.kmeans(self.dist_obs_3_features, k_mpi_np)

        #Check results
        self.assertTrue(self.__compare_labels(scipy_labels, mpids_labels))
        self.assertTrue(self.__compare_centroids(scipy_centriods, mpids_centriods))
        #Check returned data types
        self.assertTrue(isinstance(mpids_centriods, Undistributed))
        self.assertTrue(isinstance(mpids_labels, Undistributed))
        #Check number of returned elements
        self.assertTrue(mpids_centriods.globalshape[0] ==  len(k))
        self.assertTrue(mpids_labels.globalshape[0] == self.obs_3_features.shape[0])


if __name__ == '__main__':
    unittest.main()

"""Module which provides a gpflow kernel taking into account the environment structure"""
import numpy as np
from copa_map.kernel.kernel_grid import KernelGrid
from copa_map.util import util
from gpflow.kernels.stationaries import Matern52
import tensorflow as tf
from brescount import matrix_minima_on_line
from sklearn.preprocessing import StandardScaler


def make_pd(matrix):
    """
    Makes matrix positive definite

    Is used for computing the nearest pd matrix by wrapping the function in a TensorFlow op,
    which executes it eagerly
    """
    if util.isPD(matrix) or matrix.shape[0] != matrix.shape[1] or tf.reduce_sum(matrix) == 0.:
        return matrix
    else:
        # util.logger().info("Calculating approximate positive definite K-Matrix")
        return util.nearestPD(matrix)


@tf.custom_gradient
def nearest_pd_grad_layer(matrix):
    """
    Sets Custom gradient for computing of nearest pd

    Gradient layer is used to ensure that the gradient is not lost with calculation of the nearest
    positive definite matrix
    """
    def grad(dy):
        return dy
    y = tf.numpy_function(func=make_pd, inp=[matrix], Tout=tf.float64, name="MakePosDefinite")
    return y, grad


class MaternMap(Matern52):
    """
    Kernel class

    Class for kernel built by environment map and Matern52 covariance function
    """
    def __init__(self, variance=1., lengthscales=1., active_dims=None, digitize_size=0.2, num_of_borders=2,
                 kernel_map=None, occ_map=None, nearest_pd=True, display_information=True,
                 data_scaler : StandardScaler = None, min_adjustment_val=1e-3):
        """
        Constructor

        Args:
            variance: Init variance of Matern function
            lengthscales: Init lengthscale of Matern function
            active_dims: Active dimensions
            digitize_size: Discretization size for map grid
            num_of_borders: Number of cells around occupied cells, from which covariance factor increases
                            linearly from 0 to 1
            kernel_map: Discretized environment structure; must be calculated only once for one map
            occ_map: Environment map is used to calculate kernel_map
            nearest_pd: If true, nearest positive definite covariance matrix is calculated to increase
                        numerical stability
            display_information: Enabled log messages
            data_scaler: Fitted scaler of the spatial data. Necessary to retransform the scaled values to metric values
            min_adjustment_val: Minimal value of the adjustment matrix. If one, adjustment matrix has no influence.
        """
        super(MaternMap, self).__init__(variance=variance, lengthscales=lengthscales,
                                        active_dims=active_dims, name='Mat52_Map')
        assert np.all(active_dims == [0, 1]), "The two spatial dimensions are expected in column " \
                                              "0 and 1 of the training data"

        if kernel_map is None:
            assert occ_map, "Occupancy grid map must be given to kernel, if no kernel map is given"
            # Calculate grid providing occupancy values of the map
            self.kernel_map = KernelGrid(occ_map, digitize_size, num_of_borders)
        else:
            self.kernel_map = kernel_map

        self.display_information = display_information
        self.nearest_pd = nearest_pd
        self.data_scaler = data_scaler
        if self.nearest_pd and self.display_information:
            print("Will compute nearest positive definite matrix if necessary for Cholesky decomposition")

        # For approximation by means of r2_max, from which square distance the points have no correlation to each other:
        self.use_sparse_kernel = True  # enables approximation
        self.cov_min = 1e-2  # Calculation of the maximum distance via definition of the minimum covariance
        # Kernel parameters used in the calculation of the maximum distance:
        self.var_last = variance
        self.len_last = lengthscales

        # Number of cpu cores
        self.cpu_count = util.get_cpu_count()

        # You can set output data for non-symmetric kernel
        self.Yd = None
        self.symmetric_kernel = True  # Do not distinguish between occupied and unoccupied training data
        # Enables Bresenham kernel
        # Reuse of calculated adjustment matrices during optimization
        self._inducing_flag = False
        self.last_kernel_XX = None
        self.last_kernel_ZZ = None
        self.last_kernel_ZX = None
        # Store for prediction pre-calculated adjustment matrix (useful for prediction in time dimension)
        self.predict_kernel = {"X": None, "X2": None, "kernel": None}
        # Value of the adjustment matrix if there is no correlation
        self.min_value = min_adjustment_val

    # @staticmethod
    # def matern52(r, var, len, cov):
    #     """Function value of the Matern52 covariance function"""
    #     sqrt5 = np.sqrt(5.0)
    #     return cov - var * (1.0 + sqrt5 * r / len + (5. * np.square(r)) /
    #                         (3. * np.square(len))) * np.exp(-sqrt5 * r / len)

    # @staticmethod
    # def squared_exponential(var, len, lower_cov):
    #     """
    #     Quadratic distance for specific covariance (RBF)
    #
    #     Returns quadratic Euclidean distance from which the RBF covariance function is smaller
    #     than the defined lower limit of covariance
    #     """
    #     r2 = 2 * np.square(len) * np.log(var / lower_cov)
    #     return r2

    def _inverse_scale_data_if_needed(self, X):
        """
        Retransforms the data to actual metric values.

        Bresenham kernel is based on the environment, so the scale must
        be inverted before value calculation
        Args:
            X: Scaled data

        Returns:
            Rescaled data, if data scaler is defined. Else original data.
        """
        if self.data_scaler is not None:
            # Inverse scale the data so that it works with the map based kernel
            return self.data_scaler.inverse_transform(X)
        else:
            return X

    def pre_init_kernel_map(self, X, var=None, len=None, inducing=False, Z=None):
        """
        Inititalizing the covariance matrix enables speed advantage for optimization in tensorflows graph mode

        Saving of the adjustment matrix only possible with initialization
        or if optimization is always in Eager mode.

        Args:
            X: Training data
            var: Is used to calculate r2_max. Otherwise internal parameter from initialization is used
            len: Is used to calculate r2_max. Otherwise internal parameter from initialization is used
            inducing: Specifies whether optimization is performed with inducing points
            Z: inducing points
        """
        print("Compute pre_init_kernel_map...")
        X = X[:, :2]  # data without time dimension
        if None not in (var, len):
            self.var_last = var
            self.len_last = len

        if not inducing:
            self.last_kernel_XX = tf.convert_to_tensor(self.calc_adjustment_matrix(X, X))
            self._inducing_flag = False
        else:
            assert Z is not None,\
                "Inducing points must be given if optimization should be performed with inducing points"
            Z = Z[:, :2]
            assert Z.shape[0] < X.shape[0], "Must use less inducing points than input points"
            self.last_kernel_ZZ = tf.convert_to_tensor(self.calc_adjustment_matrix(Z, Z))
            self.last_kernel_ZX = tf.convert_to_tensor(self.calc_adjustment_matrix(Z, X))
            self._inducing_flag = True

    def pre_predict_kernel_map(self, X_data, X_test, store_kernel_map=False):
        """
        Initialization enables calculation of the adjustment matrix

        Args:
            X_data: Training data
            X_test: Test data
            store_kernel_map: Useful for prediction in temporal dimension
        """
        # Without temporal dimension
        X = X_data[:, :2]
        X2 = X_test[:, :2]

        self.nearest_pd = True  # always calculate next positive definite matrix for prediction if necessary
        # gf.utilities.reset_cache_bijectors(self)  # reset the caches for multiprocessing (eager mode problem)

        # If for prediction in time dimension the adjustment matrix was saved before, it will be restored later
        if store_kernel_map and np.all(self.predict_kernel["X"] == X) and np.all(self.predict_kernel["X2"] == X2):
            return

        # Pre-Calculate K_map, but save X and X2 only if necessary (because of memory problems)
        self.predict_kernel = {"X": X if store_kernel_map else None, "X2": X2 if store_kernel_map else None,
                               "kernel": self.calc_adjustment_matrix(X, X2)}

    def store_kernels(self, kernel, is_symmetric):
        """
        Stores adjustment matrices so that they do not have to be recalculated in each optimization step

        Args:
            kernel: Matrix to store
            is_symmetric: During optimization matrices are symmetric, except when using inducing points
        """
        if not self._inducing_flag:
            self.last_kernel_XX = kernel
        elif is_symmetric:
            self.last_kernel_ZZ = kernel
        else:
            self.last_kernel_ZX = kernel
        return kernel

    def calc_adjustment_matrix(self, X1, X2, in_grid_frame=False):
        """
        Covariance adjustment matrix between X1 and X2

        Calculates the covariance  adjustment matrix in dependence of the room map by determining cells between two
        discretized points with a Bresenham line and characterizing decreasing covariances near walls with values
        from 1 (unchanged covariance) to 0 (no covariance)

        Params:
            X1: Array with shape (n,2)
            X2: Array with shape (m,2)

        Returns:
            Covariance adjustment matrix with shape (n,m)
        """
        assert self.symmetric_kernel

        if self.min_value == 1.0:
            # We need no calculation if we only multiply by one
            return np.ones([X1.shape[0], X2.shape[0]])

        # print("Calculate [" + str(X1.shape[0]) + "," + str(X2.shape[0]) +"] adjustment matrix")
        X1 = self._inverse_scale_data_if_needed(X1)
        X2 = self._inverse_scale_data_if_needed(X2)

        # Corresponding cells of the pre-calculated kernel map
        if not in_grid_frame:
            X1 = self.kernel_map.tf_to(X1)
            X2 = self.kernel_map.tf_to(X2)

        X1_digs = self.kernel_map.index_from_pos(X1, clip=True)
        X2_digs = self.kernel_map.index_from_pos(X2, clip=True)
        # print("Calc adjustment matrix with shape [{}x{}]".format(X1_digs.shape[0], X2_digs.shape[0]))

        X12_combs = np.ascontiguousarray(np.hstack([
            np.repeat(X1_digs, X2_digs.shape[0], axis=0),
            np.tile(X2_digs, (X1_digs.shape[0], 1))]
        ))
        # Transpose map because we use (ix, iy), map uses it vice versa
        grid_2d = np.ascontiguousarray(self.kernel_map.map.T)

        KXX = matrix_minima_on_line(X12_combs, grid_2d.astype(np.float64), self.min_value,
                                    num_threads=self.cpu_count).reshape([X1_digs.shape[0], -1])
        # KXX = np.ones([X1.shape[0], X2.shape[0]])
        return KXX

    def K_tf_op(self, X, X2, var=None, len=None):
        """Is used for computing the adjustment matrix by wrapping used functions in a tensorflow operation"""
        # Compute adjustment kernel
        kernel = tf.numpy_function(func=self.calc_adjustment_matrix, inp=[X, X2],
                                   Tout=tf.float64)
        kernel.set_shape([X.shape[0], X2.shape[0]])
        return kernel

    def K(self, X, X2=None):
        """
        By map modified Matern52 kernel

        Calculates a covariance matrix using a Matern52 covariance function and multiplies the matrix entries element-
        wise with values from the adjustment matrix, which takes values between 0 (no covariance) and 1 (no adjustment)

        Args:
            X: Array with shape (n,2)
            X2: Array with shape (m,2)

        Returns:
            Covariance matrix with shape (n,m)
        """
        X = X[:, :2]  # Without temporal dimension
        if X2 is None:
            X2 = X
            is_symmetric = True
        else:
            is_symmetric = False

        # try:
        #     X.numpy()
        #     # If the hyper-parameters have increased, the adjustment matrix must be recalculated due to an
        #     # increase in r2_max (sparse kernel)
        #     # >>> Initial high values avoid frequent recalculations of the adjustment matrix <<<
        #     re_calc = self.variance > self.var_last or self.lengthscales > self.len_last
        # except AttributeError:
        #     re_calc = False
        re_calc = False

        def _retrieve_adj_matrix(X, X2, is_symmetric):
            # Possible kernels to use for restoring
            # kernels_init =
            # [self.last_kernel_ZZ, self.last_kernel_ZX] if self._inducing_flag else [self.last_kernel_XX]
            # kernels = [self.predict_kernel["kernel"]] + kernels_init

            # Restore kernel
            kernel = None
            # for k in kernels:
            #     if k is not None and k.shape == [X.shape[0], X2.shape[0]]:
            #         kernel = k
            #         break

            if kernel is not None:
                # Kernel restored, but recalculation may be necessary due to an increase in the hyper-parameters
                if re_calc:
                    self.var_last = tf.math.ceil(self.variance)
                    self.len_last = tf.math.ceil(self.lengthscales)
                    adj_matrix = self.K_tf_op(X, X2)
                    self.store_kernels(adj_matrix, is_symmetric)
                else:
                    adj_matrix = kernel
            else:
                # Kernel not restored
                # if None in kernels_init:
                adj_matrix = self.K_tf_op(X, X2)
                # The kernel can be restored in future function calls only if it was computed in eager mode
                self.store_kernels(adj_matrix, is_symmetric)
                adj_matrix = self.K_tf_op(X, X2, var=self.variance, len=self.lengthscales)
                # else:
                #    raise ValueError("Possibly the training data has changed during optimization")
            return adj_matrix

        # Compute matrix with distances and get adjustment matrix
        # adding small value increases numerical stability when points are very close to each other
        r2 = self.scaled_squared_euclid_dist(X, X2) + 1e-6

        # Compute stationary kernel matrix and multiply element wise with adjustment matrix
        K_matern = self.K_r2(r2)
        K_map = _retrieve_adj_matrix(X=X, X2=X2, is_symmetric=is_symmetric)

        K_product = tf.math.multiply(K_map, K_matern)
        # Compute nearest positive definite matrix if desired and necessary
        if self.nearest_pd:
            K_product_pd = nearest_pd_grad_layer(K_product)
            return K_product_pd
        else:
            return K_product

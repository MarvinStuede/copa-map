"""
Module for basic Gaussian Process People Model

Every model variant consists of a parameter dataclass, that inherits from CoPA;apParams and a model class inheriting
from CoPAMapBase
"""
import numpy as np
import gpflow

import tensorflow as tf
from dataclasses import dataclass, field
from tqdm import tqdm
from termcolor import colored
import copa_map.model.model_utils as mu
from copa_map.model.ModelInterface import ModelInterface, ModelParams
from copa_map.model import Likelihoods
from typing import Type
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from copa_map.model.Optimization import Optimization
from gpflow import set_trainable
from functools import reduce
import operator
from tensorflow.python.framework.ops import EagerTensor
from copa_map.util import util as ut
from copy import copy
from typing import Dict


@dataclass
class CoPAMapParams(ModelParams):
    """Basic parameters for People Gaussian process"""

    # Maximum iterations for GP optimization
    opt_max_iters: int = 2500
    # Maximum duration (in s) for GP optimization
    opt_max_dur: int = 3600
    # Use inducing points for global approximation
    use_inducing: bool = True
    # Optimize location of inducing points
    train_inducing: bool = True
    # Size of minibatches to speed up optimization. If 0, no minibatches are used
    minibatch_size: float = 0
    # Normalize input and output data
    # Input data will be normalized to range [0, 1] based on the spatial dimension with maximum range
    # Output data will be standardized to have zero mean and standard deviation=1
    normalize_output: bool = True
    normalize_input: bool = False
    # Approximate closest PD matrix if Cholesky decomposition fails
    run_with_nearest_pd: bool = False
    # Type of temporal kernel
    # Can be "periodic", "rbf", or "periodic-rbf"
    temporal_kernel: str = 'periodic'
    # Specifies the optimizer to use
    # You can use 'adam_natgrad', 'L-BFGS-B' or 'adam'
    optimizer: str = "adam_natgrad"
    # gaussian_ml, gaussian or poisson
    # gaussian_ml is a multi-latent gaussian likelihood, where a second GP is used for the variance function
    likelihood: str = "gaussian_ml"

    # Parameters (Hyper, Trainable) of GP Model and Scalers
    # (Everything needed for reconstruction of already trained model)
    gp_model_stor: dict = field(default_factory=dict)

    # ## KERNEL PARAMETERS
    # List of expected temporal periods. Each period will result in a periodic kernel
    periods: list = field(default_factory=list)
    # Weights of the periods. Will be used to initialize variance parameter
    period_weights: list = field(default_factory=list)
    # Set periods as trainable by GP optimization
    periods_trainable: bool = True
    # The following parameters are bounds (low, high) and initial parameters (parameter) for the kernel hyperparameters
    # The bound of the lengthscale parameter avoids unrealistically large lengthscales
    # The variance should be smaller than one, because of standardized data

    # Lengthscale/Variance of spatial kernel
    kern_spat_lengthsc: Dict[str, float] = field(
        default_factory=lambda: ({'low': 1e-10, 'high': 3.5, 'parameter': 0.15}))
    kern_spat_var: Dict[str, float] = field(default_factory=lambda: ({'low': 1e-9, 'high': 1.0, 'parameter': 0.3}))

    # Lengthscale/Variance of each periodic base kernel
    # Derived based on normalized input data. The upper bounds are already unrealistically large values
    kern_temp_per_lengthsc = {'low': 1e-1, 'high': 0.6, 'parameter': 0.4}
    kern_temp_per_var = {'low': 1e-9, 'high': 1.01, 'parameter': 0.3}


def _likelihood_is_ml(params):
    return (params.likelihood == "gaussian_ml")


class CoPAMapBase(ModelInterface):
    """Base class for Gaussian Process based People Occurrence modeling"""

    def __init__(self, params: Type[CoPAMapParams] = CoPAMapParams()):
        """Constructor"""
        self.pr = params
        self.x_spat_scaler = None
        self.x_temp_scaler = None
        self.y_scaler = None

        self.num_latent_gp = 2 if _likelihood_is_ml(self.pr) else 1

        # If these params exist, write them to the model (e.g. because of preceding optimization)
        if len(params.gp_model_stor) > 0:
            self.write_model_params(params)

    def kernel(self, use_temp_dim=False, use_nearest_pd=False, init=False):
        """Kernel in spatial and temporal dimension

        Args:
            use_temp_dim: Specifies whether the temporal dimension is to be modeled in addition to the spatial dimension
            use_nearest_pd: Specifies whether the next positive definite kernel matrix should be used if kernel matrix
                            is not positive definite
            init: To check other hyper-parameters and their effect on stability when initializing

        Returns:
            kernel
        """
        # Spatial kernel
        kern = self._kernel_spatial_dim(use_nearest_pd=use_nearest_pd)
        # Add temporal kernel
        if use_temp_dim:
            kern = kern * self._kernel_temporal_dim()
        # In a multi latent model use an RBF kernel for latent function g
        if self.num_latent_gp == 2:
            # Create multi-latent kernels
            kern = gpflow.kernels.SeparateIndependent(
                [
                    kern,  # kernel for latent function f
                    # Values are chosen as small start values because for normalized input and standardized output
                    # data the standard values are not reasonable
                    gpflow.kernels.SquaredExponential(lengthscales=0.02, variance=0.5, active_dims=[0, 1, 2]),
                    # kernel for latent function g
                ]
            )

        return kern

    def _kernel_spatial_dim(self, *args, **kwargs):
        """Kernel in spatial dimension"""
        kern = gpflow.kernels.Matern52(active_dims=[0, 1])
        kern.variance = mu.bounded_hyperparameter(**self.pr.kern_spat_var)
        kern.lengthscales = mu.bounded_hyperparameter(**self.pr.kern_spat_lengthsc)

        # Add white noise kernel to increase numerical stability
        kern = kern + gpflow.kernels.White(active_dims=[0, 1], variance=1e-9)

        return kern

    def _kernel_temporal_dim(self):
        """Kernel in temporal dimension"""

        def create_periodic_kernel(period, weight):
            """Given a period and weight, creates a periodic kernel with an RBF base kernel"""
            kern_per = gpflow.kernels.Periodic(gpflow.kernels.RBF(active_dims=[2]), period)
            gpflow.set_trainable(kern_per.period, self.pr.periods_trainable)
            # Set lengthscale and variance as bounded hyperparameters
            kern_per.base_kernel.lengthscales = mu.bounded_hyperparameter(**self.pr.kern_temp_per_lengthsc)
            self.pr.kern_temp_per_var['parameter'] = weight
            kern_per.base_kernel.variance = mu.bounded_hyperparameter(**self.pr.kern_temp_per_var)
            return copy(kern_per)

        if not self.pr.periods:  # If no periods are given, just use an RBF kernel
            self.pr.temporal_kernel = "rbf"

        # Create the periodic kernels
        if "periodic" in self.pr.temporal_kernel:
            # When inputs are scaled, periods must be scaled in the same way as the temporal dimension of the inputs
            if self.pr.normalize_input and self.x_temp_scaler is not None:
                periods = self.x_temp_scaler.transform(np.array([self.pr.periods]).reshape(-1, 1))
            else:
                periods = self.pr.periods
            if self.pr.period_weights:
                assert len(periods) == len(self.pr.period_weights), "Must give same number of weights as periods"
            else:
                self.pr.period_weights = [1] * len(periods)
            kernels_period = list(map(create_periodic_kernel, periods, self.pr.period_weights))

        kern_rbf = gpflow.kernels.SquaredExponential(0.5, 0.3, active_dims=[2], name="periodic_rbf")
        gpflow.set_trainable(kern_rbf.lengthscales, True)
        gpflow.set_trainable(kern_rbf.variance, True)

        if "periodic" in self.pr.temporal_kernel and "rbf" in self.pr.temporal_kernel:
            kern = kern_rbf * reduce(operator.add, kernels_period)
        elif "periodic" in self.pr.temporal_kernel:
            # Sum up all periodic kernels
            kern = reduce(operator.add, kernels_period)
        elif "rbf" in self.pr.temporal_kernel:
            kern = kern_rbf
        else:
            raise NotImplementedError

        return kern

    def likelihood(self, lik):
        """
        Defines the likelihood

        Args:
            lik: String indicating which likelihood to use

        Returns:
            gpflow Likelihood to use
        """
        if lik == "poisson":
            return gpflow.likelihoods.Poisson()
        elif lik == "gaussian":
            return gpflow.likelihoods.Gaussian(variance=0.7)
        elif lik == "gaussian_ml":
            return Likelihoods.HeteroskedasticMLGaussian()
        else:
            raise NotImplementedError

    def get_model(self, x, y, kern, likelihood, inducing_points=None):
        """
        Creates the model

        If inducing points are passed, an SVGP is used; otherwise VGP
        Args:
            x: Data inputs
            y:  Targets corresponding to inputs
            kern: kernel to use
            likelihood: likelihood to use
            inducing_points: Positions of inducing points

        Returns:
            gpflow.model object
        """
        if inducing_points is None:
            model = gpflow.models.VGP(data=(tf.convert_to_tensor(x), tf.convert_to_tensor(y.astype(float))),
                                      kernel=kern, likelihood=likelihood,
                                      num_latent_gps=self.num_latent_gp)
        else:
            num_data = x.shape[0] if x is not None else None
            model = gpflow.models.SVGP(kernel=kern, likelihood=likelihood,
                                       inducing_variable=inducing_points, num_data=num_data,
                                       num_latent_gps=self.num_latent_gp)
        return model

    def learn(self, X, Y, Z=None, log_dir=None, **kwargs):
        """
        Learn the model (all domains/ entire map)

        Args:
            X: Input Positions
            Y: Observation rate corresponding to the input locations
            Z: Inducing variables
            log_dir: Path to a directory where Tensorboard logs are stored
        """
        self.log_dir = log_dir
        assert np.all(Y[:] > 0), "Rate must be larger than zero"
        # Transform data
        # data to positive value range afterwards
        ut.logger().info("Train Y rate, Max: " + str(Y[:, 0].max())
                         + ", Mean: " + str(Y[:, 0].mean())
                         + ", Min: " + str(Y[:, 0].min()))
        # Add small value to increase numerical stability if there are many zeros
        Y += 1e-10

        # Normalization or scaling to predict values of 0 in unobserved regions
        if self.pr.normalize_output:
            print(colored("Normalize output data", "green"))

        self.model, stored_model = self._learn(X=X, Y=Y, Z=Z, **kwargs)
        return self.model, stored_model

    def _learn(self, X, Y, Z=None, rmse_clbl=None):
        """
        Learn the model (part of the map)

        Args:
            X: Input Positions
            Y: Observation rate corresponding to the input locations
            Z: Inducing variables


        Returns:
            (Optimized model, recovery model)
        """
        use_temp_dim = True if X.shape[1] == 3 else False

        if self.pr.normalize_input or self.pr.normalize_output:
            self.X, self.Y, self.Z = self._scale_data(X, Y, Z)
        else:
            self.X = X
            self.Y = Y
            self.Z = Z

        ut.logger().info("Learning model with " + str(self.X.shape[0]) + " input points, and " + str(self.Z.shape[0])
                         + " inducing points")

        self.optimization = self._set_optimization(self.log_dir, rmse_clbl)

        # Determine kernel, model and optimization parameters depending on the selected method
        likelihood = self.likelihood(lik=self.pr.likelihood)
        kernel = self.kernel(use_temp_dim=use_temp_dim, use_nearest_pd=self.pr.run_with_nearest_pd, init=False)
        model = self._set_learning_model(likelihood, kernel, self.pr.use_inducing)

        # train_iter used to rebuild loss function if needed
        train_iter = None if not isinstance(model, gpflow.models.SVGP) else (self.X, self.Y.astype(float))
        if self.pr.minibatch_size > 0:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (self.X, self.Y.astype(float))).repeat().shuffle(self.X.shape[0])
        else:
            train_dataset = None
        # Min/Max spatial values for plotting
        xmin = np.min(self.X[:, :2])
        xmax = np.max(self.X[:, :2])
        # Start optimization process
        optimized = self.optimization.optimize(model=model, optimizer=self.pr.optimizer,
                                               train_dataset=train_dataset, train_iter=train_iter,
                                               minibatch_size=self.pr.minibatch_size,
                                               run_with_nearest_pd=self.pr.run_with_nearest_pd,
                                               xmin=xmin, xmax=xmax)

        return (self.optimization.model, self.optimization.stored_model) if optimized else (None, None)

    def _scale_data(self, X, Y, Z):
        """
        Scale the data to normalize/standardize

        If a gaussian likelihood is used, the targets are standardized, else normalized to a given range.
        Inducing inputs are normalized just like the input points.
        Args:
            X: Input points
            Y: Targets
            Z: Inducing points

        Returns:
            Scaled X, Y, Z
        """
        use_temp_dim = True if X.shape[1] == 3 else False

        if self.pr.normalize_input:
            # Different scalers for temporal and spatial dimension
            # This is necessary for kernels that only work on spatial dimensions
            # Also, we want to scale the two spatial dimensions with the same factor to not skew the data
            self.x_spat_scaler = MinMaxScaler()
            self.x_temp_scaler = MinMaxScaler()
            # Fit the spatial scaler to all input points and inducing points
            self.x_spat_scaler.fit(np.vstack([X[:, :2], Z[:, :2]]))
            # Use normalizer for spatial dim with the largest range (Smallest scale)
            use_id = 0 if self.x_spat_scaler.scale_[0] < self.x_spat_scaler.scale_[1] else 1
            # We set the scale to the other dimension --> Data will not be skewed
            self.x_spat_scaler.scale_ = np.repeat(self.x_spat_scaler.scale_[use_id], 2)
            # Scaler expects 2d array, so reshape is necessary for time dimension
            if use_temp_dim:
                Xs = np.hstack([self.x_spat_scaler.transform(X[:, :2]),
                                self.x_temp_scaler.fit_transform(X[:, 2].reshape(-1, 1))])
            else:
                Xs = self.x_spat_scaler.transform(X[:, :2])

            if use_temp_dim:
                Zs = np.hstack([self.x_spat_scaler.transform(Z[:, :2]),
                                self.x_temp_scaler.transform(Z[:, 2].reshape(-1, 1))])
            else:
                Zs = self.x_spat_scaler.transform(Z[:, :2]) if Z.size else Z
        else:
            Xs = X
            Zs = Z

        if self.pr.normalize_output:

            if "gaussian" in self.pr.likelihood:
                self.y_scaler = StandardScaler()
            else:
                self.y_scaler = MinMaxScaler(feature_range=(1e-8, 30))

            Ys = self.y_scaler.fit_transform(Y[:, 0].reshape(-1, 1))

        else:
            Ys = Y

        return Xs, Ys, Zs

    def _inverse_scale(self, Xs, Ys):
        """
        Inverse scaling of input and output data

        Args:
            Xs: Scaled input values
            Ys: Scaled targets

        Returns:
            Rescaled X and Y
        """
        X = np.hstack([self.x_spat_scaler.inverse_transform(Xs[:, :2]),
                       self.x_temp_scaler.inverse_transform(Xs[:, 2].reshape(-1, 1))])
        Y = self.y_scaler.inverse_transform(Ys.reshape(-1, 1))
        return X, Y

    def _set_learning_model(self, likelihood, kernel, use_inducing):
        """Given the model parameters, create the specific model"""
        if self.num_latent_gp == 2:
            inducing_variable = gpflow.inducing_variables.SeparateIndependentInducingVariables(
                [  # Both latent functions use the same initial inducing points
                    gpflow.inducing_variables.InducingPoints(self.Z),  # f(Z1)
                    gpflow.inducing_variables.InducingPoints(self.Z),  # g(Z2)
                ]
            )
        else:
            inducing_variable = self.Z
        model = self.get_model(self.X, self.Y, kernel, likelihood,
                               inducing_points=inducing_variable if use_inducing else None)

        if use_inducing:
            set_trainable(model.inducing_variable, self.pr.train_inducing)
        return model

    def _set_optimization(self, log_dir, rmse_clbl):
        return Optimization(method="without_map", max_duration_for_opt=self.pr.opt_max_dur,
                            max_iters_for_opt=self.pr.opt_max_iters, seed=self.pr.seed, log_dir=log_dir,
                            rmse_clbl=rmse_clbl)

    def predict(self, Xt):
        """
        Predict the models output at given positions

        Args:
            Xt: inputs (n x 3) location and time

        Returns:
            Mean and std. dev
        """
        tf.print("predict...")
        mean, stdd = self._predict(Xt)
        return mean, stdd

    def _get_predict_model(self):
        """Get the model for prediction which can be the optimized model or a restored model"""
        return self.optimization.model if (hasattr(self, 'optimization') and self.optimization.model is not None) \
            else self.model

    def _predict(self, x):
        # Prediction and reverse transformation

        if self.x_spat_scaler is not None:
            if x.shape[1] == 3:
                x = np.hstack([self.x_spat_scaler.transform(x[:, :2]),
                               self.x_temp_scaler.transform(x[:, 2].reshape(-1, 1))])
            else:
                x = self.x_spat_scaler.transform(x[:, :2])

        def predict_fun(x):
            if self.num_latent_gp == 2:
                return self._get_predict_model().predict_y(x)
                # m, v = self._get_predict_model().predict_f(x)
                # return m[:, 0], v[:, 0]
            else:
                return self._get_predict_model().predict_f(x)

        # Workaround for large prediction datasets
        # Will divide the dataset into multiple smaller datasets, if size is above some empirically defined threshold
        if x.shape[1] < 3 or x.shape[0] < 2000:
            m, v = predict_fun(x)
        else:
            m = np.ones((x.shape[0], 1)) * float("Inf")
            v = np.ones((x.shape[0], 1)) * float("Inf")
            t0 = np.unique(x[:, 2])[0]
            step = 1
            total_len = np.unique(x[:, 2]).shape[0]
            with tqdm(total=total_len) as pbar:
                for i in range(step, total_len + step, step):
                    pbar.update(step)
                    if i < total_len:
                        t1 = np.unique(x[:, 2])[i]
                        sel = (x[:, 2] >= t0) & (x[:, 2] < t1)

                    else:
                        t1 = np.unique(x[:, 2])[-1]
                        sel = (x[:, 2] >= t0) & (x[:, 2] <= t1)
                    xpart = x[sel]
                    mpart, vpart = predict_fun(xpart)

                    def from_eager(arr):
                        return arr.numpy() if (isinstance(arr, EagerTensor)) else arr

                    mpart = from_eager(mpart).reshape(-1, 1)
                    vpart = from_eager(vpart).reshape(-1, 1)
                    m[sel] = mpart
                    v[sel] = vpart
                    t0 = t1
            assert np.all(np.isfinite(m)), "Distributed prediction went wrong for mean"
            assert np.all(np.isfinite(v)), "Distributed prediction went wrong for variance"
        # v should be positive, but in case of numerical errors use absolute
        stdd = np.sqrt(np.abs(v))
        if self.y_scaler is not None:
            m = self.y_scaler.inverse_transform(m)
            stdd *= self.y_scaler.scale_

        m = np.clip(m, a_min=0, a_max=None)
        m = self.replace_zeros_with_eps(m)
        print("m_y max: " + str(m.max()))
        print("m_y min: " + str(m.min()))
        print("m_y mean: " + str(m.mean()))
        return m, stdd

    def get_model_params(self):
        """Return the model parameters"""
        self.pr.gp_model_stor = {
            "model_params": gpflow.utilities.read_values(self.model),
            "x_temp_scaler": self.x_temp_scaler,
            "x_spat_scaler": self.x_spat_scaler,
            "y_scaler": self.y_scaler,
        }
        return self.pr

    def write_model_params(self, params):
        """Set the model params from a variable

        Can e.g. be used to restore a model from a file
        """
        if ".inducing_variable.Z" not in params.gp_model_stor["model_params"] and \
                ".inducing_variable.inducing_variable_list[0].Z" not in params.gp_model_stor["model_params"]:
            print("Not writing model because no inducing variable found")
            return
        if _likelihood_is_ml(params):
            inducing_variable = gpflow.inducing_variables.SeparateIndependentInducingVariables(
                [
                    gpflow.inducing_variables.InducingPoints(params.gp_model_stor["model_params"]
                                                             [".inducing_variable.inducing_variable_list[0].Z"]),
                    gpflow.inducing_variables.InducingPoints(params.gp_model_stor["model_params"]
                                                             [".inducing_variable.inducing_variable_list[1].Z"]),
                ]
            )
            use_temp_dim = \
                params.gp_model_stor["model_params"][".inducing_variable.inducing_variable_list[0].Z"].shape[1] == 3
        else:
            inducing_variable = params.gp_model_stor["model_params"][".inducing_variable.Z"]
            use_temp_dim = inducing_variable.shape[1] == 3

        assert inducing_variable is not None, \
            "Writing an old model only works with SVGP"

        self.pr = params
        # Write the saved scalers which is needed for scaling/inverse scaling during prediction
        self.x_temp_scaler = params.gp_model_stor["x_temp_scaler"]
        self.x_spat_scaler = params.gp_model_stor["x_spat_scaler"]
        self.y_scaler = params.gp_model_stor["y_scaler"]
        # Recreate the model based on the parameters
        likelihood = self.likelihood(lik=self.pr.likelihood)
        kernel = self.kernel(use_temp_dim=use_temp_dim, use_nearest_pd=self.pr.run_with_nearest_pd, init=False)
        model = self.get_model(x=None, y=None, kern=kernel, likelihood=likelihood, inducing_points=inducing_variable)
        gpflow.utilities.multiple_assign(model, params.gp_model_stor["model_params"])
        self.model = model

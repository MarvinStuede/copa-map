"""Module for the Bresenham Kernel based GP Model"""
import gpflow
from dataclasses import dataclass
from cmr_people_gp.kernel.kernel_grid import KernelGrid
import cmr_people_gp.model.model_utils as mu
from cmr_people_gp.kernel.kernel_matern_map import MaternMap
from cmr_people_gp.model.Optimization import Optimization
from cmr_people_gp.model.CoPAMapBase import CoPAMapBase, CoPAMapParams
from cmr_people_gp.util.occ_grid import OccGrid


@dataclass
class PeopleGPBresenhamParams(CoPAMapParams):
    """Model parameters additionally needed for this kernel"""
    # Occupancy map of the environment to calculate covariance matrix
    occ_map: OccGrid = OccGrid(None, width=1, height=1)
    pred_store_kernel_map: bool = True
    # Resolution of the kernel grid in meters
    # Can be used to downscale the map to decrease the number of cells necessary to check for Bresenham kernel
    resolution_kernel_map: float = 0.1
    # Number of borders around walls for inflation.
    num_borders_kernel_map: int = 2
    # Minimal value of adjustment matrix
    min_adj_val: float = 1e-3
    only_predict: bool = False


class CoPAMapBresenham(CoPAMapBase):
    """
    Bresenham Kernel based Gaussian Process Model

    The Bresenham kernel is a non-stationary kernel which incorporates environmental information into posterior
    calculation.
    """
    def __init__(self, params : PeopleGPBresenhamParams):
        """
        Constructor

        Args:
            params: GP Model parameters
        """
        super(CoPAMapBresenham, self).__init__(params)
        self.pr = params
        self.kernel_map = None

    def _kernel_spatial_dim(self, variance_max=1., use_nearest_pd=False, use_base=None):
        """Kernel in spatial dimension"""
        if use_base is None:
            use_base = self.pr.only_predict
        if use_base:
            min_adj_val = 1.0
        else:
            min_adj_val = self.pr.min_adj_val

        if self.kernel_map is None:
            self.kernel_map = KernelGrid(base_occ_map=self.pr.occ_map, digitize_size=self.pr.resolution_kernel_map,
                                         num_of_borders=self.pr.num_borders_kernel_map)
        # Possibly activate log messages when calculating the Bresenham kernel
        display_information = False
        # Set initial large hyper-parameters to use when calling pre_init_kernel_map() to reduce the number of
        # recalculations of the adjustment matrix during optimization.

        kern = MaternMap(active_dims=[0, 1],
                         occ_map=self.pr.occ_map, kernel_map=self.kernel_map,
                         nearest_pd=use_nearest_pd, display_information=display_information,
                         data_scaler=self.x_spat_scaler, min_adjustment_val=min_adj_val)

        # Increases stability (the variance may be increased in the course of troubleshooting during optimization)
        kern = kern + gpflow.kernels.White(active_dims=[0, 1], variance=1e-9)

        # Without limitation variance becomes infinitely high with less training data
        mu.get_kernel_instance(kern, "Mat52_Map").variance \
            = mu.bounded_hyperparameter(**self.pr.kern_spat_var)
        mu.get_kernel_instance(kern, "Mat52_Map").lengthscales \
            = mu.bounded_hyperparameter(**self.pr.kern_spat_lengthsc)

        return kern

    def _set_optimization(self, log_dir, rmse_clbl):
        return Optimization(method="bresenham", max_duration_for_opt=self.pr.opt_max_dur,
                            max_iters_for_opt=self.pr.opt_max_iters, seed=self.pr.seed, log_dir=log_dir,
                            rmse_clbl=rmse_clbl)

    def predict(self, Xt):
        """
        Predict the models output at given positions

        Args:
            Xt: inputs (n x 3) location and time

        Returns:
            Model output
            Data indexes
        """
        def set_kernel(min_value, nearest_pd):
            mu.get_kernel_instance(self._get_predict_model().kernel, "Mat52_Map").min_value = min_value
            mu.get_kernel_instance(self._get_predict_model().kernel, "Mat52_Map").nearest_pd = nearest_pd

        if self.pr.only_predict:
            set_kernel(self.pr.min_adj_val, True)

        m, v = super().predict(Xt)

        if self.pr.only_predict:
            set_kernel(1.0, False)

        return m, v

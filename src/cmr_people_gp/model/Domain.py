"""Module for domains, used for data subdivision into multiple GP"""

import numpy as np
from cmr_people_gp.model.CoPAMapBase import CoPAMapBase, CoPAMapParams
from cmr_people_gp.model.CoPAMapBresenham import CoPAMapBresenham, PeopleGPBresenhamParams
from cmr_people_gp.model.PeopleModel import ModelInterface
from cmr_people_gp.util.grid import Grid
from typing import Type, List
# from termcolor import colored
from abc import abstractmethod
from cmr_people_gp.plots.plot_domain_allocation import PlotDomains
from cmr_people_gp.util import util as ut
import os
from datetime import datetime
from fractions import gcd


def get_model_by_params(model_params):
    """Based on parameters, return the corresponding Model"""
    if type(model_params) is CoPAMapParams:
        model = CoPAMapBase(params=model_params)
    elif type(model_params) is PeopleGPBresenhamParams:
        model = CoPAMapBresenham(params=model_params)
    else:
        raise NotImplementedError("PeopleModel not implemented")
    return model


class Domain(ModelInterface):
    """
    Represents an area in an environment that uses a subset of training data

    Multiple domains can be used to subdivide the enviroment into independent models, each with different training sets.
    This can be useful to parallelize training and reduce training complexity. Each domain contains
    e.g. a Gaussian Process model
    """

    def __init__(self, model_params: Type[CoPAMapParams]):
        """
        Constructor

        Args:
            model_params: Gaussian Process model parameters, either base class or a class inherited from PeopleGPparams
        """
        self.model = get_model_by_params(model_params)
        self.max_dur_opt = None
        self.opt_model = None
        self.stored_model = None

    def _filter(self, X, Y=None, use_ext=False):
        """Determines data which is within a domain

        Args:
            X: Position data, which are filtered
            Y: List of data which are filtered via the associated x-values (If None, X is used)

        Returns:
            Tuple of filtered data
        """
        # Indices of x-data within the domain
        index = self.idx_in_domain(X, use_ext)
        # Return the y-data (or x-data itself) filtered on the basis of the indices
        return tuple([data[index, :] for data in Y]) if isinstance(Y, list) else X[index, :]

    @abstractmethod
    def idx_in_domain(self, X, use_ext):
        """
        Implement this abstract method to return mask of input locations that lie in domain

        Args:
            X: nd array of dim [2xn] of input locations
            use_ext: use extended area (can be used for training)

        Returns:
            Vector of dimension [1xn] where True refers to a domain inlier at this location
        """
        pass

    def set_max_opt_dur(self, max_dur):
        """Set the maximum optimization duration for this domain"""
        self.max_dur_opt = max_dur

    def is_trained(self):
        """Model was trained"""
        if hasattr(self.model, 'optimization'):
            has_opt = hasattr(self.model.optimization, 'model') and self.model.optimization.model is not None
        else:
            has_opt = False
        has_model = hasattr(self.model, 'model') and self.model.model is not None
        return has_opt or has_model

    def learn(self, X, Y, Z, log_subfolder="", **kwargs):
        """Learn this domain"""
        Xf, Yf = self._filter(X=X, Y=[X, Y], use_ext=True)

        if Z is None:
            use_inducing = False
        else:
            Zf = (self._filter(Z[0]), self._filter(Z[1]))
            use_inducing = True

        # Check if training data are within domain
        data_available = Xf.shape[0] > 0 if not use_inducing else Xf.shape[0] > 0 and Zf[0].shape[0] > 0
        if data_available:
            log_dir = os.path.join(ut.abs_path(), "logs", log_subfolder)
            self.model.learn(Xf, Yf, Zf, log_dir=log_dir, **kwargs)
        else:
            print("No data available")

    def predict(self, Xt):
        """
        Predict Rates in domains

        Args:
            X: Input locations/time

        Returns:
            mean, cov and locations
        """
        if not self.is_trained():
            raise ValueError("No trained model available. Call learn(...) first")
        return self.model.predict(self._filter(Xt, use_ext=False))

    def get_model_params(self):
        """
        Return the model parameters

        Returns:
            Instance of the ModelParams class or subclass
        """
        return self.model.get_model_params()

    def write_model_params(self, params):
        """
        Write the model parameters

        Args:
            params: Instance of the ModelParams class or subclass

        """
        self.model.write_model_params(params)


class DomainRectangle(Domain):
    """Class to represent a Rectangular Domain"""

    def __init__(self, *args, **kwargs):
        """Constructor"""
        super(DomainRectangle, self).__init__(*args, **kwargs)
        self.grid = None

    def idx_in_domain(self, X, use_ext):
        """
        Get mask of input Vector X that lie in domain area

        Args:
            X: nd array of dim [2xn] of input locations
            use_ext: use extended area (can be used for training)

        Returns:
            Vector of dimension [1xn] where True refers to a domain inlier at this location
        """
        if self.grid is None:
            raise NameError("Coordinates of domain not defined. Call set_coordinates(...) first")

        return self.grid.pos_in_grid(X[:, :2], extend=self.ext_size if use_ext else 0)

    def set_coordinates(self, origin, rotation, width, height, extend_sz):
        """
        Set the coordinates of the domain

        Uses lower left and upper right corner
        Args:

            extend_sz: Size in meters to extend the domain for training

        """
        self.grid = Grid(width=width, height=height, resolution=gcd(height, width), origin=origin, rotation=rotation)
        self.ext_size = extend_sz


class Domains(ModelInterface):
    """Abstract class to represent an arrangement of domains"""

    def __init__(self, model_params: Type[CoPAMapParams]):
        """Constructor"""
        self.model_params = model_params

    @abstractmethod
    def _create_gp_models(self, X):
        """Implement this method to define the valid area of GP models"""
        pass

    @abstractmethod
    def _plot_domains(self):
        pass

    @abstractmethod
    def _plot_domain_status(self, i_d, status):
        pass

    def learn(self, X, Y, Z, domains: List[int] = None, plot=True, log_subfolder=None, **kwargs):
        """
        Learn the GPs of the specified domains

        Args:
            X: Input locations
            Y: Observations corresponding to input locations
            Z: Inducing points
            domains: List of domain numbers to learn
            plot: If true, plot the domains on a map and current optimization status
            log_subfolder: Path to subfolder where tensorboard logs should be stored
        """
        if plot:
            self._plot_domains()
        if domains is None:
            # Use all
            domains = np.arange(0, len(self.domains))
        if log_subfolder is None:
            set_name = True
        else:
            set_name = False
        # Datetime for log folder
        now = datetime.now()
        for i_d in domains:
            domain = self.domains[i_d]
            # input_data = domain.filter_and_set(X, Yg, Z, use_inducing=gp_params.use_inducing, use_ext=True)
            # if input_data is not None:
            if plot:
                self._plot_domain_status(i_d, 0)

            if set_name:
                log_subfolder = os.path.join(now.strftime("%Y%m%d-%H%M%S"), "dom-" + str(i_d))

            domain.learn(X, Y, Z, log_subfolder=log_subfolder, **kwargs)
            if plot:
                self._plot_domain_status(i_d, 1)

    def predict(self, Xt):
        """
        Predict Rates in domains

        Args:
            X: Input locations and time

        Returns:
            mean, cov and locations
        """
        # Initialize non-predicated values
        mean = (np.ones(Xt.shape[0]) * (-1)).reshape(-1, 1)
        cov = (np.ones(Xt.shape[0]) * (-1)).reshape(-1, 1)

        for domain in self.domains:
            idx = domain.idx_in_domain(Xt, use_ext=False)
            if domain.is_trained() and ~np.all(~idx):
                mean_dom, cov_dom = domain.predict(Xt)
                mean[idx] = mean_dom
                cov[idx] = cov_dom

        return mean, cov

    def get_model_params(self, domains: List[int] = None):
        """
        Return the model parameters

        Returns:
            Instance of the ModelParams class or subclass
        """
        params = []
        if domains is None:
            # Use all
            domains = np.arange(0, len(self.domains))
        for i_d in domains:
            params.append(self.domains[i_d].get_model_params())
        return params

    def write_model_params(self, params, domains: List[int] = None):
        """
        Write the model parameters

        Args:
            params: Instance of the ModelParams class or subclass
            domains: List of domains where parameters will be written to

        """
        if domains is not None:
            assert isinstance(params, list), "Params must be list, if list of domains given"
            assert len(domains) == len(params), "Param list length must be equal to domain list length"
        if domains is None:
            # Use all
            domains = np.arange(0, len(self.domains))
            params = [params] * len(self.domains)
        for i_d in domains:
            self.domains[i_d].write_model_params(params[i_d])


class DomainGrid(Domains):
    """Class to represent a Grid of rectangular Domains"""

    def __init__(self, occ_map, domain_size, extend_size, *args, **kwargs):
        """
        Constructor

        Args:
            occ_map: Occupancy map representing the environment
            domain_size: Domain size in meters
            extend_size: Length in meters to extend the domain size for training
        """
        super(DomainGrid, self).__init__(*args, **kwargs)
        self.occ_map = occ_map
        self.domain_sz = domain_size
        self.extend_sz = extend_size
        self.domains = None
        self.plotter = None
        g_width = np.max([self.occ_map.width, domain_size])
        g_height = np.max([self.occ_map.height, domain_size])
        self.grid = Grid(width=g_width, height=g_height,
                         resolution=domain_size, origin=self.occ_map.orig, rotation=self.occ_map.rotation)
        self._domains_by_map()
        self._create_gp_models(self.model_params)

    def _domains_by_map(self):
        """Divides the map into local GP"""
        # x_steps = math.ceil(self.occ_map.width / self.domain_sz)
        # y_steps = math.ceil(self.occ_map.height / self.domain_sz)
        print("Map divided into {} domains of edge length {}m".format(self.grid.elements_y * self.grid.elements_y,
                                                                      self.domain_sz))

        coord_x = np.arange(0, self.grid.elements_x) * self.grid.resolution
        coord_y = np.arange(0, self.grid.elements_y) * self.grid.resolution
        mx, my = np.meshgrid(coord_x, coord_y)
        self.ll = np.vstack([mx.reshape(-1), my.reshape(-1)]).T
        ll_t = self.grid.tf_from(self.ll)
        lr_t = self.grid.tf_from(self.ll + np.array([self.domain_sz, 0]))
        ul_t = self.grid.tf_from(self.ll + np.array([0, self.domain_sz]))
        ur_t = self.grid.tf_from(self.ll + self.domain_sz)
        self.plotter = PlotDomains(self.domain_sz, self.occ_map, ll_t, lr_t, ul_t, ur_t)

    def _create_gp_models(self, model_params):
        def create_domain(ll):
            domain = DomainRectangle(model_params=model_params)
            orig = np.hstack([self.grid.tf_from(ll), 0.])
            rotation = self.occ_map.rotation
            width = self.grid.resolution
            domain.set_coordinates(origin=orig, rotation=rotation, width=width, height=width, extend_sz=self.extend_sz)
            return domain

        self.domains = list(map(create_domain, self.ll))

    def _plot_domains(self):
        self.plotter.plot_local_domains()

    def _plot_domain_status(self, i_d, status):
        self.plotter.set_status(i_d, status)

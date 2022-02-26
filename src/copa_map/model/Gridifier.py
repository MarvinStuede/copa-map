"""Module to preprocess detection data and robot poses to create usable input for ML model"""
from copa_map.util import hist_grid, rate_grid, util

import numpy as np
from dataclasses import dataclass
from termcolor import colored
import os
from matplotlib import pyplot as plt
import pickle
from matplotlib.widgets import Slider
from copa_map.util.fov import FOV
from copa_map.util.occ_grid import OccGrid
from copy import copy
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class GridParams:
    """Dataclass for model parameters"""
    # Origin of the grid in world coordinates
    origin: list
    # Rotation of the grid in rad
    rotation: float = 0.
    # Height of the grid in m
    height: float = 70.0
    # Width of the grid in m
    width: float = 70.0
    # Resolution of the histogram cells in meters
    cell_resolution: float = 1.0
    # Ratio of cells with zero counts to keep
    zero_ratio: float = 1.0
    # Ratio of cells with small rate to keep
    small_rate_ratio: float = 1.0
    # Ratio of number of all cells to number of inducing points (alpha)
    inducing_ratio: float = 0.02
    # Seed for use in random selections
    bin_seed: float = None
    # 2D-Kmeans, 3D-Kmeans
    inducing_method: str = "2D-KMeans"
    # Minimum rate to set because gamma distribution requires values greater than zero
    rate_min: float = 1e-5
    # Minimum observation time in seconds for use cells in training data (reduces instability).
    observation_time_min: float = 20.
    # Bin size in seconds
    bin_size: float = 60. * 60.
    # Normalize rate to bin timestamps
    # Rate will then correspond to count per bin_size
    normalize_to_bin: bool = True
    # Outlier contamination
    # Remove this ratio of total data points as outliers
    outlier_contam: float = 0.003
    # Instead of removing outliers by masking, set the value to min value
    set_outlier_to_min: bool = False
    # When creating the cells based on robot positions, incorporate the occupancy grid map to exclude cells that
    # were not visible because of obstructions
    fov_use_map: bool = True
    # Save the input data to the grid object. If this class is pickled via the respective method, the original
    # input data can then be accessed later
    save_raw_data: bool = True


class Gridifier:
    """Class to preprocess the people detection and robot data, creating a grid like setup with counts and rates"""

    def __init__(self, occ_map: OccGrid, fov: FOV, params: GridParams, create_gt=False):
        """
        Constructor

        Args:
            occ_map: Occupancy map to calculate observation durations of different areas of the environment.
                     Obstacles (black areas in the map) will be regarded as impenetrable by the robot so that the
                    observation duration behind obstacles is not increased
            fov:    FOV object which represents the field of view of the robot (e.g. a Circle with radius)
            params: GridParams parametrizing the grid
            create_gt: If true the observation durations and obstacles will be ignored. Still, only the cells are
                       considered that lie within the maximum range of the robot's FOV. But every visible cell will have
                       maximum observation duration.
        """
        self.occ_map = occ_map
        self.fov = fov
        self.params = params
        self.obv_duration = None
        self.mask_fov = None
        self.X_data = None
        self.Y_data = None
        self.Y_data_all = None
        self.df_data = None
        self.df_rob = None
        self.num_bins = None
        # Scales the timestamps to bin counts
        if self.params.normalize_to_bin:
            self.scale_seconds = self.params.bin_size
            self.scale_seconds_text = str(self.scale_seconds / 60) + "min"
        else:
            self.scale_seconds = 60.
            self.scale_seconds_text = {self.scale_seconds == 60: "min", self.scale_seconds == 3600: "hour"}[True]
        # Number of cpu cores
        self.cpu_count = util.get_cpu_count()
        self.create_gt = create_gt

    def to_file(self, path):
        """
        Write the instance of this class to a pickle

        Args:
            path: Absolute path

        """
        assert self._data_read()
        data_dict = {'X': self.X_data, 'Y': self.Y_data, 'Yg': self.Y_data_all,
                     'param': self.params, 'occ_map': self.occ_map, 'fov': self.fov,
                     'grids': self.grids, 'timestamps': self.timestamps, 'data': self.df_data}
        print("Writing gridifier data to " + str(path))
        pickle.dump(data_dict, open(path, "wb"))

    def output_to_text(self, path):
        """
        Write the output values (X, Y, Z) to csv files.

        Args:
            path: Path with csv suffix
        """
        assert self._data_read()
        sX = self.X_data
        sY = self.Y_data_all[:, 0].reshape(-1, 1)
        sY[sY == 1e-3] = 1e-6
        pdXY = pd.DataFrame(data=np.hstack([sX, sY]), columns=["x1", "x2", "t", "y"])
        pdXY.to_csv(path, index=False)

    @classmethod
    def from_file(cls, path):
        """
        Read an instance of this class from a pickle

        Args:
            path: Absolute path to file
        """
        assert os.path.isfile(path)
        print("Restoring gridifier data from " + str(path))
        data_dict = pickle.load(open(path, "rb"))
        inst = cls(data_dict['occ_map'], data_dict['fov'], data_dict['param'])
        inst.X_data = data_dict['X']
        inst.Y_data = data_dict['Y']
        inst.Y_data_all = data_dict['Yg']
        inst.grids = data_dict['grids']
        inst.timestamps = data_dict['timestamps']
        inst.df_data = data_dict['data'] if 'data' in data_dict else None
        return inst

    def get_grid(self, num=-1):
        """Return the specified grid for a bin number"""
        return self.grids[num]

    def get_input_points(self):
        """Return the lattice like input points"""
        self._chk_data_set(self.X_data)
        return self.X_data

    def get_count(self):
        """Return the counts at these points"""
        self._chk_data_set(self.Y_data)
        return self.Y_data

    def get_observations(self):
        """Return the rate, std dev, counts, observation duration"""
        self._chk_data_set(self.Y_data_all)
        return self.Y_data_all

    def _chk_data_set(self, dat):
        if dat is None:
            raise NameError("No gridified data available. Call setup_data(..) first.")

    def setup_data(self, df_data: pd.DataFrame, df_rob: pd.DataFrame = pd.DataFrame({'': []})):
        """
        Set the data to the gridifier.

        Based on detections, robot path and parameters given in constructor constructs lattice like input points
        and observations that can be used as input for machine learning models

        Args:
            df_data:      Pandas dataframe containing the people data to use. Expected format are columns named
                          (pos_x, pos_y, tidx_bin, t) where tidx_bin refers to the time index of the time bin.
            df_rob:       Pandas dataframe containing the robot positions. Expected format are columns named
                          (robot_x, robot_y, delta_t, t). delta_t refers to the dwell time at each pose.
        """
        # For each timestamp, grids are created in a loop, which extend the following training data arrays
        self.X_data = np.array([]).reshape(0, df_data.shape[1] - 1)  # Minus 1 because of t column
        # self.vis_data = np.array([]).reshape(0, X_detections[0].shape[1])
        self.Y_data = np.array([]).reshape(0, 1)
        self.Y_data_all = np.array([]).reshape(0, 4)

        #
        if self.params.save_raw_data:
            self.df_data = df_data
            self.df_rob = df_rob

        self.grids = list()
        self.timestamps = list()

        # Delta_t is given
        if not df_rob.empty and 'delta_t' in df_rob:
            self.grid = rate_grid.RateGrid(width=self.params.width, height=self.params.height,
                                           resolution=self.params.cell_resolution, origin=self.params.origin,
                                           min_rate=self.params.rate_min, scale_seconds=self.scale_seconds,
                                           rotation=self.params.rotation)
        else:
            # Histogram to represent person counts in a grid-like fashion
            self.grid = hist_grid.HistGrid(width=self.params.width, height=self.params.height,
                                           resolution=self.params.cell_resolution, origin=self.params.origin,
                                           rotation=self.params.rotation)

        t_arr = df_rob.tidx_bin.unique() if not df_rob.empty else df_data.tidx_bin.unique()

        for idx, tidx in enumerate(t_arr):
            # self.grid = copy(grid_orig)
            df_sub_data = df_data.loc[df_data.tidx_bin == tidx]

            print("\nComputing grid {}/{}".format(idx + 1, len(t_arr)))

            print("Timespan Data: " + str(df_sub_data.t.min()) + " -- " + str(df_sub_data.t.max()))

            if not df_rob.empty:
                df_sub_rob = df_rob.loc[df_rob.tidx_bin == tidx]
                print("Timespan Robot: " + str(df_sub_rob.t.min()) + " -- " + str(df_sub_rob.t.max()))
            X = df_sub_data.drop(columns=['t']).to_numpy()

            R_bin, delta_t, new_path = self._bin_rob_pos_and_dwell(tidx, df_rob)

            X_data, Y_data, Y_data_all = \
                self._bin_prepare_data(self.params, X, R_bin, delta_t, new_path,
                                       self.fov, self.occ_map, tidx)

            # Add training data of the timestamp to remaining training data
            self.X_data = np.vstack((self.X_data, X_data))
            # TODO: Replace member variable
            # ##self.vis_data = np.vstack((self.vis_data, vis_data))
            self.Y_data = np.vstack((self.Y_data, Y_data))
            self.Y_data_all = np.vstack((self.Y_data_all, Y_data_all)) if Y_data_all is not None else None
            self.grids.append(copy(self.grid))
            self.timestamps.append(df_sub_data.t.min())
        if self.params.outlier_contam > 0.0:
            self.X_data, self.Y_data_all, self.Y_data = \
                self._mask_outliers(self.X_data, self.Y_data_all, self.Y_data)

    def _mask_outliers(self, X, Yg, Y):
        """Removes outliers from the final data"""
        Yc = IsolationForest(contamination=self.params.outlier_contam).fit_predict(Yg[:, 0].reshape(-1, 1))
        rX = X[Yc == 1]
        rY = Y[Yc == 1]
        rYg = Yg[Yc == 1]
        util.logger().info("Removed " + str(Y[Yc == -1].shape[0]) + " outliers from " + str(Y.shape[0]) +
                           " data points. Old max: " + str(Y[:, 0].max()) + ", mean: " + str(Y[:, 0].mean()) +
                           ", New max: " + str(rY[:, 0].max()) + ", mean: " + str(rY[:, 0].mean()))
        ts_with_outl = np.unique(X[Yc == -1][:, 2])

        def mask_grid(ti, Yc):
            # Get index of cells which should be masked for this timestamp
            ti = int(ti)
            poses = self.grids[ti].tf_to(X[(X[:, 2] == ti) & (Yc == -1)][:, :2])
            ind_mask = self.grids[ti].index_from_pos(poses)
            if not self.params.set_outlier_to_min:
                # Mask the rate array
                self.grids[ti].rate.mask[ind_mask[:, 0], ind_mask[:, 1]] = True
            else:
                # Instead of masking, set the value to zero (or min value) if parameter is set
                self.grids[ti].rate[ind_mask[:, 0], ind_mask[:, 1]] = self.params.rate_min

        # np.apply_along_axis(mask_grid, arr=ts_with_outl.reshape(-1, 1), axis=1)
        # map(mask_grid, ts_with_outl.reshape(-1, 1))
        [mask_grid(ti, Yc) for ti in ts_with_outl.reshape(-1, 1)]
        if self.params.set_outlier_to_min:
            return X, Yg, Y
        else:
            return rX, rYg, rY

    def _data_read(self):
        return self.X_data is not None and self.Y_data is not None

    def _get_fov_mask(self, R, fov, occ_map):
        if R is None:
            mask_fov = ~self.grid.empty_mask()
        else:
            # Using the field of view and robot poses, determine which cells were visible for the robot
            print("Compute mask_fov...")
            mask_fov = fov.path_mask(poses=R, grid=self.grid, occ_map=occ_map if self.params.fov_use_map else None,
                                     cpu_count=self.cpu_count)
            assert ~np.all(~mask_fov), "Visibility area contains no cells"
        return mask_fov

    def _bin_prepare_data(self, params, X_bin, R_bin, delta_t, new_path, fov, occ_map, timestamp):

        # Histogram to represent person counts in a grid-like fashion
        self.grid.set_counts(X_bin[:, :2])

        if new_path:
            self.obv_duration = None
            self.mask_fov = self._get_fov_mask(R=R_bin, fov=fov,
                                               occ_map=occ_map)
        elif new_path is None:
            self.mask_fov = None
        if params.bin_seed is not None:
            # Modify the seed based on the timestamp (or how many grid were already saved, which is the same)
            # This keeps different methods comparable, but avoids that samples are distributed equally for consecutive
            # timesteps
            seed = len(self.grids) * params.bin_seed
        else:
            seed = None

        self.counts = self.grid.masked_counts(ratio=params.zero_ratio,
                                              mask=~self.mask_fov if self.mask_fov is not None else None,
                                              seed=seed)

        # vis_data = self.grid.get_centers(as_3d=True)[self.mask_fov]

        if isinstance(self.grid, rate_grid.RateGrid):
            self.grid.set_by_path(R_bin, delta_t, fov, occ_map, min_obs_time=params.observation_time_min,
                                  create_gt=self.create_gt, mask_fov=self.mask_fov, new_path=new_path)
            self.grid.mask_small_rate(rate_min=params.rate_min, ratio=params.small_rate_ratio)
            Y_data_all = \
                self.grid.get_stacked(norm_obs_dur=self.params.bin_size if self.params.normalize_to_bin else None)
            print(colored("Created grid with " + str(params.cell_resolution) + "m resolution", "green"))

            def print_vars(name, matrix, unit):
                print(colored("Max " + name + ": " + f"{matrix.max():.2f}" + " " + unit +
                              ", Min " + name + ": " + f"{matrix.min():.2f}" + " " + unit +
                              ", Mean " + name + ": " + f"{matrix.mean():.2f}" + " " + unit, "green"))

            print_vars("Counts", self.grid.counts_masked[~self.grid.rate.mask], unit="people")
            print_vars("Obv duration", self.grid.obv_duration, unit=self.scale_seconds_text)
            print_vars("Rate", self.grid.rate, unit="people/" + self.scale_seconds_text)
        else:
            Y_data_all = None
            print(colored("Created grid with " + str(params.cell_resolution)
                          + "m resolution, max counts per cell: " + str(np.max(self.counts)), "green"))

        X_data, Y_data = self._bin_input_for_gp(self.counts)

        X_data, Y_data, Y_data_all = \
            self._bin_drop_outside_map(occ_map, X_data, Y_data, Y_data_all)

        # Add respective timestamp to training data if it is known
        def add_timestamp_if_not_none(arr, timestamp):
            if arr is not None:
                return np.hstack((arr, np.ones((arr.shape[0], 1)) * timestamp))
            else:
                return arr

        if timestamp is not None:
            X_data = add_timestamp_if_not_none(X_data, timestamp)

        return X_data, Y_data, Y_data_all

    def _bin_drop_outside_map(self, occ_map, X_data, Y_data, Y_data_all):
        """Drop all data outside of the occupancy map"""
        def keep_ind(data):
            if data is None:
                return []
            data_t = self.occ_map.tf_to(data)
            return (data_t[:, 0] > 0) & (data_t[:, 1] > 0) & (data_t[:, 0] <= occ_map.width) & \
                   (data_t[:, 1] <= occ_map.height)

        keep_X = keep_ind(X_data)
        X_data = X_data[keep_X] if X_data is not None else None

        if Y_data_all is not None:
            Y_data_all = Y_data_all[keep_X]
        Y_data = Y_data[keep_X]
        return X_data, Y_data, Y_data_all

    def _bin_input_for_gp(self, counts):

        if isinstance(self.grid, rate_grid.RateGrid):
            # If observation duration known, do not use cells with low observation time and rate
            mask = np.logical_or(counts.mask, self.grid.rate.mask)
        else:
            # Masked array representing the counts in each cell, masked by the visibility area
            mask = counts.mask
        # Input data for the GP
        X_data = self.grid.get_centers(as_3d=True)[~mask]
        # Counting data as the outcome
        Y_data = counts.data[~mask].ravel(order='F').reshape(-1, 1)

        return X_data, Y_data

    def _bin_rob_pos_and_dwell(self, tidx, df_rob: pd.DataFrame):
        """Get the robot path and dwell times during a specific bin.

        Also returns a variable that indicates if this exact path was returned for the last bin,
        to avoid multiple calculations of the same path
        """
        try:
            # Positions of the robot path
            df_bin = df_rob.loc[(df_rob.tidx_bin == tidx) & (df_rob.delta_t > 0)].drop(columns=['t'])
            R = df_bin[['robot_x', 'robot_y']].to_numpy()
            # To the positions associated dwell times
            delta_t = df_bin[['delta_t']].to_numpy()
            delta_t = delta_t.reshape(delta_t.shape[0], )
            tidxs = df_rob.tidx_bin.unique()
            tidx_before = (np.argwhere(tidxs == tidx) - 1)[0][0]
            # If the same simulative robot path is used for all timestamps, some computations can be reused
            if tidx == df_rob.tidx_bin.unique()[0]:
                new_path = True
            else:
                # If all data is the same as in the bin before, its the same path
                arr_bef = df_rob[df_rob.tidx_bin == tidx_before][['robot_x', 'robot_y', 'delta_t']].values
                arr_now = df_rob[df_rob.tidx_bin == tidx][['robot_x', 'robot_y', 'delta_t']].values
                if np.array_equal(arr_bef, arr_now):
                    new_path = False
                else:
                    new_path = True
        except Exception as e:
            print(colored("Robot path not found: {}".format(e), "red"))
            R = delta_t = new_path = None
        return R, delta_t, new_path

    def plot(self, figurename="Gridifier"):  # pragma: no cover
        """Plots the gridified data to a 2x2 plot window"""
        assert self.Y_data_all is not None, "Plot only works with Rate data"
        """Plot the counts, observation duration, variance and rate"""
        fig, axs = plt.subplots(2, 2, figsize=(18, 14), sharex=True, sharey=True, num=figurename)

        self.occ_map.plot(axs[0, 0], transparent=True, zorder=2, black_thresh=200)
        self.occ_map.plot(axs[0, 1], transparent=True, zorder=2, black_thresh=200)
        self.occ_map.plot(axs[1, 0], transparent=True, zorder=2, black_thresh=200)
        self.occ_map.plot(axs[1, 1], transparent=True, zorder=2, black_thresh=200)

        axs[0, 0].set_ylim((self.params.origin[1], self.params.origin[1] + self.params.height))
        axs[0, 0].set_xlim((self.params.origin[0], self.params.origin[0] + self.params.width))
        axs[0, 0].set_title("Data: People counts")
        grid_edges = self.grids[0].get_edges(as_3d=True)
        # mesh_c, mesh_m = self.grids[0].plot_counts(axs[0, 0], masked=True, vmin=0,
        #                                            vmax=(self.Y_data_all[:, 0] * self.Y_data_all[:, 1]).max())
        counts_max = 120
        mesh_c, mesh_m = self.grids[0].plot_counts(axs[0, 0], masked=True, vmin=0,
                                                   vmax=counts_max)

        # def get_bin_data(num):
        #     Z = self.Z_data[self.Z_data[:, 2] == num][:, :2]
        #     return Z
        #
        # Z = get_bin_data(0)

        def plot_dat(axs, ma, vmax=1):
            return axs.pcolormesh(grid_edges[:, :, 0], grid_edges[:, :, 1], ma,
                                  shading='auto', cmap='jet', alpha=0.5, vmin=0, vmax=vmax)

        rate_max = max(np.max(grid.rate) for grid in self.grids)
        obv_mesh = plot_dat(axs[0, 1], self.grids[0].obv_duration, vmax=self.grids[0].obv_duration.max())
        var_mesh = plot_dat(axs[1, 0], self.grids[0].stddev, vmax=self.Y_data_all[:, 1].max() * 0.75)
        rat_mesh = plot_dat(axs[1, 1], self.grids[0].rate, vmax=rate_max)

        # z_scatter = axs[1, 1].scatter(Z[:, 0], Z[:, 1], marker='o', color='black')

        axs[0, 1].set_title("Observation duration (filtered)")
        axs[1, 0].set_title("Std deviation (filtered) (Deprecated)")
        axs[1, 1].set_title("Rate (filtered)")
        fig.colorbar(mesh_c, ax=axs[0, 0])
        fig.colorbar(obv_mesh, ax=axs[0, 1])
        fig.colorbar(var_mesh, ax=axs[1, 0])
        fig.colorbar(rat_mesh, ax=axs[1, 1])

        ax_sl = fig.add_axes([0.2, 0.05, 0.75, 0.03])

        def update(val):
            i = int(slider.val)
            # Z = get_bin_data(i)
            obv_mesh.set_array(self.grids[i].obv_duration.ravel())
            var_mesh.set_array(self.grids[i].stddev.ravel())
            rat_mesh.set_array(self.grids[i].rate.ravel())
            mesh_c.set_array(self.grids[i].counts.ravel())
            mesh_m.set_array(~self.grids[i].counts_masked.mask.ravel())
            # z_scatter.set_offsets(np.vstack([Z[:, 0], Z[:, 1]]).T)
            ts = self.timestamps[i]
            if pd.isnull(ts):
                strtime = ""
            else:
                strtime = ts.strftime('%d-%m-%Y: %H:%M')
            axs[0, 0].set_title("Data: People counts. Timestamp: {}, {}".format(i, strtime))
            fig.canvas.draw()
            fig.canvas.flush_events()

        # ax_sl.set_xlim(0, len(self.grids) - 1)
        slider = Slider(ax_sl, 'Timestamp %i' % 1, 0, len(self.grids) - 1,
                        valinit=0, valfmt='%i')
        slider.on_changed(update)
        plt.show()

"""
Module for Gamma grid

Contains a class derived from Histogram grid to  consider observation time of cells and variance
"""

from copa_map.util import hist_grid, util
import numpy as np
import multiprocessing as mp
from copy import copy


def runner_obv_duration(grid, fov, poses, occ_map, d_t, mask):
    """Worker function used in get_cell_obv_duration for multiprocessing

    Determines all visible cells for each robot position.
    'd_t' defines the observation durations, which are summed up for all 'poses'.

    """

    def durs_for_pose(row):
        mask_pos = fov.visible_cells(pos=row[0:2], grid=grid, occ_map=occ_map, mask=None if mask is None else ~mask)
        return row[2] * mask_pos

    conc_pos_dur = np.hstack([poses, np.array([d_t]).T])
    durs = np.apply_along_axis(durs_for_pose, axis=1, arr=conc_pos_dur)
    dur_cum = np.sum(durs, axis=0)
    return dur_cum


class RateGrid(hist_grid.HistGrid):
    """
    Rate grid to represent rate of each cell

    By considering the detection duration and count of cells, the expected rate and variance can be calculated.
    The rate grid can be used to calculate observation times of cells based on a robot path and detection are and
    resulting variance
    """

    def __init__(self, min_rate=None, cpu_count=None, scale_seconds=60., *args, **kwargs):
        """Constructor"""
        super(RateGrid, self).__init__(*args, **kwargs)
        self.obv_duration = None
        self.rate = None
        self.stddev = None
        self.counts_masked = None
        self.path = None
        self.d_t = None
        self.scale_seconds = scale_seconds
        self.min_rate = min_rate
        self.min_obs_time = None

        if cpu_count is None:
            self.cpu_count = util.get_cpu_count()
        else:
            self.cpu_count = cpu_count

    def set_by_path(self, path, d_t, fov, occ_map, mask_fov, min_obs_time=0., lazy=True, create_gt=False,
                    new_path=True):
        """
        Set the class parameters based on a path

        Set the class parameters based on a path (given as array of xy positions), dwell times (d_t) and a detection
        area (fov).

        Also considers the environmental structure as an occupancy map to check which cells are visible and which are
        not

        Args:
            path: Path on the grid (in world coordinates) given as [2xn] numpy array
            d_t: Dwell time for each position of the path. Must have same length as path vector.
            fov: Field of view of the robot to determine which cells are visible for each position
            occ_map: Occupancy map to represent the environment
            mask_fov: mask resulting from the fov of the robot. All masked cells will not be considered
            min_obs_time: (Optional) Minimal observation time (in s). Cells below this threshold will be masked and
            not considered
            lazy: If true, check if path and dwell times are the same as for the first run. Observation times will
                    then only be calculated once to save computation time.
            create_gt: If true, every visible cell will be considered as being fully visible for the complete bin period
            new_path: If true, observation duration is computed (again)

        """
        if self.counts_masked is None:
            raise NameError("Counts must be defined before gamma params can be set. Call 'masked_counts(..)' first")
        if d_t.shape[0] != path.shape[0]:
            raise ValueError("Dwell time and path array must have same length")

        if create_gt:
            min_obs_time = 0.0

        if not lazy or self.obv_duration is None or new_path:
            print("Computing observation duration")
            self.path = path
            self.d_t = d_t

            if create_gt:
                self.obv_duration = np.ma.masked_array(mask_fov * 1.0, mask=~mask_fov)
            else:
                obv_duration = self._compute_obv_duration(path, d_t, fov, occ_map, ~self.counts_masked.mask)
                self.obv_duration = np.ma.masked_where(obv_duration <= (min_obs_time / self.scale_seconds),
                                                       obv_duration)

        # Rate 'counts per observation time' in each cell
        counts = np.ma.masked_array(self.counts_masked.data.astype(float),
                                    mask=copy(np.ma.mask_or(self.counts_masked.mask, self.obv_duration.mask)))
        self.rate = counts / self.obv_duration

        if self.min_rate is not None:
            # Set minimum rate
            self.rate[~self.rate.mask] = np.where(self.rate[~self.rate.mask] < self.min_rate,
                                                  self.min_rate, self.rate[~self.rate.mask])
            print("Replaced zero rates with {}".format(self.min_rate))

        # If you want to play around with the shape parameter of the gamma distribution:
        # ix = np.where(self.cell_counts <= 1)[0]
        # self.obv_duration_masked[ix] = 1.1 / self.rate_masked[ix]  # shape parameter p = 1.1
        # print("Modified observation durations of cells with a shape parameter <= 1, rate remains constant.")

        # Variance 'counts per (observation duration)²' in each cell
        # self.variance = copy(self.rate)
        # self.variance = self.rate / self.obv_duration

        # counts_mean = counts.mean()
        # counts_mean_weighted = np.sum(counts * self.obv_duration) / np.sum(self.obv_duration)
        # n_thresh = copy(counts)
        # n_thresh[n_thresh < counts_mean_weighted] = counts_mean_weighted
        # # Set minimum value of ratio -1 to zero
        assert np.all(np.greater_equal(self.obv_duration, 0.0)), "Obv duration has negative values"
        # Small rounding errors are ok
        assert np.all(np.less_equal(self.obv_duration, 1.0 + 1e-12)), "Obv duration must be <= 1.0"
        # t_ratio = np.clip((1 / self.obv_duration - 1), a_min=0, a_max=None)
        self.stddev = (1 / self.obv_duration - 1) * counts
        self.stddev += 1e-6

    def mask_small_rate(self, rate_min, ratio=1):
        """
        Given a minimal rate, masks the cells in the rate array with a rate lower equal this rate.

        Modifies the rate member variable.

        Args:
            rate_min: Minimal rate
            ratio: Ratio of cells to mask between [0,1]. If 0 mask all, if 1 mask none.

        """
        if self.rate is None:
            raise NameError("Rate not set. Call 'set_by_path(..)' first")

        # Clamp ratio to [0,1]
        ratio = max(min(ratio, 1.0), 0.0)
        # Condition where rate is low
        it = np.where(self.rate[~self.rate.mask] <= rate_min)[0]
        num_cells_low_rate = self.rate[~self.rate.mask].data[it].shape[0]
        drop_count = int((1 - ratio) * num_cells_low_rate)
        # Shuffle the vector to randomly mask ratio of cells with low rate
        np.random.shuffle(it)
        drop_ix = it[:drop_count]
        # Copy the mask. Necessary because mask in masked array cannot be set directly
        buf_mask = self.rate.mask.astype(bool)
        low_rate_mask = self.rate[~self.rate.mask].mask
        low_rate_mask[drop_ix] = True
        buf_mask[~self.rate.mask] = low_rate_mask
        self.rate = np.ma.masked_array(self.rate.data, mask=buf_mask)

    def get_stacked(self, norm_obs_dur=None):
        """
        Get the values of the grid as numpy array with [n x 4] dimension.

        n corresponds to the number of cells that are not masked due to small rate or short observation

        Returns:
            numpy nd array
        """
        if self.rate is None:
            raise NameError("Rate not set. Call 'set_by_path(..)' first")
        # Mask all cells that were not visible and that have a low rate and low observation dur
        mask = np.logical_or(self.counts_masked.mask, self.rate.mask)

        # Create numpy arrays from the masked arrays
        rate = np.array(self.rate[~mask].ravel(order='F').reshape(-1, 1))
        stddev = np.array(self.stddev[~mask].ravel(order='F').reshape(-1, 1))
        obv_dur = np.array(self.obv_duration[~mask].ravel(order='F').reshape(-1, 1))
        if norm_obs_dur is not None:
            obv_dur = obv_dur * self.scale_seconds / norm_obs_dur
        cell_counts = np.array(self.counts_masked[~mask].ravel(order='F').reshape(-1, 1))

        # Rate, stddev, counts, scale parameters for gamma distribution
        return np.hstack([rate, stddev, cell_counts, obv_dur])

    def _compute_obv_duration(self, path, d_t, fov, occ_map, mask):
        """Computes the observation duration for the cells"""
        # Only use multiprocessing if path is not very short
        if path.shape[0] > self.cpu_count and False:
            # Use multiprocessing:
            # Split the robot positions 'path' and determine the cumulative observation time of all cells afterwards
            indexes = np.array_split(np.arange(0, path.shape[0]), self.cpu_count)
            pool = mp.Pool(self.cpu_count)
            result = pool.starmap(runner_obv_duration,
                                  [(self, fov, path[index], occ_map, d_t[index], mask) for index in indexes])

            pool.close()

            dur_cum = sum(result)
        else:
            dur_cum = runner_obv_duration(self, fov, path, occ_map, d_t, mask)

        # Scale the observation duration of each cell
        dur_cum = dur_cum / self.scale_seconds

        return dur_cum

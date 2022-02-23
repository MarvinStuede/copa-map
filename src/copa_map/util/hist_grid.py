"""
Module for Histogram grid

Contains a class derived from grid to represent a histogram with counts
"""

import chaospy
import numpy as np
from sklearn.cluster import KMeans

from copa_map.util import util
from copa_map.util.grid import Grid


class HistGrid(Grid):
    """
    HistGrid class

    Represents a grid with count values for every cell
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor

        Creates the histogram grid

        Args:
            *args: args of base class
            **kwargs: kwargs of base class
        """
        super(HistGrid, self).__init__(*args, **kwargs)
        self.counts = None

    def get_counts(self):
        """
        Receive the counts

        Returns:
            Histogram counts
        """
        if self.counts is None:
            raise Exception('No data, call method set_counts first')
        else:
            return self.counts

    def set_counts(self, positions, in_grid_frame=False):
        """
        Receives positions and counts how many fall into the corresponding grid cells

        Args:
            positions: Positions to count
            in_grid_frame: Positions given in frame of the grid, transforms otherwise

        Returns:
            Histogram counts

        """
        if in_grid_frame:
            pos = positions
        else:
            pos = self.tf_to(positions)
        self.counts, _ = np.histogramdd(pos, bins=(self.elements_x, self.elements_y),
                                        range=((0, self.width), (0, self.height)))
        self.counts = np.int_(self.counts)
        return self.counts

    def sample(self, sample_num, duplicates=None, rnd_ratio=None, use_clusters=False, seed=None):
        """
        Sample the histogram

        Sample a number of cell centers, weighted by the count values
        Args:
            sample_num: Number of samples
            duplicates: (Optional) choose:
                delete: Delete duplicate positions in the list
                replace: Replace duplicates in the list with other values
            rnd_ratio: (Optional) ratio of random cells to cells of highest count values (1-ratio).
                        Instead of weighted cells
            use_clusters: Uses k-means clusters if True
            seed: (Optional) seed for evaluation

        Returns:
            Array of sampled positions (num x 2)
        """
        # Create the cumulative density function and norm it
        counts = self.counts.ravel()
        cdf = np.cumsum(counts)
        cdf = cdf / cdf[-1]
        rand = None

        if counts.max() == 0:
            util.logger().warn("Sampling a grid without counts. Returning empty array.")
            return np.array([])

        if seed is not None:
            print("Uses {} seed for sampling positions".format(seed))
            np.random.seed(seed)

        if use_clusters:
            data = np.where(self.counts > 0)
            pos = self.centers_grid[data]
            sample_weight = self.counts[data]
            # todo: set random_state
            km = KMeans(init='k-means++', n_clusters=min(sample_num, pos.shape[0]), n_init=10,
                        random_state=seed).fit(pos, sample_weight=sample_weight)
            return km.cluster_centers_

        if rnd_ratio is not None:
            num_random = int(rnd_ratio * sample_num)
            num_highest = sample_num - num_random
            if num_highest > 0:
                bin_val = np.argpartition(counts, -num_highest)[-num_highest:]
                x_idx, y_idx = np.unravel_index(bin_val, (self.elements_x, self.elements_y))
                rand = self.centers_grid[x_idx, y_idx]
            num = num_random
        else:
            num = sample_num

        while num > 0:
            # Create random values and determine in which bin they fall
            if rnd_ratio is None:
                rand_val = np.random.rand(num)
                bin_val = np.searchsorted(cdf, rand_val)
            else:
                indices = np.where(counts > 0)[0]
                rng = np.random.default_rng(seed=seed)
                rng.shuffle(indices, axis=0)
                bin_val = indices[:num]

            x_idx, y_idx = np.unravel_index(bin_val, (self.elements_x, self.elements_y))

            _rand = self.centers_grid[x_idx, y_idx]
            rand = _rand if rand is None else np.vstack((rand, _rand))

            if duplicates == "replace":
                rand = np.unique(rand, axis=0)
                num = sample_num - rand.shape[0]
            elif duplicates == "delete":
                return np.unique(rand, axis=0)
            else:
                return rand

        return rand

    def get_top_n(self, num):
        """
        Get the cells with maximum count

        Args:
            num: Number of cells to return

        Returns:
            np array of indices of the top cells
        """
        indices = (-self.counts).argpartition(num, axis=None)[:num]
        x, y = np.unravel_index(indices, self.counts.shape)
        return np.array(list(zip(x, y)))

    def sample_zero(self, num=None, mask=None, use_halton=False, ratio=None, seed=None):
        """
        Sample a number of cell centers where the count is zero

        Args:
            num: (Optional) Number of cells to sample
            mask: (Optional) mask to exclude cells, same size as grid
            use_halton: (Optional) Instead of uniform random, use halton sampling
            ratio: (Optional) Instead of num parameter, use ratio of total number of zeros
            seed: (Optional) seed for evaluation

        Returns:
            Array of sampled positions (num x 2)
        """
        # Get indices of cells with zero counts
        indices = self._get_ind_of_zeros(mask)
        if ratio is None and num is None:
            return self.centers_grid[tuple(indices)]
        if ratio is not None:
            if ratio < 0 or ratio > 1:
                raise ValueError('Invalid ratio for zero sampling')

            num_sample = int(ratio * indices.shape[1])
        else:
            num_sample = num

        if num_sample == 0:
            return np.array([])

        ind_sampled = self._sample(num_sample, indices, use_halton, seed)

        centers = self.centers_grid[tuple(ind_sampled)]
        return centers

    def masked_counts(self, ratio, mask=None, use_halton=False, seed=None):
        """
        Create a member variable of counts where cells without any counts are masked.

        A ratio can be given to keep cells without counts. Also, an existing mask can be extended.
        Args:
            use_halton: Instead of uniform random, use halton sampling
            ratio: Ratio of cells to mask (1: mask all, 0: mask none)
            mask: (Optional) Mask to extend
            seed: Seed for evaluation

        Returns:
            numpy ma array of masked counts

        """
        new_mask = self.mask_zeros(ratio=ratio, mask=mask, seed=seed, use_halton=use_halton)
        self.counts_masked = np.ma.masked_array(self.get_counts(), mask=new_mask)
        return self.counts_masked

    def plot_counts(self, plt, masked=False, zorder=0, **kwargs):  # pragma: no cover
        """
        Plot the counts to an matplotlib object

        Args:
            plt: Axis object
            masked: (Optional) If true, plot a grey mask over masked counts
            zorder: Zorder of the plot
            **kwargs: Optional named arguments

        """
        if not masked and self.counts is None:
            raise NameError("Counts must be defined before gamma parameters can be set. Call 'set_counts(..)' first")
        if masked and self.counts_masked is None:
            raise NameError("Counts masked must be defined before gamma parameters can be set."
                            "Call 'masked_counts(..)' first")

        grid_edges = self.get_edges(as_3d=True)
        # Plot the histogram
        mesh_c = plt.pcolormesh(grid_edges[:, :, 0], grid_edges[:, :, 1], self.counts, shading='auto', cmap='jet',
                                alpha=0.5, zorder=zorder, **kwargs)
        if masked:
            # Overlay for area outside of FOV
            mesh_m = plt.pcolormesh(grid_edges[:, :, 0], grid_edges[:, :, 1], ~self.counts_masked.mask,
                                    shading='auto', cmap='Greys', zorder=zorder + 1, alpha=0.3)
        else:
            mesh_m = None

        return mesh_c, mesh_m

    def mask_zeros(self, ratio, mask=None, use_halton=False, seed=None):
        """
        Create a mask where the cells have no counts

        Can use a ratio parameter to randomly mask the cells partially
        Args:
            use_halton: Instead of uniform random, use halton sampling
            ratio: Ratio of cells to mask (0: mask all, 1: mask none)
            mask: (Optional) Mask to extend
            seed: Seed for evaluation

        Returns:
            Mask array
        """
        if mask is None:
            cmask = ~(self.counts == 0)
        else:
            cmask = np.logical_or(~(self.counts == 0), mask)

        if ratio == 1.0:
            return np.ones(cmask.shape) == 0

        # 'cmask' now masks all the cells that should NOT be considered. Therefore ~cmask gives the zero cells
        indices = np.array(np.where(~cmask))

        if ratio == 0 or indices.size == 0:
            # If want to mask all zeros or there are no cells with zeros
            return np.logical_or(self.counts == 0, mask) if mask is not None else self.counts == 0

        # Determine the number of indices to keep
        num_mask = np.clip(int(indices.shape[1] * ratio), 0, indices.shape[1] - 1)

        indices = self._sample(num_mask, indices, use_halton, seed)
        cmask[tuple(indices)] = True
        if mask is not None:
            cmask[mask] = False
        return ~cmask

    def _sample(self, num, indices, use_halton, seed):
        if use_halton:
            # Create a uniform distribution with the number of indices as the size
            distribution = chaospy.J(chaospy.Uniform(0, indices.shape[1] - 1))
            samples = np.int_(distribution.sample(num, rule="halton"))  # duplicates because of int...
            indices = indices[:, samples]
        else:
            # Randomly shuffle the array
            if seed is not None:
                print("Uses {} seed for masking zeros".format(seed))
            rng = np.random.default_rng(seed=seed)
            # rng.shuffle(indices, axis=1)
            # Shuffle indices of indices
            # Numpy < 1.18 does not support axis parameter
            indices_ind = np.arange(0, indices.shape[1])
            rng.shuffle(indices_ind)
            indices = indices[:, indices_ind]
            indices = indices[:, :num]
        return indices

    def _get_ind_of_zeros(self, mask):
        if mask is None:
            cmask = ~(self.counts == 0)
        else:
            cmask = np.logical_or(~(self.counts == 0), mask)
        indices = np.array(np.where(~cmask))

        return indices

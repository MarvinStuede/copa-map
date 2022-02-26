"""Module to represent time series and spectral data for grid like structures"""

from copa_map.util import util
from copa_map.util.hist_grid import HistGrid
from copa_map.util.spectral import FreMEn, NUFFT
import pandas as pd
import numpy as np
from math import pi
from copa_map.util import util as ut
from joblib import Parallel, delayed
from sklearn.cluster import KMeans


class GridTimeSeries(HistGrid):
    """
    The GridTimeSeries class

    Represents a grid, where the cells can contain time series data e.g. for frequency analysis
    """

    def __init__(self, Tp_max=3600 * 20, Tp_min=3600 * 0.5, Tp_res=10000, freq_mode="fremen-aam", max_freq_num=10,
                 *args, **kwargs):
        """
        Constructor

        Args:
            Tp_max: Maximum expected  period (in s). Based on this value, the minimal expected frequency will be
                    calculated. This frequency is the lower bound for all considered frequencies of the NUFFT
            Tp_min: Minimum expected  period (in s). Based on this value, the maximum expected frequency will be
                    calculated. This frequency is the upper bound for all considered frequencies of the NUFFT
            Tp_res: Resolution of expected frequencies. Will divide the interval [Tp_min, Tp_max] to obtain the given
                    number of frequencies inside the interval
            freq_mode:  Decides which method will be used to calculate the spectral data.
                        Possible inputs: "nufft" or "fremen"
            num_aam_freq: Number of frequencies that will be stored for aam method. A subset of these frequencies will
                          be used for prediction.

            *args:
            **kwargs:
        """
        self.Tp_max = Tp_max
        self.Tp_min = Tp_min
        self.Tp_res = Tp_res
        self.freq_mode = freq_mode
        self.freq_norm = 1.0
        self.max_freq_num = max_freq_num
        self.A_arr = []
        self.O_arr = []
        self.p_arr = []

        assert (self.freq_mode == "nufft" or self.freq_mode == "fremen-bam" or self.freq_mode == "fremen-aam"), \
            "GridTimeSeries.freq_mode should be 'nufft' or 'fremen-aam'  or 'fremen-bam'"
        super(GridTimeSeries, self).__init__(*args, **kwargs)
        # Create an empty array of time series cells

        self.cpu_count = util.get_cpu_count()

        self.cells_ts = [[None for i in range(self.elements_y)] for j in range(self.elements_x)]

    def set_data(self, positions: np.ndarray, counts, dwell_time, t: np.ndarray, in_grid_frame=False,
                 split_train_valid=True):
        """
        Set the data of the time series.

        Given a number of (x,y) positions and timestamps the cells of the grid will be filled with this data.
        Each cell then contains the respective measurements and time series containing the number of positions for each
        contained timestamp
        Args:
            positions: nd array of positions in dimension (n, 2)
            t: nd array of timestamps with dimenstion (n,)
            in_grid_frame: If true, expect positions to be given in grid frame, else in world frame


        """
        # Should be same length
        assert positions.shape[0] == t.shape[0]

        if in_grid_frame:
            pos = positions
        else:
            pos = self.tf_to(positions)

        # Get cell indices for all positions
        ind = self.index_from_pos(pos)
        # Get all unique cells to iterate over cells with measurements
        ind_unq = np.unique(ind, axis=0)
        # Filter out invalid indices (detections outside the grid)
        valid_indices = np.apply_along_axis(self.is_index_valid, 1, ind_unq)
        ind_unq = ind_unq[valid_indices]
        self.indices_spectral = ind_unq
        if ind_unq == []:
            raise Exception("Grid does not contain measurements. Did you set the dimensions correctly?")

        if split_train_valid:
            num_folds = 5
        else:
            # tstart_valid = None
            num_folds = 1

        # Iterate the cells with measurements
        for cell in ind_unq:
            # Get row indices of all rows with this specific index
            row_i = np.where(np.all(ind == cell, axis=1))
            # Write the data of this cell to a np array, then create a pandas dataframe from it
            data = np.column_stack([t[row_i], pos[row_i], counts[row_i], dwell_time[row_i]])
            df = pd.DataFrame(data=data, columns=["t", "pos_x", "pos_y", "counts", "d_t"])
            # Convert to datetime, if parameter is given
            if self.freq_mode == "nufft":
                self.cells_ts[cell[0]][cell[1]] = NUFFT(df=df, num_folds=num_folds)
            elif self.freq_mode == "fremen-aam":
                self.cells_ts[cell[0]][cell[1]] = FreMEn(df=df, num_folds=num_folds, mode="aam",
                                                         max_freq_num=self.max_freq_num)
            elif self.freq_mode == "fremen-bam":
                self.cells_ts[cell[0]][cell[1]] = FreMEn(df=df, num_folds=num_folds, mode="bam",
                                                         max_freq_num=self.max_freq_num)

        # self.set_counts(pos, in_grid_frame=True)
        cells_sorted = ind[np.flipud(np.argsort(counts))]
        ind_and_count = np.hstack([cells_sorted, counts.reshape(-1, 1)])
        _, idx = np.unique(cells_sorted, axis=0, return_index=True)
        # restore the sorted order after unique operation
        self.top_cells = cells_sorted[np.sort(idx)]
        self.sample_weights = np.array([np.sum(ind_and_count[np.all(ind_and_count[:, :2] == t_cell, axis=1)][:, 2])
                                        for t_cell in self.top_cells]).reshape(-1, 1)

    def _get_freq_candidates(self):
        """
        Given on the period values, creates an array of candidate frequencies

        Frequencies are given as circular frequencies
        Returns: The range

        """
        step = (self.Tp_max - self.Tp_min) / self.Tp_res
        np.arange(self.Tp_min, self.Tp_max, step)
        return np.flipud(2 * pi / np.arange(self.Tp_min, self.Tp_max, step))

    def predict(self, Xt: np.ndarray):
        """Predict at stamped positions"""
        assert self.top_count_ind is not None, "Call 'calc_freqs(...)' first"
        results = np.zeros(Xt.shape[0])
        indices = self.index_from_pos(self.tf_to(Xt[:, :2]))

        for index in np.unique(indices, axis=0):
            ind_learned = any(np.equal(self.top_count_ind, index).all(1))
            if ind_learned:
                index_rows = np.all(indices == index, axis=1)
                t = Xt[index_rows][:, 2]
                p = self.cells_ts[index[0]][index[1]].predict(t)
                results[index_rows] = p
            else:
                ut.logger().warn("Cell [" + str(index[0]) + "," + str(index[1]) + "] not learned")
        return results

    def calc_freqs(self, n_cells=-1, sample=True, use_dwell_time=False):
        """
        Calculates the prominent frequencies for a number of cells

        For the 'n_cells' cells with the most positions inside them, a spectral analysis will be done.
        This includes the calculation of the Non uniform fourier transform to obtain the spectral components and
        frequencies.
        Then, the 'n_prom_freq' most prominent frequencies will be calculated (based on the absolute value of the
        corresponding spectral component). This results in an array with dimension (n_cells, n_prom_freq)

        Args:
            n_cells: Number of cells for which to do the spectral analysis
            mode: "bam" or "aam", Best Amplitude Model or Additional Amplitude Model
            use_dwell_time: If true, dwell times of the robot will be incorporated in frequency analysis

        """
        # Get the defined number of cells with maximum counts
        if n_cells == -1:
            self.top_count_ind = self.top_cells
        elif n_cells > 0:
            if sample:
                ind_s = np.arange(0, self.top_cells.shape[0])
                draw = np.random.choice(ind_s, n_cells, p=self.sample_weights.ravel() / np.sum(self.sample_weights),
                                        replace=False)
                self.top_count_ind = self.top_cells[draw]
            else:
                self.top_count_ind = self.top_cells[:n_cells]
        else:
            raise Exception("Invalid value for n_cells")
        # Get the list of frequency cadidates
        self.freq_candidates = self._get_freq_candidates()

        # Iterate through the cells

        def freq_analysis_process(index, cell, freq_cand):
            # Calculate the NUFFT or FreMEn for the cell
            cell.freq_analysis(freq_cand, use_dwell_time)
            print("Freq. Analysis for cell " + str(index) + ", num freq: " + str(cell.num_pred_freqs))
            return cell

        # Do the frequency analysis on multiple cores
        # If vector is smaller than number of cpus, only use this number
        n_cpu = min([self.cpu_count, self.top_count_ind.shape[0]])
        cells = Parallel(n_jobs=n_cpu)(delayed(freq_analysis_process)(
            index, self.cells_ts[index[0]][index[1]], self.freq_candidates) for index in self.top_count_ind)
        for i in range(0, self.top_count_ind.shape[0]):
            self.cells_ts[self.top_count_ind[i, 0]][self.top_count_ind[i, 1]] = cells[i]

        return self._gather_freqs_from_cells()

    def _gather_freqs_from_cells(self):

        self.A_arr = []
        self.O_arr = []
        self.p_arr = []
        for index in self.top_count_ind:
            cell = self.cells_ts[index[0]][index[1]]
            A_cell = np.abs(cell.prom_cplx_comp[:cell.num_pred_freqs])
            O_cell = 2 * pi / cell.prom_freq[:cell.num_pred_freqs]
            p_cell = cell.num_pred_freqs
            self.A_arr.append(A_cell)
            self.O_arr.append(O_cell)
            self.p_arr.append(p_cell)

    def get_clustered_periods(self):
        """Return the cluster centers of calculated frequencies"""
        psi = int(np.round(np.average(np.array(self.p_arr))))
        O_all = np.concatenate(self.O_arr)
        A_all = np.concatenate(self.A_arr)
        assert O_all.shape == A_all.shape
        km = KMeans(n_clusters=psi, max_iter=1200).fit(O_all.reshape(-1, 1),
                                                       sample_weight=A_all)
        # Sum up the weights for each cluster
        weighted_centers = np.zeros_like(km.cluster_centers_)
        for label in np.unique(km.labels_):
            weighted_centers[label] = np.sum(A_all[km.labels_ == label])
        weighted_centers /= np.max(weighted_centers)
        # Return the cluster centers, sorted by influence in descending order (based on amplitude)
        clusters = km.cluster_centers_.ravel()[np.argsort(-weighted_centers.ravel(), axis=0)]
        weighted_centers = weighted_centers.ravel()[np.argsort(-weighted_centers.ravel(), axis=0)]
        self.num_pred_freq = psi
        return clusters, weighted_centers

    def plot_cell_spectra(self, plto):
        """
        Plot the spectra of the cells

        Args:
            plto: matplotlib object to plot to

        """
        assert self.top_count_ind is not None
        for index in self.top_count_ind:
            self.cells_ts[index[0]][index[1]].plot_spectrum(plto, label=self.counts[index[0]][index[1]])
        # Plot a legend with the count as labels
        plto.legend(loc="upper left")

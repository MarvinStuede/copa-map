from copa_map.util import util
import pandas as pd
import numpy as np
from finufft import nufft1d3
from pandas import DataFrame
from scipy.signal import argrelextrema
from copy import copy
from sklearn.model_selection import KFold
from abc import abstractmethod


class SpectralData():
    """
    The SpectralData class

    Class to represent a number of detections as a dataframe, resulting timeseries and spectral components of this data
    """

    def __init__(self, df: DataFrame, max_freq_num=10, num_folds=1):
        """
        Constructor

        Args:
            df: Dataframe that contains (t, pos_x, pos_y) as columns
            aam_freq_num: Number of frequencies that will be stored for aam method. A subset of these frequencies will
                          be used for prediction.
            num_folds: If one, to hold out cross validation, else do k-fold with this number of folds
        """
        self.df = df.groupby("t").sum()
        self.df.reset_index(level=0, inplace=True)
        self.num_folds = num_folds
        # Timeseries for count data (number of detections per time step)
        self.ds = pd.Series(data=self.df.counts)
        self.dt = pd.Series(data=self.df.d_t)
        self.ds.index = self.df.t
        # self.ds = self.df['t'].value_counts().sort_index()
        # Length of the timeseries
        self.t_len = self.ds.__len__()
        # Complex spectral components
        self.cplx_comp = None

        self.max_freq_num = max_freq_num
        self.num_pred_freqs = 0
        # Frequency candidates
        self.freqs_cand = None

    def freq_analysis(self, freq_candidates, use_dwell_time):
        """
        Do the frequency analysis with optional cross validation

        Args:
            freq_candidates: Array with frequency candidates
            method: nufft or fremen-aam or fremen-bam
            use_dwell_time: If true, dwell time of robot are considered in intensity calculation

        """

        if self.num_folds <= 1:
            raise NotImplementedError
            # self.const_comp, self.prom_cplx_comp, self.prom_freq = self._freq_analysis(freq_candidates, use_dwell_time)
            # self.num_pred_freqs, _ = self._determine_predict_freq_num(self.const_comp,
            #                                                           self.prom_cplx_comp,
            #                                                           self.prom_freq)
        else:
            kf = KFold(n_splits=self.num_folds, shuffle=False)
            self.ds_buf = copy(self.ds)
            self.dt_buf = copy(self.dt)
            rmse = float('inf')

            for train_index, test_index in kf.split(self.ds_buf.to_numpy()):

                tr_dat = self.ds_buf.values[train_index]
                # Always use the same timestamps for training (to avoid gaps in the data)
                tr_t = self.ds_buf.index[train_index]
                tr_dt_dat = self.dt_buf.values[train_index]
                tr_dt_i = self.dt_buf.index[train_index]
                test_dat = self.ds_buf.values[test_index]
                test_t = self.ds_buf.index[test_index]
                self.ds = pd.Series(data=tr_dat, index=tr_t)
                self.dt = pd.Series(data=tr_dt_dat, index=tr_dt_i)
                self.ds_valid = pd.Series(data=test_dat, index=test_t)
                const_comp, prom_cplx_comp, prom_freq = self._freq_analysis(freq_candidates, use_dwell_time)

                num_pred_freqs, cur_rmse = self._determine_predict_freq_num(const_comp, prom_cplx_comp, prom_freq)

                if cur_rmse < rmse:
                    # self.prom_cplx_comp = copy(prom_cplx_comp)
                    # self.prom_freq = copy(prom_freq)
                    # self.const_comp = copy(const_comp)
                    self.num_pred_freqs = copy(num_pred_freqs)
            self.ds = copy(self.ds_buf)
            self.dt = copy(self.dt_buf)
            self.const_comp, self.prom_cplx_comp, self.prom_freq = self._freq_analysis(freq_candidates, use_dwell_time)

    @abstractmethod
    def _freq_analysis(self, freq_candidates, use_dwell_time):
        pass

    def _poisson(self, t, s, dwell_time=None):
        """
        Method to calculate poisson parameters

        Args:
            s (np.array(float)): array which contains observations s at time t.
            Default: 3600/6 seconds or 10 minutes.

        Returns:
            t_new (np.array(float)): array which contains times t where the poisson rate parameter gets evaluated
            self.poisson_lambda (np.array(float)): array which contains poisson rate parameter at time t_
        """
        if dwell_time is None:
            return s
        self.poisson_alpha = np.cumsum(s) + 1
        self.poisson_beta = np.cumsum(dwell_time) + 1
        self.poisson_lambda = self.poisson_alpha / self.poisson_beta
        return self.poisson_lambda

    def _determine_predict_freq_num(self, const_comp, prom_cplx_comp, prom_freq, min_freq=1):
        """
        Check how many frequencies should be used for prediction

        calculates the RMSE for prediction with increasing number of frequencies and stores the number with smallest
        RMSE
        """
        # Create range for every possible number of predictive freqs
        pred_num_candidates = np.arange(min_freq, self.max_freq_num + 1).reshape(-1, 1)

        def calc_rmse(num_freqs):
            # Predict for this number of frequencies
            pred = self.predict(self.ds_valid.index.to_numpy(), const_comp, prom_cplx_comp, prom_freq,
                                num_prom_freq=num_freqs[0])
            # Calculate the rmse on the validation set
            rmse = np.sqrt(((pred - self.ds_valid.values) ** 2).mean())
            # print("RMSE for " + str(num_freqs[0]) + " frequencies: "+str(rmse))
            return rmse

        # Use the number of frequencies with minimal RMSE
        rmse_arr = np.apply_along_axis(calc_rmse, arr=pred_num_candidates, axis=1)
        num_pred_freqs = pred_num_candidates[np.argmin(rmse_arr)][0]
        return num_pred_freqs, np.min(rmse_arr)

    def create_cosine(self, t, gamma_0, omega, gamma):
        """
        Create a cosine signal

        Args:
            t: timestamps
            gamma_0: Constant offset
            omega: Circular frequency
            gamma: Complex components

        Returns:
            Cosine values at t
        """
        cos_term = omega.reshape(-1, 1) * t + np.angle(gamma).reshape(-1, 1)
        if cos_term.size == 0:
            cos_term = np.zeros(t.shape)
        p = gamma_0 + (np.abs(gamma).reshape(-1, 1) * np.cos(cos_term)).sum(axis=0)
        return p

    def predict(self, t_rec, const_comp, prom_cplx_comp, prom_freq, num_prom_freq=None):
        """Predict at timestamps"""

        if num_prom_freq is None:
            num_prom_freq = self.num_pred_freqs
        # print("Predicting with " + str(num_prom_freq) + " freqs")
        p = self.create_cosine(t_rec, const_comp, prom_freq[:num_prom_freq],
                               prom_cplx_comp[:num_prom_freq])
        p = np.clip(p, a_min=0, a_max=None)
        return p

    def calc_prominent_strengths(self, num_prom_freq, cplx_comp, freq_candidates):
        """
        Given the complex components, find out the most prominent ones

        Based on the magnitude of the spectral components, the most prominent frequencies will be calculated

        Args:
            num_prom_freq: Number of frequencies

        Returns:
            frequencies, complex components
        """
        # if self.cplx_comp is None:
        #     raise AssertionError("Call nufft method before checking for prominent frequencies")
        # Calculate the indices of peaks in frequency spectrum, sorted by max amplitude in descending order
        ind_maxima = self._get_ind_of_freq_maxima(cplx_comp, self.t_len)
        # Get the l most prominent frequencies
        prom_freq = freq_candidates[ind_maxima][:num_prom_freq]
        # Get the corresponding most prominent complex strengths
        prom_cplx_comp = cplx_comp[ind_maxima][:num_prom_freq]
        return prom_cplx_comp, prom_freq

    def _get_ind_of_freq_maxima(self, cplx_comp, len):
        """
        Get the indices of the array of frequency candidates, where the spectral component has maximum magnitude

        Assumes that the frequency spectrum is smooth and calculates the local maxima
        Args:
            cplx_comp: Complex components of the frequency spectrum
            len: length of the signal

        Returns:
            Array indices for the array of frequency candidates
        """
        f = np.abs(cplx_comp / len)
        ind_maxima = argrelextrema(f, np.greater)[0]
        ind_maxima = np.flip(ind_maxima[np.argsort(f[ind_maxima])])
        return ind_maxima

    def plot_spectrum(self, ax, scale=3600, label=None):
        """
        Plot the spectrum of this data

        Args:
            ax: matplotlib object to plot to
            scale: scale the t values by this value
            label: label for legend
        """
        f = np.abs(self.cplx_comp / self.t_len)
        ax.plot(self.freqs_cand * scale, f, label=label)


class NUFFT(SpectralData):
    def __init__(self, *args, **kwargs):
        """Constructor"""
        super(NUFFT, self).__init__(*args, **kwargs)

    def _freq_analysis(self, freq_candidates, use_dwell_time=False):
        """
        Do the Non-uniform Fast Fourier Transform

        Given a range of frequencies candidates, the timeseries will be transformed to frequency space
        Args:
            freq_candidates:

        Returns:
            An array of complex components, representing the time series in the frequency space

        """

        t = self.ds.index.to_numpy()
        # convert to seconds (float)
        if isinstance(t[0], np.timedelta64):
            t = t / np.timedelta64(1, 's')

        # Strengths of the timeseries
        s = self.ds.values.astype(np.double)

        s = self._poisson(t, s)

        # Calculate the spectral frequeny components by a type 3 NUFFT
        cplx_comp = nufft1d3(t.astype(np.float32), s.astype(np.float32), freq_candidates.astype(np.float32)) / \
                    self.t_len
        const_comp = s.mean()
        prom_cplx_comp, prom_freq = self.calc_prominent_strengths(num_prom_freq=self.max_freq_num,
                                                                  cplx_comp=cplx_comp,
                                                                  freq_candidates=freq_candidates)
        return const_comp, prom_cplx_comp, prom_freq


class FreMEn(SpectralData):
    def __init__(self, mode="aam", *args, **kwargs):
        """Constructor"""
        super(FreMEn, self).__init__(*args, **kwargs)
        self.mode = mode
        self.gamma_0 = None
        self.gamma = None
        self.omega = None

    def _calc_components(self, t, s, freq_candidates):
        """
        Method to update spectral data with observations s made at time t.

        Args:
            t (np.array(float)): array which contains times t where observations s were made.
            s (np.array(float)): array which contains observations s at time t.
            freq_candidates (np.array(float)): set of candidate frequencies
        """
        assert (t.shape == s.shape and len(t.shape) == len(
            s.shape)), "Observation arrays s and t must be the same dimension"

        # update complex components by utilizing fremen method
        gamma_0 = s.mean()
        angles = t * freq_candidates.reshape(-1, 1)
        gamma = (((s - gamma_0) * np.exp(-1j * angles)).sum(axis=1)) / (t.shape[0])

        return gamma_0, gamma

    def _freq_analysis(self, freq_candidates, use_dwell_time=False):
        """
        Do the Frequency Map Enhancement method

        Given a range of frequencies candidates, the timeseries will be transformed to frequency space
        Args:
            freq_candidates:    array of frequencies, where the spectrum will be evaluated. Equal to set
                                Omega in FreMEn notation

        Returns:
            An array of complex components, representing the time series in the frequency space
        """
        t = self.ds.index.to_numpy()
        # convert to seconds (float)
        if isinstance(t[0], np.timedelta64):
            t = t / np.timedelta64(1, 's')
        # Strengths of the timeseries
        cs = self.ds.values.astype(np.double)
        dt = self.dt.values.astype(np.double)

        s = self._poisson(t, cs, dwell_time=dt if use_dwell_time else None)

        if self.mode == "aam":
            const_comp, prom_cplx_comp, prom_freq = self.aam(t, s, freq_candidates=freq_candidates)
        elif self.mode == "bam":
            const_comp, gamma = self._calc_components(t=t, s=s, freq_candidates=freq_candidates)
            prom_cplx_comp, prom_freq = self.calc_prominent_strengths(num_prom_freq=self.max_freq_num,
                                                                      cplx_comp=gamma,
                                                                      freq_candidates=freq_candidates)
        else:
            raise NotImplementedError("Frequency mode must be 'bam' or 'aam'")
        return const_comp, prom_cplx_comp, prom_freq

    def aam(self, t, s, freq_candidates):
        """
        Additional amplitude model from Jovan 2016

        "A Poisson-spectral model for modelling temporal patterns in human data observed by a robot"
        Args:
            t: timestamps
            s: Activations

        """
        S = pd.DataFrame([[s.mean(), 0, 0]], columns=['o_abs', 'o_arg', 'omega'])

        def FT(num_freq, s_use):
            _, cplx_comp = self._calc_components(t=t, s=s_use, freq_candidates=freq_candidates)
            gamma, omega = self.calc_prominent_strengths(num_freq, cplx_comp, freq_candidates)
            return omega, gamma

        FT(self.max_freq_num, s)
        s_use = copy(s)
        i = 0
        while S.shape[0] < (self.max_freq_num + 1):
            omega, gamma = FT(1, s_use)
            if len(omega) == 0 or len(gamma) == 0:
                break
            omega = omega[0]
            gamma = gamma[0]
            # if value is in dataframe
            c = np.isclose(S.omega, omega, atol=1e-6)
            if ~np.all(~c):
                Sk = S.loc[c]
                S.loc[c, 'o_abs'] = Sk.o_abs + np.abs(gamma)
                S.loc[c, 'o_arg'] = (Sk.o_arg + np.angle(gamma)) / 2
            else:
                S = S.append({'o_abs': np.abs(gamma),
                              'o_arg': np.angle(gamma),
                              'omega': omega}, ignore_index=True)
            s_ = self.create_cosine(t, 0, omega, gamma)
            s_use -= s_
            i += 1
            if i > 100:
                util.logger().warning("Cancel AAM loop, not enough freqs found")
                break

        const_comp = S.o_abs[0]
        cplx_comp = np.array(S.iloc[1:].o_abs) * np.exp(1j * np.array(S.iloc[1:].o_arg))
        # self.prom_cplx_comp = cplx_comp.reshape(-1)[np.argsort(cplx_comp).reshape(-1)][:self.num_prom_freq]
        prom_cplx_comp = cplx_comp.reshape(-1)
        freq = np.array(S.iloc[1:].omega)
        # self.prom_freq = freq.reshape(-1)[np.argsort(cplx_comp).reshape(-1)][:self.num_prom_freq]
        prom_freq = freq.reshape(-1)
        assert prom_cplx_comp is not None
        return const_comp, prom_cplx_comp, prom_freq
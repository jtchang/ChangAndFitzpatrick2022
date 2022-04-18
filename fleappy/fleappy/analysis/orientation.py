import os
from itertools import chain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numba as nb
from scipy.special import i0
from scipy.optimize import curve_fit
from scipy.stats import percentileofscore
from skimage.color import rgb2hsv, hsv2rgb
from skimage.transform import pyramid_expand
from sklearn.metrics import r2_score
from fleappy.analysis.blockwise import BlockwiseAnalysis
from matplotlib.axes import Axes
import logging
from typing import Tuple


def _von_mises(x, kappa, mu):
    return np.exp(kappa * np.cos(x-mu)) / (2 * np.pi * i0(kappa))


class OrientationAnalysis(BlockwiseAnalysis):
    """Analysis class for orientation tuned stimuli.

    Analysis for orientation stimuli where stimulus codes are directions equally spaced. This is a subclass of the
    blockwise analysis.

    Attributes:
        metrics (pandas.dataframe): Collection of metrics

    """

    __slots__ = []

    def __init__(self, expt, field, **kwargs):
        super().__init__(expt, field, **kwargs)

    def __setattr_(self, name, value):
        if hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            raise TypeError('Cannot set name %r on object of type %s' %
                            (name, self.__class__.__name___))

    def __str__(self):
        str_ret = f'{self.__class__.__name__}: {os.linesep}'
        for key in chain.from_iterable(getattr(cls, '__slots__', []) for cls in OrientationAnalysis.__mro__):
            if key is 'metrics':
                str_ret = str_ret + f'metrics: {self.metrics.columns}'
            else:
                str_ret = str_ret + f'{key}:{getattr(self, key)}{os.linesep}'
        return str_ret

    @staticmethod
    def _von_mises(x, kappa, mu):
        _von_mises(x, kappa, mu)

    @staticmethod
    def von_mises_or_fit(x: np.ndarray, A: float, kappa: float, mu: float, offset: float):
        """[summary]

        Args:
            x (np.ndarray): Range of angles (rad)
            A (float): Multiplicative Scaling
            B (float): DC offset
            kappa (float): Von Mises Kappa Value
            mu (float): Von Mises Mu Value (Peak center)

        Returns:
            [np.ndarray]: Calculated Von Mises Curve for given parameters
        """
        return A * _von_mises(x, kappa, mu) + offset

    @staticmethod
    def von_mises_dr_fit(x, Aa, Ab, kappa_a, kappa_b, mu, offset):
        return Aa * _von_mises(x, kappa_a, mu) + \
            Ab * _von_mises(x, kappa_b, mu+np.pi) + \
            offset

    def _pref_ortho_responses(self, orientation=True):
        """[summary]

        Args:
            orientation (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        responses = self.single_trial_responses()
        responses = responses[:, :-1, :] if self.expt.do_blank() else responses

        pref_responses = np.empty((responses.shape[0], responses.shape[2]))

        pref_stim = np.argmax(np.mean(responses, axis=2), axis=1)

        if orientation:
            ortho_responses = np.empty((responses.shape[0], responses.shape[2]*2))
            ortho_stim_a = (pref_stim + responses.shape[1]//4) % responses.shape[1]
            ortho_stim_b = (pref_stim - responses.shape[1]//4) % responses.shape[1]
        else:
            ortho_responses = np.empty((responses.shape[0], responses.shape[2]))
            ortho_stim = (pref_stim + responses.shape[1]//2) % responses.shape[1]

        for cell_id, resp in enumerate(responses):

            pref_responses[cell_id, :] = resp[pref_stim[cell_id], :]

            if orientation:
                ortho_responses[cell_id, :] = np.concatenate((responses[cell_id, ortho_stim_a[cell_id].astype(int), :],
                                                              responses[cell_id, ortho_stim_b[cell_id].astype(int), :]), axis=0)
            else:
                ortho_responses[cell_id, :] = responses[cell_id, ortho_stim[cell_id].astype(int), :]

        return pref_responses, ortho_responses

    def run(self, **kwargs):
        super().run(**kwargs)
        vectors, _ = self.vector_sum_responses()
        self.metrics['orientation'] = np.mod((np.rad2deg(np.angle(vectors)) + 360), 360)/2
        self.compute_one_minus_cv()

        for orientation in [True, False]:
            self.fit_tuning_curves(orientation=orientation)
            self.compute_cohensd(orientation=orientation)

        self.compute_si(orientation=False, fit=True)

        self.pairwise_orientation_matrix(pref_type='dr_fit', orientation=True)

    def vector_sum_responses(self, orientation: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the vector sum for all roi.

        Args:
            orientation (bool, optional): Defaults to True. Collapse direction to orientation.

        Returns:
            [np.ndarray]: Complex vector sums for responses, Complex vector amplitudes
        """

        responses = self.single_trial_responses()

        if self.expt.do_blank():
            responses = responses[:, :-1, :]
        if orientation:
            responses = np.concatenate((responses[:, :int(responses.shape[1]/2), :],
                                        responses[:, int(responses.shape[1]/2):, :]), axis=2)
        responses[responses < 0] = 0
        responses = np.median(responses, axis=2)
        angles = np.arange(responses.shape[1]) * (2*np.pi/responses.shape[1])
        unit_vectors = np.tile(np.expand_dims(np.exp(1j*angles), axis=0), (responses.shape[0], 1))

        vector_sums = np.sum(responses*unit_vectors, axis=1)

        amplitudes = np.sum(responses, axis=1)

        return vector_sums, amplitudes

    def scatter_preferences(self, ax: Axes = None, **kwargs):
        """ Scatter plot of orientation preferences.


        Args:
            orientation (bool, optional): Defaults to True. Use orientation space.
            ax (matplotlib.axes.Axes, optional): Defaults to None. Axis to plot scatter.

        Kwargs:
            osi_threshold (float): Threshold for 'tuned' cells based on OSI
            show-untuned (bool): Show untuned cells as gray x's in the plot
            threshold (float): Threshold for R^2 value
            fov (tuple): Frame size  ((xmin, xmax), (ymin, ymax))
            scalebar (float): Size of scalebar to draw
            pref_type (str): Preference type to use for plotting (see get_orientation_preferences)
            sig_tuned (bool) : significantly tuned cells only
            Kwargs related to get_orientation_preferences

        Returns:
            [matplotlib.figure.Figure]: Figure with the scatter plot
        """
        orientation = kwargs.get('orientation', True)
        pref_type = kwargs.pop('pref_type', 'fit')

        show_untuned = kwargs.get('show_untuned', True)
        kwargs['threshold'] = kwargs.get('threshold', 0.6)
        kwargs['colorbar'] = kwargs.get('colorbar', False)
        kwargs['si_threshold'] = kwargs.get('si_threshold', 0)
        kwargs['si_threshold'] = kwargs.get('osi_threshold', kwargs['si_threshold'])
        kwargs['si_threshold'] = kwargs.get('dsi_threshold', kwargs['si_threshold'])
        kwargs['cmap'] = kwargs.get('cmap', 'hsv')
        x, y = self.expt.pixel_frame_size()
        kwargs['fov'] = kwargs.get('fov', ((0, x), (y, 0)))
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        if orientation:
            preferences = self.get_orientation_preferences(pref_type, **kwargs)
            vmin = 0
            vmax = 180

            well_fit = self.well_fit(rsq_threshold=kwargs['threshold'],
                                     si_threshold=kwargs['si_threshold'],
                                     orientation=True)
            preferences[~well_fit] = np.nan
            if kwargs.get('sig_tuned', False) and 'OSI_sig' in self.metrics.keys():
                preferences[~self.metrics['OSI_sig'].values] = np.nan

        else:
            preferences = self.get_orientation_preferences(pref_type, **kwargs)
            vmin = 0
            vmax = 360
            well_fit = self.well_fit(rsq_threshold=kwargs['threshold'],
                                     si_threshold=kwargs['si_threshold'],
                                     orientation=False)
            preferences[~well_fit] = np.nan
            if kwargs.get('sig_tuned', False) and 'DSI_sig' in self.metrics.keys():
                preferences[~self.metrics['DSI_sig'].values] = np.nan
        centroids = self.expt.roi_positions()

        # untuned cells

        untuned_cells = np.isnan(preferences)

        _ = ax.scatter(centroids[~untuned_cells, 0],
                       centroids[~untuned_cells, 1],
                       s=25,
                       c=preferences[~untuned_cells],
                       cmap=kwargs['cmap'], vmin=vmin, vmax=vmax, zorder=2)
        if show_untuned:

            _ = ax.scatter(centroids[untuned_cells, 0],
                           centroids[untuned_cells, 1],
                           marker='x',
                           s=15,
                           c='gray',
                           cmap=kwargs['cmap'], vmin=vmin, vmax=vmax, zorder=1)

        self.format_scatter_plot(ax, **kwargs)

    def fit_tuning_curves(self, orientation: bool = True, **kwargs) -> None:
        """Fits von Mises function to responses.

        Handles the fitting of von Mises functions to response data using the analysis window specified in the
        OrientationAnalysis. Fit parameters are stored in metrics as:
            Orientation (or_params): [Amplitude,Kappa, Mu, Offset]
            Direction (dr_param): [Amplitude_1, Amplitude_2,  Kappa_1, Kappa_2, Mu, offset]

        Also computes the R^2 of the fit.
        N.B. Direction peaks are default offset by pi radians

        kwargs:
            override (bool): Force Recompute. Defaults to False

        Args:
            orientation (bool, optional): Defaults to True. [description]

        Returns: None


        """
        kwargs['override'] = kwargs.get('override', False)
        prefix = 'or' if orientation else 'dr'
        if not kwargs['override'] and prefix+'_params' in self.metrics and prefix+'_rsq' in self.metrics:
            logging.info('%s: Skipping refitting %s tuning curve', self.expt.animal_id, prefix)
            return None

        responses = self.single_trial_responses()
        responses = responses[:, :-1, :] if self.expt.do_blank() else responses

        if orientation:

            responses = np.concatenate((responses[:, :int(responses.shape[1]/2), :],
                                        responses[:, int(responses.shape[1]/2):, :]), axis=2)

            fit_func = OrientationAnalysis.von_mises_or_fit
            fit_params = np.empty((responses.shape[0], 4))
            bounds = ((0, 0, 0, 0), (np.inf, 2*np.pi, 2*np.pi, np.inf))

        else:
            fit_func = OrientationAnalysis.von_mises_dr_fit
            fit_params = np.empty((responses.shape[0], 6))
            bounds = ((0, 0, 0, 0, 0, 0), (np.inf, np.inf, 2*np.pi, 2*np.pi, np.pi, np.inf))

        responses[responses < 0] = 0
        responses[responses == np.inf] = np.nan
        responses = np.nanmedian(responses, 2)
        angles = np.arange(responses.shape[1]) * (2*np.pi/responses.shape[1])
        r_sq = np.empty((responses.shape[0]))
        for cell_id, resp in enumerate(responses):
            try:
                fit_params[cell_id, :], _ = curve_fit(fit_func,
                                                      angles,
                                                      resp,
                                                      bounds=bounds)
                r_sq[cell_id] = r2_score(resp, fit_func(angles,
                                                        *fit_params[cell_id, :]))

            except ValueError as _:
                logging.warning('%s Could not fit %s for cell # %i',
                                self.expt.animal_id,
                                prefix,
                                cell_id)
                fit_params[cell_id, :] = np.nan
                r_sq[cell_id] = np.nan
            except RuntimeError as _:
                logging.warning('%s Could not fit %s for cell # %i',
                                self.expt.animal_id,
                                prefix,
                                cell_id)
                fit_params[cell_id, :] = np.nan
                r_sq[cell_id] = np.nan

        self.metrics[prefix+'_params'] = fit_params.tolist()
        self.metrics[prefix+'_rsq'] = r_sq.tolist()

    def metric_columns(self):
        return self.metrics.columns

    def compute_one_minus_cv(self):
        """Computes the one minus CV

        Returns:
            [numpy array]: 1-CV
        """
        vectors, amplitudes = self.vector_sum_responses()

        self.metrics['1-CV'] = np.abs(vectors)/amplitudes

    def compute_cohensd(self, orientation: bool = True):
        """Calculate Cohen's d for orientation/direction.

        Computes the Cohen's d for either orientation or direction. Cohen's d is defined as:

            $Cohen's\:d = {\frac{\mu_1 -\mu_2}{SD_{pooled}}}$

        where:

            $SD_{pooled} = \sqrt{\frac{(n_1 -1) \times SD_{1}^2 + (n_2 - 1) \times SD_1^2}{n_1 + n_2 -2}}$

        Args:
            orientation (bool, optional): [description]. Defaults to True.
        """
        pref_responses, ortho_responses = self._pref_ortho_responses(orientation=orientation)

        mean_difference = np.mean(pref_responses, axis=1)-np.mean(ortho_responses, axis=1)

        # pooled SD calculator
        numerator_a = (pref_responses.shape[1]-1) * np.std(pref_responses, axis=1)**2
        numerator_b = (ortho_responses.shape[1]-1) * np.std(ortho_responses, axis=1)**2
        denominator = pref_responses.shape[1]+ortho_responses.shape[1]-2
        sd_pooled = np.sqrt((numerator_a + numerator_b) / denominator)

        if orientation:
            self.metrics['CohensDOR'] = (mean_difference/sd_pooled).tolist()
        else:
            self.metrics['CohensDDR'] = (mean_difference/sd_pooled).tolist()

    def compute_si(self, orientation: bool = False, fit: bool = True, **kwargs):
        """Compute the Classic OSI and DSI

        Args:
            orientation (bool, optional): [description]. Defaults to False.
            fit (bool, optional): [description]. Defaults to True.
        """

        if fit and orientation:
            or_params = np.stack(self.metrics['or_params'].values)
            pref_resp = np.empty((or_params.shape[0],))
            orth_resp = np.empty((or_params.shape[0],))

            for cell_idx, params in enumerate(or_params):
                angle = params[-2]
                pref_resp[cell_idx] = OrientationAnalysis.von_mises_or_fit(angle,  *params)
                orth_resp[cell_idx] = OrientationAnalysis.von_mises_or_fit(angle+np.pi, *params)
            selectivity_index = np.abs(pref_resp - orth_resp) / (pref_resp + orth_resp)

            self.metrics['OSI'] = selectivity_index.tolist()
        elif fit and not orientation:
            dr_params = np.stack(self.metrics['dr_params'].values)

            dsi = np.empty((dr_params.shape[0]))
            osi = np.empty((dr_params.shape[0]))

            for cell_idx, params in enumerate(dr_params):
                osi[cell_idx], dsi[cell_idx] = OrientationAnalysis.calculate_dsi_osi_from_drfit(params)

            self.metrics['DSI'] = dsi.tolist()
            self.metrics['OSI'] = osi.tolist()
        else:
            pref_resp, ortho_resp = self._pref_ortho_responses(orientation=orientation)

            pref_resp = np.mean(pref_resp, axis=1)
            orth_resp = np.mean(ortho_resp, axis=1)
            selectivity_index = np.abs(pref_resp - orth_resp) / (pref_resp + orth_resp)

            if orientation:
                self.metrics['OSI'] = selectivity_index.tolist()
            else:
                self.metrics['DSI'] = selectivity_index.tolist()

    def significance_test_si(self, orientation: bool = False, fit: bool = True, iterations=10000, **kwargs):

        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['override'] = kwargs.get('override', True)

        if not orientation and fit:
            if 'OSI_sig' in self.metrics and 'DSI_sig' in self.metrics:
                return self.metrics['OSI_sig'].values, self.metrics['DSI_sig'].values
            else:
                observed_osi = self.metrics['OSI'].values
                observed_dsi = self.metrics['DSI'].values
                dr_rsqs = self.metrics['dr_rsq'].values

                significant_osi = np.zeros((observed_osi.size), dtype=bool)
                significant_dsi = np.zeros((observed_osi.size), dtype=bool)

                responses = self.single_trial_responses()
                responses = responses[:, :-1, :] if self.expt.do_blank() else responses  # cells x stims x trials
                angles = np.arange(responses.shape[1]) * (2*np.pi/responses.shape[1])
                for cell_idx, cell_responses in enumerate(responses):
                    logging.info('[%s] Running Significance Testing Shuffles for Cell #%i',
                                 self.expt.animal_id, cell_idx)

                    if dr_rsqs[cell_idx] > 0.6:
                        flattened_cell_responses = np.reshape(cell_responses, (cell_responses.size))

                        cell_osi_shuffles = np.full((iterations), np.nan)
                        cell_dsi_shuffles = np.full((iterations), np.nan)
                        for shuffle_num in range(iterations):
                            params = np.nan
                            loop_limit = 2

                            while np.any(np.isnan(params)) and loop_limit > 0:
                                shuffle_trials = np.random.randint(0, 
                                                                   flattened_cell_responses.size, 
                                                                   (flattened_cell_responses.size))
                                shuffle_responses = np.reshape(flattened_cell_responses[shuffle_trials],
                                                               (cell_responses.shape[0], cell_responses.shape[1]))
                                try:
                                    params, _ = curve_fit(OrientationAnalysis.von_mises_dr_fit,
                                                          angles,
                                                          np.nanmedian(shuffle_responses, 1),
                                                          bounds=((0, 0, 0, 0, 0, 0), (np.inf, np.inf, 2*np.pi, 2*np.pi, np.pi, np.inf)))

                                except:
                                    loop_limit = loop_limit - 1

                            if loop_limit == 0:
                                logging.info('[%s] Failed to fit a shuffle within attempt limit for cell #%i',
                                             self.expt.animal_id, cell_idx)
                                cell_osi_shuffles[shuffle_num], cell_dsi_shuffles[shuffle_num] = (0, 0)
                            else:
                                cell_osi_shuffles[shuffle_num], cell_dsi_shuffles[shuffle_num] = OrientationAnalysis.calculate_dsi_osi_from_drfit(
                                    params)

                        significant_osi[cell_idx] = percentileofscore(cell_osi_shuffles[np.isfinite(cell_osi_shuffles)],
                                                                      observed_osi[cell_idx]) > 95
                        significant_dsi[cell_idx] = percentileofscore(cell_dsi_shuffles[np.isfinite(cell_dsi_shuffles)],
                                                                      observed_dsi[cell_idx]) > 95
                    else:
                        significant_osi[cell_idx] = False
                        significant_dsi[cell_idx] = False
                self.metrics['OSI_sig'] = significant_osi.tolist()
                self.metrics['DSI_sig'] = significant_dsi.tolist()

        else:
            raise NotImplementedError('Significance testing based on orienation space iisn''t implemented!')

    def tuning_curve(self, angles: np.ndarray, cell_num: int, orientation: bool = True) -> np.ndarray:
        """Returns the tuning curve for a cell based on the given angles.

        Args:
            angles (np.ndarray): Angle Range in degrees.
            cell_num (int): Cell Number
            orientation (bool, optional): Orientation or Direction. Defaults to True.

        Returns:
            np.ndarray: Computed Tuning Curve
        """
        if orientation:
            fit_params = self.metrics['or_params'][cell_num] if 'or_params' in self.metrics.columns else [np.nan]
            rad_angles = np.deg2rad(angles*2)
            fit_func = OrientationAnalysis.von_mises_or_fit
        else:
            fit_params = self.metrics['dr_params'][cell_num] if 'dr_params' in self.metrics.columns else [np.nan]
            rad_angles = np.deg2rad(angles)
            fit_func = OrientationAnalysis.von_mises_dr_fit

        if np.any(np.isnan(fit_params)):
            return np.nan * np.ones(angles.shape)

        return fit_func(rad_angles, *fit_params)

    def get_orientation_preferences(self, pref_type: str = 'dr_fit', orientation: bool = True, **kwargs) -> np.ndarray:
        """Retrieve Orientation Preferences in Degrees

        Returns cellular orientation preferences in degrees based on metric of either vector sum, orientation fit, or direction fit.
        TODO:
            * Should run the analysis if orientation can not be found as an attribute. Currently this will just crash.
        Args:
            pref_type (str): Type of orientation calculation (either vsum or fit or dr_fit)
            orientation (bool): orientation or direction
            threshold (float): threshold for fitting

        Raises:
            NotImplementedError: Thrown when the angle preference type isn't known

        Returns:
            np.ndarray: List of angle preferences in Degrees
        """

        responsive = kwargs['responsive'] if 'responsive' in kwargs else True
        kwargs['threshold'] = kwargs['threshold'] if 'threshold' in kwargs else -1
        if orientation:
            if pref_type == 'fit':
                if 'or_params' not in self.metrics:
                    self.fit_tuning_curves(orientation=True)
                all_ors = np.rad2deg(np.stack(self.metrics['or_params'].values)[:, -2])/2
                if 'threshold' in kwargs:
                    rsqs = np.stack(self.metrics['or_rsq'].values)
                    all_ors[rsqs <= kwargs['threshold']] = np.nan
            elif pref_type == 'vsum':
                all_ors = np.rad2deg(np.stack(self.metrics['orientation'].values)/2)
            elif pref_type == 'dr_fit':
                if 'dr_params' not in self.metrics:
                    self.fit_tuning_curves(orientation=False)

                all_ors = np.rad2deg(np.stack(self.metrics['dr_params'].values)[:, -2])
                if 'threshold' in kwargs:
                    rsqs = np.stack(self.metrics['dr_rsq'].values)
                    all_ors[rsqs <= kwargs['threshold']] = np.nan
            else:
                raise NotImplementedError('Orientation preference type %s is not implemented!' % pref_type)
            with np.errstate(invalid='ignore'):
                all_ors = np.mod(all_ors, 180)
        else:
            if pref_type == 'fit' or pref_type == 'dr_fit':
                if 'dr_params' not in self.metrics:
                    self.fit_tuning_curves(orientation=False)
                amp_one = np.stack(self.metrics['dr_params'].values)[:, 0]
                amp_two = np.stack(self.metrics['dr_params'].values)[:, 1]
                all_ors = np.rad2deg(np.stack(self.metrics['dr_params'].values)[:, -2])
                all_ors[amp_two > amp_one] = all_ors[amp_two > amp_one] + 180

                if 'threshold' in kwargs:
                    rsqs = np.stack(self.metrics['dr_rsq'].values)
                    all_ors[rsqs <= kwargs['threshold']] = np.nan

            with np.errstate(invalid='ignore'):
                all_ors = np.mod(all_ors, 360)
        if responsive:
            all_ors[np.logical_not(self.responsive())] = np.nan

        return all_ors

    def pairwise_orientation_matrix(self, pref_type: str, orientation: bool = True, **kwargs) -> np.ndarray:
        """Compute the maatrix of orientation preference differences

        Args:
            pref_type (str): Type of preference to use (see get_orientation_preferences)
            orientation (bool, optional): Use orientation preference (True), or direction (False). Defaults to True.
        Kwargs:
            absolute (bool): Absolute difference. Defaults to True
        Returns:
            np.ndarray[float]: absolute orientation Preference differences
        """
        or_matrix = None
        kwargs['absolute'] = kwargs['absolute'] if 'absolute' in kwargs else True
        kwargs['cache'] = kwargs['cache'] if 'cache' in kwargs else True

        if kwargs['cache']:
            if orientation and 'or_diff_matrix' in self.cache:
                or_matrix = self.cache['or_diff_matrix']
            elif not orientation and 'dr_diff_matrix' in self.cache:
                or_matrix = self.cache['dr_diff_matrix']

        if or_matrix is None:
            or_correction = 2 if orientation else 1
            thetas = self.get_orientation_preferences(pref_type, orientation=orientation, **kwargs)
            or_matrix = np.empty((thetas.shape[0], thetas.shape[0]))

            for idx_a, theta_a in enumerate(thetas):
                for idx_b, theta_b in enumerate(thetas):
                    or_matrix[idx_a, idx_b] = np.abs(OrientationAnalysis.angle_diff(theta_a,
                                                                                    theta_b,
                                                                                    orcorrection=or_correction))

            if kwargs['cache']:
                if orientation:
                    self.cache['or_diff_matrix'] = or_matrix
                else:
                    self.cache['dr_diff_matrix'] = or_matrix
        if kwargs['absolute']:
            or_matrix = np.abs(or_matrix)

        return or_matrix

    def well_fit(self, rsq_threshold=0.6, si_threshold=None, orientation=True, pref_type='dr_fit', **kwargs):
        sel_idx = 'OSI' if orientation else 'DSI'
        prefix = 'dr' if pref_type is 'dr_fit' else 'or'

        well_fit_cells = np.ones((self.expt.num_roi(), ), dtype=bool)

        if rsq_threshold is not None:
            rsqs = np.stack(self.metrics[prefix+'_rsq'])
            well_fit_cells = np.logical_and(well_fit_cells, rsqs > rsq_threshold)
        if si_threshold is not None:
            sis = np.stack(self.metrics[sel_idx])
            well_fit_cells = np.logical_and(well_fit_cells, sis > si_threshold)
        return well_fit_cells

    def fraction_of_active_trials(self):

        responsive = self.responsive()

        responses = self.single_trial_responses()
        thresholds = self.threshold(responses=responses)[responsive, :]
        responses = responses[responsive, :-1, :]
        responses = np.reshape(responses, (responses.shape[0], responses.shape[1]*responses.shape[2]))

        threshold_responses = responses > np.tile(thresholds, [1, responses.shape[1]])

        percent_active_trials = np.sum(threshold_responses, axis=1) / threshold_responses.shape[1]

        return percent_active_trials

    # Static Methods

    @staticmethod
    def rad_angle_diff(theta_a: float, theta_b: float, orcorrection: int = 2) -> float:
        """Compute the difference between two angles in radians.

        Args:
            theta_a (float): First angle in radians
            theta_b (float): Second angle in radians
            orcorrection (int, optional): Correction factor for orientation. Defaults to 2.

        Returns:
            [float]: Angle difference in radians
        """
        if orcorrection not in [1, 2]:
            ValueError('ORCorrection must be 1 (direction) or 2 (orientation')
        vector_a = np.exp(orcorrection*1j*theta_a)
        vector_b = np.exp(orcorrection*1j*theta_b)
        return np.angle(vector_a/vector_b)/orcorrection

        return np.angle(vector_a/vector_b)

    @staticmethod
    def angle_diff(theta_a: float, theta_b: float, orcorrection: int = 2) -> float:
        """Compute the difference between two angles in degrees.

        Args:
            theta_a (float): First angle in degrees
            theta_b (float): Second angle in degrees
            orcorrection (int, optional): Correction factor for orientation. Defaults to 2.

        Returns:
            [float]: Angle difference in degrees
        """

        return np.rad2deg(OrientationAnalysis.rad_angle_diff(np.deg2rad(theta_a),
                                                             np.deg2rad(theta_b), orcorrection=orcorrection))

    @staticmethod
    def calculate_dsi_osi_from_drfit(params):

        if np.any(np.isnan(params)):
            return np.nan, np.nan

        angle = params[-2] if params[0] > params[1] else params[-2] + np.pi

        pref_resp = OrientationAnalysis.von_mises_dr_fit(angle, *params)
        anti_pref = OrientationAnalysis.von_mises_dr_fit(angle+np.pi, *params)
        orth_resp = np.mean([OrientationAnalysis.von_mises_dr_fit(angle+np.pi/2, *params),
                             OrientationAnalysis.von_mises_dr_fit(angle-np.pi/2, *params)])

        dsi = np.abs(pref_resp - anti_pref) / (pref_resp + anti_pref)

        osi = (pref_resp - orth_resp) / (pref_resp + orth_resp)

        if osi < 0:
            osi = 0
        elif osi > 1:
            osi = 1

        if dsi < 0:
            dsi = 0
        elif dsi > 1:
            dsi = 1

        return osi, dsi

    # Plotting Functions

    def plot_raster(self, ax=None, **kwargs):

        if ax is None:
            ax = plt.subplot()

        responses = self.single_trial_responses()
        responsive = self.responsive() if kwargs.get('responsive', True) else np.ones(responses.shape[0], dtype=bool)
        responses = responses[responsive, :, :]

        if kwargs.get('sort_orientation', True):
            orientations = np.argsort(self.get_orientation_preferences(orientation=False)[responsive])
            responses = responses[orientations, :, :]

        responses = np.reshape(responses, (responses.shape[0], responses.shape[1]*responses.shape[2]))

        responses = responses / np.repeat(np.expand_dims(np.max(responses, axis=1), axis=1), responses.shape[1], axis=1)

        responses[responses < 0] = 0

        c = ax.imshow(responses, vmin=0, vmax=1, cmap='magma', aspect='auto')
        ax.set_ylabel('Cell Number')
        ax.set_xlabel('Trial Number')
        if kwargs.get('title', None) is not None:
            ax.set_title(kwargs.get('title', None))
        ax.set_xticks(np.arange(0, 161, 40))

        if kwargs.get('colorbar', False):
            plt.colorbar(c)

    def roi_preferences(self, orientation: bool = True, ax: Axes = None, **kwargs):
        orientation = kwargs.get('orientation', True)
        pref_type = kwargs.pop('pref_type', 'fit')

        show_untuned = kwargs.get('show_untuned', True)
        kwargs['threshold'] = kwargs.get('threshold', 0.6)
        kwargs['colorbar'] = kwargs.get('colorbar', False)

        x, y = self.expt.pixel_frame_size()
        kwargs['fov'] = kwargs.get('fov', ((0, x), (y, 0)))

        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        if orientation:
            orientations = self.get_orientation_preferences(pref_type, **kwargs)
            vmin = 0
            vmax = 180
        else:
            raise NotImplementedError('Has not been implemented yet')

        full_frame = np.zeros((x, y, 4))
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='hsv')
        for roi_idx, roi in enumerate(self.expt.roi):
            if np.isnan(orientations[roi_idx]):
                full_frame[roi.mask.todense(), :] = [.75, .75, .75, 1]
            else:
                full_frame[roi.mask.todense(), :] = mapper.to_rgba(orientations[roi_idx])

        ax.imshow(full_frame)

        if kwargs['colorbar']:
            plt.colorbar(mapper, ax=ax)

    # tif Functions

    def vector_sum_trial_images(self, **kwargs):
        selectivity_map = None
        responsivity_map = None
        selectivity_max = kwargs.get('selectivity_max', 99.9)  # percentile to clip selectivity
        responsivity_max = kwargs.get('responsivity_max', 99.9)

        trial_images = self.get_trial_image(prepad=0, postpad=0)

        if self.expt.do_blank():
            trial_images = trial_images[:-1, :, :, :, :]

        trial_images = np.mean(trial_images, axis=2)

        if kwargs.get('orientation', True):
            trial_images = np.concatenate((trial_images[:trial_images.shape[0]//2, :, :, :],
                                           trial_images[trial_images.shape[0]//2:, :, :, :]),
                                          axis=1)

        avg_imgs = np.median(trial_images, axis=1)

        vectors = np.exp(1j*np.arange(0, 2*np.pi, 2*np.pi/avg_imgs.shape[0]))
        vectors = np.tile(np.expand_dims(vectors, axis=(1, 2)),
                          [1, avg_imgs.shape[1], avg_imgs.shape[2]])

        vector_sum = np.sum(avg_imgs * vectors, axis=0)
        norm_img = Normalize(vmin=0, vmax=2*np.pi, clip=True)(np.mod(np.angle(vector_sum), 2*np.pi))
        color_map = plt.cm.hsv
        color_img = color_map(norm_img)
        hsv_img = rgb2hsv(color_img[:, :, 0:3])

        if kwargs.get('responsivity', True):
            responsivity_map = np.nanmax(avg_imgs, axis=0)
            hsv_img[:, :, 2] = Normalize(vmin=0,
                                         vmax=np.nanpercentile(responsivity_map, responsivity_max),
                                         clip=True)(responsivity_map)

        if kwargs.get('selectivity', True):
            selectivity_map = np.abs(vector_sum) / np.sum(avg_imgs, axis=0)
            hsv_img[:, :, 1] = Normalize(vmin=0,
                                         vmax=kwargs.get(selectivity_max, 1),
                                         clip=True)(selectivity_map)

        rgb_img = hsv2rgb(hsv_img)

        if kwargs.get('upscale', False):

            rgb_img = pyramid_expand(rgb_img, upscale=kwargs.get('upscale', 1), multichannel=True)

        return rgb_img, vector_sum, selectivity_map, responsivity_map

from itertools import combinations
import copy
from collections import defaultdict
import logging
from os import stat
import numpy as np

from matplotlib.pyplot import colorbar
from numpy.core.defchararray import _to_string_or_unicode_array
from skimage.draw import circle_perimeter
from skimage.measure import find_contours
from fleappy.analysis.epi import EpiAnalysis
from fleappy.filter import fermi
from fleappy.analysis.epimapstructure import fit_wavelet


class EpiBlockwiseAnalysis(EpiAnalysis):
    """Class for the handling of blockwise stimuli.

    Class for the analysis of stimuli which can be subdivided into stimulus x trials. This is a subclass of the
    BaseAnalysis class.

    Attributes:
        stim_period (tuple): Stimulus period (start, stop).
        prepad (float): Period before stimulus onset to analyze.
        postpad (float): Period after stimulus offset to analyze.
        analysis_period (tuple): Period during stimulus to analyze (start, stop).
        cache (defaultdict): Dictionary which stores computed analysis information. Default values are None.
    """

    __slots__ = ['stim_period', 'prepad',
                 'postpad', 'analysis_period']

    def __init__(self, expt, field, analysis_period=(0, -1), prepad=0, postpad=0, **kwargs):

        super().__init__(expt, field, **kwargs)

        if isinstance(self.expt, list):
            if analysis_period[1] == -1:
                self.analysis_period = [(analysis_period[0], float(
                    ex.metadata.stim_duration())) for ex in self.expt]
            else:
                self.analysis_period = [analysis_period for _ in self.expt]
            self.stim_period = [(0, ex.metadata.stim_duration())
                                for ex in self.expt]
        else:
            self.analysis_period = analysis_period
            self.analysis_period = (self.analysis_period[0], float(
                self.expt.metadata.stim_duration()))
            self.stim_period = (0, expt.metadata.stim_duration)
        self.prepad = prepad
        self.postpad = postpad

    def run(self, **kwargs):
        kwargs['cache'] = kwargs.pop('cache', True)
        kwargs['override'] = kwargs.pop('override', False)

        super().run(**kwargs)

        if kwargs.get('pattern_variabiliity', True):
            logging.info('[%s] (%s):Computing Pattern Variability', self.expt.animal_id, self.id)
            _ = self.trial_to_trial_matrix(**kwargs)
            _ = self.pattern_variability(**kwargs)

        if kwargs.get('binned_correlations', True):
            roi_center = self.roi_centroid()
            for corr_type in ['total', 'signal', 'noise']:
                logging.info('[%s] (%s):Computing %s Correlations', self.expt.animal_id, self.id, corr_type)
                _ = self.correlation_map(roi_center, corr_type=corr_type, drop_blank=True, **kwargs)
                _ = self.binned_correlations(corr_type=corr_type, **kwargs)

        if kwargs.get('significantly_responsive', True):
            self.significantly_responsive_pixels()

        if kwargs.get('wavelet', True):
            self.wavelet_analysis(**kwargs)

    def single_trial_response_timecourse(self, **kwargs) -> np.ndarray:
        prepad = kwargs.get('prepad', self.prepad)
        postpad = kwargs.get('postpad', self.postpad)
        field = kwargs.get('field', self.field)

        trial_responses, trial_masks = self.expt.get_trial_responses(prepad=prepad,
                                                                     postpad=postpad,
                                                                     dff=kwargs.get('dff', False),
                                                                     series_type=field,
                                                                     prestim_bl=kwargs.get('prestim_bl', True),
                                                                     mask=False)

        trial_responses[~np.isfinite(trial_responses)] = 0

        if kwargs.get('filter', True) and 'cutoffs' in self.filter_params and 'resolution' in self.filter_params:
            logging.info('[%s] Filtering Responses...', self.expt.animal_id)
            trial_responses = fermi.filter(trial_responses,
                                           cutoffs=self.filter_params['cutoffs'],
                                           resolution=self.filter_params['resolution'])

        if kwargs.get('clip', True):
            trial_responses[trial_responses < 0] = 0

        roi_flag = kwargs.get('roi', True)
        if roi_flag is not None and roi_flag is not False:
            roi_mask = self.expt.roi(field)
            trial_responses[:, :, :, ~roi_mask] = np.nan
        return trial_responses, trial_masks

    def single_trial_responses(
            self, **kwargs) -> np.ndarray:
        """Return matrix of trial responses

        Args:
            dff (bool, optional): Flag to compute delta F/F based on prestimulus period. Defaults to False.
            prestim_bl (bool, optional): Flag to subtract prestimulus baseline . Defaults to False.
            cache (bool, optional): Flag to store result in analysis cache. N.B. This has a high memory overhead. Defaults to True.
            overwrite (bool, optional): Flag to recompute even if cached. Defaults to False.


        Returns:
            np.ndarray: Trial responses in stim,trial,y,x order
        """
        cache = kwargs.pop('cache', True)
        overwrite = kwargs.pop('overwrite', False)
        dff = kwargs.pop('dff', False)
        prestim_bl = kwargs.pop('prestim_bl', False)
        clip = kwargs.pop('clip', True)
        filter_resps = kwargs.pop('filter', True)
        roi = kwargs.pop('roi', True)

        if not filter_resps:
            cache = False

        if not overwrite and self.cache.get('single_trial_responses') is not None and cache:
            logging.debug('[%s] Loading Single Trial Responses from Cache', self.expt.animal_id)
            trial_responses = self.cache.get('single_trial_responses')
        else:
            logging.info('[%s] Loading single trial responses from files', self.expt.animal_id)
            frame_rate = self.expt.frame_rate()
            prepad_frames = int(np.round(self.prepad*frame_rate))

            analysis_start = prepad_frames + int(np.round(self.analysis_period[0]*frame_rate))
            analysis_stop = prepad_frames + int(np.round(self.analysis_period[1]*frame_rate))

            trial_responses, _ = self.single_trial_response_timecourse(**kwargs)

            trial_responses = np.mean(trial_responses[:, :, analysis_start:analysis_stop, :, :], axis=2)
            if cache:
                logging.debug('Saving Single Trial Responses to Cache')
                self.cache['single_trial_responses'] = copy.deepcopy(trial_responses)

        if roi is not None and roi is not False:
            roi_mask = self.roi()
            trial_responses[:, :, ~roi_mask] = np.nan

        return trial_responses

    def significantly_responsive_pixels(self, **kwargs):
        """Get the mask of significantly responsive pixels.

        kwargs:
            num_stds (float): Number of Standard Deviations over the normal a mean response must be

        Raises:
            NotImplementedError: [description]
        """
        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['roi'] = kwargs.get('roi', True)
        num_stds = kwargs.pop('num_stds', 4)
        cache_field = 'sig_responsive_pixels'

        if self.cache.get(cache_field) is not None:
            sig_responses = copy.copy(self.cache.get(cache_field))
        else:
            if not self.expt.do_blank():
                raise NotImplementedError(
                    '[%s] I do not know how to assess responsivity if there is no blank trial!', self.expt.animal_id)
            else:
                single_trials = self.single_trial_responses()
                blank_trial = single_trials[-1, :, :, :]
                maximal_resp = np.max(np.nanmean(single_trials, axis=1), axis=0)
                threshold = np.nanmean(blank_trial, axis=0) + num_stds * np.nanstd(blank_trial, axis=0)

                sig_responses = maximal_resp > threshold
        if kwargs['roi']:
            roi = self.roi()
            sig_responses[~roi] = np.nan

        return sig_responses

    def stimulus_avg_responses(self, **kwargs):
        """Returns the stimulus average responses.

        **kwargs
            dff (bool, optional): Flag to compute delta F/F based on prestimulus period. Defaults to False.
            prestim_bl (bool, optional): Subtract prestimulus baseline fluorescence. Defaults to True.
            cache (bool, optional): Use cached data. Defaults to True.

        Returns:
            np.array: Stimulus response averages (stim,  y, x )
        """

        kwargs['cache'] = True if 'cache' not in kwargs else kwargs['cache']
        kwargs['dff'] = False if 'dff' not in kwargs else kwargs['dff']
        kwargs['prestim_bl'] = True if 'prestim_bl' not in kwargs else kwargs['prestim_bl']
        kwargs['filter'] = kwargs.get('filter', True)
        roi = kwargs.pop('roi', True)  # apply the roi at the top level if needed
        kwargs['roi'] = False

        if kwargs['filter'] is False:  # never cache unfiltered responses
            kwargs['cache'] = False

        if kwargs['filter'] and self.cache.get('stimulus_average') is not None:
            avg_responses = self.cache.get('stimulus_average')
        else:
            trial_responses = self.single_trial_responses(**kwargs)
            avg_responses = np.nanmean(trial_responses, axis=1)

        if kwargs['cache']:
            self.cache['stimulus_average'] = copy.deepcopy(avg_responses)

        if roi is not None and roi:
            roi_mask = self.roi()
            avg_responses[:, ~roi_mask] = np.nan

        return avg_responses

    def pattern_variability(self, **kwargs):
        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['dff'] = kwargs.get('dff', False)
        kwargs['prestim_bl'] = True if 'prestim_bl' not in kwargs else kwargs['prestim_bl']
        kwargs['roi'] = True if 'roi' not in kwargs else kwargs['roi']
        kwargs['override'] = False if 'override' not in kwargs else kwargs['override']
        kwargs['orientation'] = False if 'orientation' not in kwargs else kwargs['orientation']
        kwargs['clip_negative'] = True if 'clip_negative' not in kwargs else kwargs['clip_negative']
        kwargs['mask_invalid'] = True if 'mask_invalid' not in kwargs else kwargs['mask_invalid']
        kwargs['keep_direction'] = kwargs.get('keep_direction', True)

        if not kwargs['override'] and kwargs['cache'] and 'variability_metric' in self.cache:
            if isinstance(self.cache['variability_metric'], tuple):
                return self.cache['variability_metric']

        corr_matrix = self.trial_to_trial_matrix(**kwargs)

        num_trials = self.expt.num_trials()
        num_stims = self.expt.num_stims()
        if self.expt.do_blank():
            num_stims = num_stims-1

        matched_mask = np.zeros(corr_matrix.shape, dtype=bool)

        for stim_id in range(num_stims):
            matched_mask[stim_id*num_trials:(stim_id+1)*num_trials, stim_id*num_trials:(stim_id+1)*num_trials] = True

        if kwargs['orientation']:
            or_mask = np.roll(matched_mask, num_stims//2*num_trials)
            or_mask[(num_stims//2)*num_trials:, :] = False  # remove the duplicate of the roll
            matched_mask = np.logical_or(matched_mask,
                                         np.roll(matched_mask, num_stims//2*num_trials))

        if kwargs['keep_direction']:
            orth_mask = np.logical_or(np.roll(matched_mask, num_stims//4*num_trials),
                                      np.roll(matched_mask, -num_stims//4*num_trials))
        else:
            orth_mask = np.roll(matched_mask, num_stims//2*num_trials)

        matched_mask[np.identity(matched_mask.shape[0], dtype=bool)] = False

        matched_metric = np.nanmean(corr_matrix[matched_mask])
        orth_metric = np.nanmean(corr_matrix[orth_mask])

        if kwargs['cache']:
            self.cache['variability_metric'] = (matched_metric, orth_metric)
        return matched_metric, orth_metric

    def trial_to_trial_matrix(self, **kwargs):
        kwargs['cache'] = True if 'cache' not in kwargs else kwargs['cache']
        kwargs['dff'] = False if 'dff' not in kwargs else kwargs['dff']
        kwargs['prestim_bl'] = True if 'prestim_bl' not in kwargs else kwargs['prestim_bl']
        kwargs['roi'] = True if 'roi' not in kwargs else kwargs['roi']
        kwargs['override'] = False if 'override' not in kwargs else kwargs['override']
        kwargs['orientation'] = False if 'orientation' not in kwargs else kwargs['orientation']
        kwargs['clip_negative'] = True if 'clip_negative' not in kwargs else kwargs['clip_negative']
        kwargs['mask_invalid'] = True if 'mask_invalid' not in kwargs else kwargs['mask_invalid']

        if not kwargs['override'] and kwargs['cache'] and 'trial_to_trial_corr_matrix' in self.cache:
            logging.info('[%s] Skipping Recomputation of trial-to-trial correlation matrix', self.expt.animal_id)
            return self.cache['trial_to_trial_corr_matrix']

        trial_resps = self.single_trial_responses(**kwargs)  # stims x trials x y x x
        if self.expt.metadata.do_blank():
            trial_resps = trial_resps[:-1, :, :, :]
        if kwargs['roi']:
            roi = self.expt.roi(self.field)
        else:
            roi = np.ones((trial_resps.shape[2], trial_resps.shape[3]), dtype=bool)

        if kwargs['mask_invalid']:
            invalids = np.any(np.logical_not(np.isfinite(trial_resps)), axis=(0, 1))
            roi = roi & np.logical_not(invalids)

        if kwargs['orientation'] and self.expt.metadata.stim_type() == 'driftingGrating':
            trial_resps = np.concatenate((trial_resps[0:int(trial_resps.shape[0]/2), :, :, :],
                                          trial_resps[int(trial_resps.shape[0]/2):, :, :, :]), axis=1)
        num_stims, num_trials, y, x = trial_resps.shape
        trial_resps = np.reshape(trial_resps, (num_stims*num_trials, y, x))
        corr_matrix = np.ones((num_stims*num_trials, num_stims*num_trials))

        for idx_a in range(corr_matrix.shape[0]):
            for idx_b in range(idx_a+1, corr_matrix.shape[0]):
                corr_matrix[idx_a, idx_b] = np.corrcoef(trial_resps[idx_a, roi], trial_resps[idx_b, roi])[0][1]
                corr_matrix[idx_b, idx_a] = corr_matrix[idx_a, idx_b]

        if kwargs['cache']:
            self.cache['trial_to_trial_corr_matrix'] = corr_matrix
        return corr_matrix

    def correlation_map(self, seed_pos, corr_type='total', **kwargs):
        kwargs['roi'] = True  # forces correlations only for roi

        seed_pos = tuple(seed_pos)     # y,x
        cache_key = corr_type + '_corr_matrix'
        cache = kwargs.pop('cache', True)

        if cache and not kwargs.get('override', False):
            if cache_key in self.cache and seed_pos in self.cache[cache_key]:
                logging.info('[%s] (%s): Getting Correlation Map from cache', self.expt.animal_id, self.id)
                return self.cache[cache_key][seed_pos]

        y, x = self.expt.pixel_frame_size(self.field)
        correlation_pattern = np.full((y*x), np.nan)
        single_trial_responses = self.single_trial_responses(cache=True, **kwargs)
        if kwargs.get('drop_blank', False):
            single_trial_responses = single_trial_responses[:-1, :, :, :]
        single_trial_responses = np.transpose(single_trial_responses, (2, 3, 0, 1))

        if corr_type == 'total':
            single_trial_responses = np.reshape(single_trial_responses,
                                                (single_trial_responses.shape[0], single_trial_responses.shape[1], single_trial_responses.shape[2] * single_trial_responses.shape[3]))

        elif corr_type == 'signal':
            single_trial_responses = np.mean(single_trial_responses, axis=3)
        elif corr_type == 'noise':
            avg_responses = np.tile(np.expand_dims(np.mean(single_trial_responses, axis=3),
                                                   axis=3), single_trial_responses.shape[3])
            single_trial_responses = single_trial_responses - avg_responses

            single_trial_responses = np.reshape(single_trial_responses,
                                                (single_trial_responses.shape[0], single_trial_responses.shape[1], single_trial_responses.shape[2] * single_trial_responses.shape[3]))

        flattened_responses = np.reshape(single_trial_responses, (x*y, single_trial_responses.shape[2]))

        roi = np.reshape(self.roi(),
                         (single_trial_responses.shape[0] * single_trial_responses.shape[1]))

        seed_response = single_trial_responses[seed_pos[0], seed_pos[1], :]
        correlations = np.empty((np.sum(roi), ))
        for idx, resp in enumerate(flattened_responses[roi, :]):
            correlations[idx] = np.corrcoef(resp, seed_response)[0][1]
        correlation_pattern[roi] = correlations
        correlation_patterns = np.reshape(correlation_pattern, (y, x))

        if cache:
            if cache_key not in self.cache:
                self.cache[cache_key] = {}
            self.cache[cache_key][seed_pos] = correlation_patterns

        return correlation_patterns

    def binned_correlations(self, corr_type='total', **kwargs):
        """Compute the mean correlation for distance bins

        Correlation of type (corr_type) for different distances. To save on computation currently only uses circle perimeter encompassing seed points.

        Kwargs:
            spacing (float): Grid Spacing to use as seed points. Defaults to 0.3 (mm)
            bins (float): Array of distances to compute. Defaults to [.5-3) with 0.5 spacing (mm).
            override (bool): Force recompute. Defaults to False
            cache (float): Load/Save to cache. Defaults to True

        Args:
            corr_type (str, optional): Correlation type to use ('total', 'signa', 'noise').  Defaults to 'total'.

        Returns:
            bins (np.ndarray): Distance bins in mm
            statistic (np.ndarray): Average corrlation for each bin.
        """
        kwargs['spacing'] = kwargs.get('spacing', 0.15)  # in mm
        kwargs['bins'] = kwargs.get('bins', np.arange(.1, 2.1, .1))  # bin sizes in mm
        kwargs['override'] = kwargs.get('override', False)
        cache = kwargs.get('cache', True)
        kwargs['cache'] = True
        frame_size = self.pixel_frame_size()
        resolution = self.resolution()[0] * 1e-3  # mm/pixel
        roi = self.roi()

        if cache and corr_type+'_binned_data' in self.cache and not kwargs['override']:
            return self.cache[corr_type+'_binned_data']

        seed_pos_list = kwargs.get('get_seed_pos_list', self.grid_roi(kwargs['spacing']))

        diameters = np.round(kwargs['bins']/resolution).astype(int)
        binned_data = np.full((len(seed_pos_list), len(diameters)), np.nan)
        logging.info('[%s] Running Binned Correlations', self.expt.animal_id)
        for seed_idx, seed_pos in enumerate(seed_pos_list):
            logging.debug('[%s] Running seed %i of %i', self.expt.animal_id, seed_idx, seed_pos_list.shape[0])
            corr_matrix = self.correlation_map(seed_pos, corr_type=corr_type, **kwargs)
            for diam_idx, diameter in enumerate(diameters):
                y, x = circle_perimeter(seed_pos[0], seed_pos[1], diameter)

                out_of_frame_pixels = ((y < 0) | (y >= frame_size[0])) | ((x < 0) | (x >= frame_size[1]))
                y = y[~out_of_frame_pixels]
                x = x[~out_of_frame_pixels]

                points_to_avg = np.zeros(frame_size, dtype=bool)
                points_to_avg[y, x] = 1
                points_to_avg = points_to_avg & roi
                if np.any(points_to_avg):
                    binned_data[seed_idx, diam_idx] = np.nanmean(corr_matrix[points_to_avg])

        binned_data = np.nanmean(binned_data, axis=0)
        if cache:
            self.cache[corr_type+'_binned_data'] = (kwargs['bins'], binned_data)

        return kwargs['bins'], binned_data

    def wavelet_analysis(self, corr_type='total', **kwargs):
        kwargs['k_base'] = kwargs.get('k_base', 7)
        kwargs['seed_points'] = kwargs.get('seed_points', self.cache[f'{corr_type}_corr_matrix'].keys())
        if 'wavelet_analysis' not in self.cache or not isinstance(self.cache['wavelet_analysis'], dict):
            self.cache['wavelet_analysis'] = {}
        roi = self.roi()
        resolution = self.resolution()[0]
        min_x, max_x, min_y, max_y = self._roi_bounding()
        trimmed_roi = roi[min_x:max_x, min_y:max_y]
        for key in kwargs['seed_points']:
            if key in self.cache['wavelet_analysis'] and not kwargs.get('overwrite', False):
                continue
            logging.info('[%s] Processing %s', self.expt.animal_id, str(key))
            corr_img = self.correlation_map(key, corr_type=corr_type, **kwargs)
            corr_img[~np.isfinite(corr_img)] = 0
            trimmed_img = corr_img[min_x:max_x, min_y:max_y] - np.mean(corr_img[roi])
            trimmed_img[~trimmed_roi] = 0
            self.cache['wavelet_analysis'][key] = fit_wavelet(trimmed_img,
                                                              resolution,
                                                              roi=trimmed_roi,
                                                              **kwargs)

    def wavelet_analysis_uniform(self, corr_type='total', **kwargs):
        kwargs['k_base'] = kwargs.get('k_base', 7)

        if 'wavelet_analysis_uniform' not in self.cache or not isinstance(self.cache['wavelet_analysis_uniform'], dict):
            self.cache['wavelet_analysis_uniform'] = {}
        roi = self.roi()
        resolution = self.resolution()[0]
        min_x, max_x, min_y, max_y = self._roi_bounding()
        trimmed_roi = roi[min_x:max_x, min_y:max_y]

        logging.info('[%s] Processing %s', self.expt.animal_id, 'uniform')

        self.cache['wavelet_analysis_uniform'] = fit_wavelet(trimmed_roi.astype(np.float32),
                                                             resolution,
                                                             roi=trimmed_roi,
                                                             **kwargs)

    # Plotting Functions

    def plot_trial_to_trial_correlation_matrix(self, ax, **kwargs) -> None:
        kwargs['cmap'] = kwargs.get('cmap', 'bwr')
        kwargs['showtrials'] = kwargs.get('showtrials', True)
        trial_matrix = self.trial_to_trial_matrix(**kwargs)

        c = ax.imshow(trial_matrix,
                      vmin=-1, vmax=1, cmap=kwargs['cmap'])

        cbar = colorbar(c, ax=ax, ticks=[-1, 0, 1])
        cbar.ax.set_ylabel('Pattern Correlation')

        num_stims = self.expt.num_stims()-1 if self.expt.do_blank() else self.expt.num_stims()
        num_trials = self.expt.num_trials()

        if kwargs['showtrials']:
            ax.hlines(np.arange(-0.5, num_stims*num_trials+1, num_trials),
                      -0.5, num_stims*num_trials,
                      linewidth=0.75, linestyles='dashed')
            ax.vlines(np.arange(-0.5, num_stims*num_trials+1, num_trials),
                      -0.5, num_stims*num_trials,
                      linewidth=0.75, linestyles='dashed')
            ax.set_xlim((-0.5, num_stims*num_trials-0.5))
            ax.set_ylim((-0.5, num_stims*num_trials-0.5))
            ax.set_xticks(np.arange(0, num_stims*num_trials+1, 40))
            ax.set_yticks(np.arange(0, num_stims*num_trials+1, 40))

            ax.set_xlabel('Trial Number')
            ax.set_ylabel('Trial Number')

    def plot_correlation_map(self, ax,  location=None, **kwargs):
        corr_type = kwargs.pop('corr_type', 'total')

        axis_options = {'fov': kwargs.pop('fov', None),
                        'plot_cbar': kwargs.pop('colorbar', True),
                        'scalebar': kwargs.pop('scalebar', None),
                        'resolution': self.resolution()[0]
                        }
        kwargs['vmin'] = kwargs.get('vmin', -1)
        kwargs['vmax'] = kwargs.get('vmax', 1)
        kwargs['cmap'] = kwargs.get('cmap', 'bwr')
        if location is None:
            location = self.roi_centroid()

        corr_map = self.correlation_map(location, corr_type=corr_type)

        axis_options['c_object'] = ax.imshow(corr_map, **kwargs)
        ax.scatter(location[1], location[0], marker='s', color='lime')

        cbar = EpiBlockwiseAnalysis._configure_image_axis(ax,
                                                          **axis_options)

        if cbar is not None:
            cbar.set_label(f'Pixelwise {corr_type.capitalize()} Correlation', rotation=270)

    def plot_single_trial(self, ax, stim, trial, **kwargs):

        kwargs['contour'] = kwargs.get('contour', None)
        kwargs['contour_color'] = 'green'

        responses = kwargs.pop('responses', self.single_trial_responses())

        axis_options = {'fov': kwargs.pop('fov', None),
                        'plot_cbar': kwargs.pop('colorbar', True),
                        'scalebar': kwargs.pop('scalebar', None),
                        'resolution': self.resolution()[0]}

        axis_options['c_object'] = ax.imshow(responses[stim, trial, :, :],
                                             cmap='gray',
                                             vmin=kwargs.get('vmin', 0),
                                             vmax=kwargs.get('vmax', np.nanmax(responses)))

        cbar = EpiBlockwiseAnalysis._configure_image_axis(ax, **axis_options)

        if kwargs['contour'] is not None:
            contours = find_contours(responses[stim, trial, :, :], kwargs['contour'])

            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color=kwargs['contour_color'])

        if cbar is not None:
            cbar.set_label(f'Normalized Response', rotation=270)

    @staticmethod
    def _corr2(mat_a, mat_b, **kwargs):

        kwargs['mask_nan'] = True if 'mask_nan' not in kwargs else kwargs['mask_nan']
        kwargs['clip_negative'] = False if 'clip_negative' not in kwargs else kwargs['clip_negative']

        if kwargs['mask_nan']:
            mat_a[np.logical_not(np.isfinite(mat_a))] = 0
            mat_b[np.logical_not(np.isfinite(mat_b))] = 0

        if kwargs['clip_negative']:
            mat_a[mat_a < 0] = 0
            mat_b[mat_b < 0] = 0

        mean2_a = np.sum(mat_a) / np.size(mat_a)
        mean2_b = np.sum(mat_b) / np.size(mat_b)

        mat_a = mat_a - mean2_a
        mat_b = mat_b - mean2_b

        return (mat_a*mat_b).sum() / np.sqrt((mat_a*mat_a).sum() * (mat_b*mat_b).sum())

    @staticmethod
    def _configure_image_axis(ax, scalebar=None, plot_cbar=True, fov=None, c_object=None, resolution=None):
        if fov is not None:
            ax.set_ylim(fov[0])
            ax.set_xlim(fov[1])
        else:
            fov = (ax.get_ylim(), ax.get_xlim())

        if scalebar is not None:
            scale_bar_length = scalebar/resolution  # pixels
            ax.plot([fov[1][1]-10-scale_bar_length, fov[1][1]-10],
                    [fov[0][0]-10, fov[0][0]-10],
                    color='k')

        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        cbar = None
        if plot_cbar and c_object is not None:
            cbar = colorbar(c_object, ax=ax)
        return cbar

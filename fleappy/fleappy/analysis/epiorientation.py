import copy
from skimage.color import rgb2hsv, hsv2rgb
import logging
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

from fleappy.analysis.epiblockwise import EpiBlockwiseAnalysis
from fleappy.analysis.epimapstructure import fit_wavelet


def vector_sum(array):

    num_stims = array.shape[0]
    vector_map = np.zeros((array.shape), dtype=np.complex)
    for stim in range(num_stims):
        vector_map[stim, :, :] = np.exp(1j * stim * 2*np.pi/num_stims) * array[stim, :, :]
    return np.sum(vector_map, axis=0)


def compute_cohens_d(single_trials):
    num_stims = single_trials.shape[0]
    num_trials = single_trials.shape[1]
    mu_theta_x = np.nanmean(single_trials, axis=1)
    mu_x = np.nanmean(mu_theta_x, axis=0)

    sigma_theta_x = np.sqrt(np.sum((single_trials - np.tile(np.expand_dims(mu_theta_x, axis=1),
                                                            (1, num_trials, 1, 1)))**2, axis=1) / (num_trials-1))

    sigma_x = np.sqrt(np.sum(np.sum((single_trials - np.tile(np.expand_dims(mu_theta_x, axis=1),
                                                             (1, num_trials, 1, 1)))**2, axis=1), axis=0) / (num_stims * num_trials-1))
    numerator = (mu_theta_x - np.tile(np.expand_dims(mu_x, axis=0), (num_stims, 1, 1)))

    denominator_a = (num_trials-1)*sigma_theta_x**2
    denominator_b = np.tile(np.expand_dims((num_stims*num_trials-1) * sigma_x**2, axis=0), (num_stims, 1, 1))
    denominator = (denominator_a - denominator_b) / ((num_stims+1)*num_trials - 2)
    cohens_d = numerator / denominator
    return cohens_d


class EpiOrientationAnalysis(EpiBlockwiseAnalysis):

    def __init__(self, expt, field, **kwargs):
        super().__init__(expt, field, **kwargs)

    def run(self, **kwargs):
        """Run orientation analysis for epi experiment.
        """
        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['override'] = kwargs.get('override', False)

        if kwargs.get('difference_maps', True):
            for orientation in [True, False]:
                self.difference_maps(orientation=orientation, normalize=True, **kwargs)

        if kwargs.get('vector_sum', True):
            _ = self.vector_sum(orientation=True, **kwargs)

        # if kwargs.get('calculate_wavelength', True):
            # self.calculate_wavelength()

        if kwargs.get('sig_tuning', True):
            self.significant_tuned_pixels(orientation=True)

        super().run(**kwargs)

    def vector_sum(self, **kwargs) -> tuple:
        """Computes the vector sum for a set of sequential stimulus
        Args:
            orientation (bool, optional): Flag for collapsing into orientation space. Defaults to True.
            dff (bool, optional): Compute delta F/F based on prestimulus period. Defaults to False.
            cache (bool, optional): Flag to save results to analysis cache. Defaults to True.
            filter_params (dict, optional): Parameters for filter responses using fermi filter requires cutoffs and resolution. Defaults to None.

        Returns:
            tuple: Vector Map, Angle Map (0-2pi), Responsivity Map (absolute), Selectivity Map (0-1)
        """
        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['filter'] = kwargs.get('filter', True)
        kwargs['orientation'] = kwargs.get('orientation', True)
        roi = kwargs.pop('roi', True)
        kwargs['roi'] = False

        prefix = 'or_' if kwargs['orientation'] else 'dr_'
        cache_keys = [prefix+key for key in ['vector_map', 'angle_map', 'responsivity_map', 'selectivity_map']]

        if kwargs['filter'] and all(key in self.cache.keys() for key in cache_keys):
            vector_map = self.cache[prefix+'vector_map']
            angle_map = self.cache[prefix+'angle_map']
            responsivity_map = self.cache[prefix+'responsivity_map']
            selectivity_map = self.cache[prefix+'selectivity_map']
        else:

            avg_stim_resp = self.stimulus_avg_responses(**kwargs)

            if self.expt.do_blank():
                avg_stim_resp = avg_stim_resp[:-1, :, :]

            if kwargs['orientation']:
                avg_stim_resp = (avg_stim_resp[:avg_stim_resp.shape[0]//2, :, :] +
                                 avg_stim_resp[avg_stim_resp.shape[0]//2:, :, :]) / 2

            vector_map = vector_sum(avg_stim_resp)
            responsivity_map = np.sum(avg_stim_resp, axis=0)

            angle_map = np.mod(np.angle(vector_map), 2*np.pi)
            selectivity_map = np.abs(vector_map)/responsivity_map

            if kwargs['cache']:
                self.cache[prefix+'vector_map'] = copy.deepcopy(vector_map)
                self.cache[prefix+'angle_map'] = copy.deepcopy(angle_map)
                self.cache[prefix+'responsivity_map'] = copy.deepcopy(responsivity_map)
                self.cache[prefix+'selectivity_map'] = copy.deepcopy(selectivity_map)
        if roi:
            vector_map[~self.roi()] = np.nan
            angle_map[~self.roi()] = np.nan
            responsivity_map[~self.roi()] = np.nan
            selectivity_map[~self.roi()] = np.nan

        return vector_map, angle_map, responsivity_map, selectivity_map

    def significant_tuned_pixels(self, shuffles=1000, **kwargs) -> np.ndarray:
        """Compute boolean image of significantly tuned pixels, based on 1-CV 

        Args:
            shuffles (int, optional): [description]. Defaults to 1000.

        kwargs:
            orientations (bool): Vector Sum in Orientation Space.
            cache (bool): Store/Retrieve result from cache.
            override (bool): Force recomputation.
        Returns:
            np.ndarray: [description]
        """
        kwargs['orientation'] = kwargs.get('orientation', True)
        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['override'] = kwargs.get('override', False)
        cache_field = 'or_sig_tuned' if kwargs['orientation'] else 'dr_sig_tuned'

        if cache_field in self.cache and not kwargs['override']:
            logging.info('[%s] 1-CV shuffles retrieved from cache', self.expt.animal_id)
            return self.cache[cache_field]

        logging.info('[%s] Calculating 1-CV shuffles', self.expt.animal_id)
        kwargs['override'] = False
        _, _, _, one_minus_cv = self.vector_sum(**kwargs)
        single_trials = self.single_trial_responses(**kwargs)  # stims x trials x X x Y
        if self.expt.do_blank():
            single_trials = single_trials[:-1, :, :, :]

        if kwargs['orientation']:
            single_trials = np.concatenate((single_trials[:single_trials.shape[0]//2, :, :, :],
                                            single_trials[single_trials.shape[0]//2:, :, :, :]),
                                           axis=1)

        single_trials[single_trials < 0] = 0
        stims, trials, y_size, x_size = single_trials.shape

        single_trials = np.transpose(single_trials, (2, 3, 0, 1))
        single_trials = np.reshape(
            single_trials, (y_size, x_size, stims*trials))
        total_trials = np.arange(0, single_trials.shape[2], dtype=int)
        one_minus_cv_shuffles = np.full((single_trials.shape[0], single_trials.shape[1], shuffles), np.nan)

        for shuf_num in range(shuffles):
            np.random.shuffle(total_trials)
            shuffle_trials = np.mean(np.reshape(
                single_trials[:, :, total_trials], (y_size, x_size, stims, trials)), axis=3)

            vsum = np.abs(vector_sum(np.transpose(shuffle_trials, (2, 0, 1)))) / np.sum(shuffle_trials, axis=2)

            one_minus_cv_shuffles[:, :, shuf_num] = vsum

        sig_tuned = one_minus_cv > np.nanpercentile(one_minus_cv_shuffles, 95, axis=2)
        sig_tuned[np.isnan(sig_tuned)] = 0

        if kwargs['cache']:
            self.cache[cache_field] = sig_tuned

        return sig_tuned

    def cohens_d_difference_maps(self, **kwargs) ->np.ndarray:

        kwargs['orientation'] = kwargs.get('orientation', True)
        kwargs['clip'] = kwargs.get('clip', True)
        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['roi'] = kwargs.get('roi', True)
        kwargs['override'] = kwargs.get('override', False)
        kwargs['filter'] = kwargs.get('filter', True)

        if not kwargs['filter']:
            kwargs['cache'] = False

        cache_field = 'or_cohens_difference_maps' if kwargs['orientation'] else 'dr_cohens_difference_maps'
        if kwargs['cache'] and not kwargs['override'] and cache_field in self.cache:
            cohens_d = self.cache[cache_field]
        else:
            single_trials = self.single_trial_responses(**kwargs)

            if self.expt.do_blank():
                single_trials = single_trials[:-1, :, :, :]
            num_stims = single_trials.shape[0]

            if kwargs['orientation']:
                single_trials = (single_trials[0:num_stims//2, :, :] + single_trials[num_stims//2:num_stims, :, :])/2
                num_stims = num_stims//2

            cohens_d = compute_cohens_d(single_trials)

        if kwargs['cache']:
            self.cache[cache_field] = cohens_d

        if kwargs['roi']:
            cohens_d[:, ~np.isnan(self.roi())] = np.nan

        return cohens_d

    def difference_maps(self, **kwargs) -> np.ndarray:
        """Returns difference maps for either orientation or direction

        kwargs:
            orientation (bool, optional): [description]. Defaults to True.
            clip_neg (bool, optional): [description]. Defaults to True.
            cache (bool, optional): Load/Save to cache. Defaults to True.
            roi (bool, optional): Set all values outside of roi to NaN. Defaults to True.
            override (bool, optional): Recompute the difference map. Defaults to False
        Returns:
            np.ndarray: [description]
        """
        kwargs['orientation'] = kwargs.get('orientation', True)
        kwargs['clip_neg'] = kwargs.get('clip_neg', True)
        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['roi'] = kwargs.get('roi', True)
        kwargs['normalize'] = kwargs.get('normalize', True)
        kwargs['override'] = kwargs.get('override', False)

        cache_field = 'or_difference_maps' if kwargs['orientation'] else 'dr_difference_maps'
        if kwargs['cache'] and not kwargs['override'] and cache_field in self.cache:
            return self.cache[cache_field]

        avg_stim_resp = self.stimulus_avg_responses(**kwargs)
        if kwargs['clip_neg']:
            avg_stim_resp[avg_stim_resp < 0] = 0

        if self.expt.do_blank():
            avg_stim_resp = avg_stim_resp[0:-1, :, :]
        num_stims = avg_stim_resp.shape[0]

        if kwargs['orientation']:
            avg_stim_resp = (avg_stim_resp[:num_stims//2, :, :] + avg_stim_resp[num_stims//2:, :, :])/2
            num_stims = num_stims//2

        roi = self.roi() if kwargs['roi'] else np.ones((avg_stim_resp.shape[1:]), dtype=bool)

        if kwargs['normalize']:
            for resp, resp_orth in zip(avg_stim_resp[:num_stims//2, :, :], avg_stim_resp[num_stims//2:, :, :]):
                max_val = np.max([np.nanmax(resp[roi]), np.nanmax(resp_orth[roi])])
                resp = resp / max_val
                resp_orth = resp/max_val

        diff_maps = np.empty((num_stims, avg_stim_resp.shape[1], avg_stim_resp.shape[2]))
        diff_maps = (avg_stim_resp[0:num_stims//2, :, :] - avg_stim_resp[num_stims//2:num_stims, :, :])

        if kwargs['roi']:
            roi = self.roi()
            diff_maps[:, ~roi] = np.nan
        if kwargs['cache']:
            self.cache[cache_field] = diff_maps

        return diff_maps

    def angles(self, orientation=True):
        num_stims = self.expt.metadata.num_stims()
        if self.expt.do_blank:
            num_stims = num_stims - 1

        if orientation:
            return np.arange(0, num_stims//2) * np.pi / (num_stims//2)

        return np.arange(0, num_stims) * 2*np.pi / (num_stims)

    def calculate_wavelength(self, **kwargs):
        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['override'] = kwargs.get('override', False)
        orientation = kwargs.pop('orientation', True)
        kwargs['diff_type'] = kwargs.get('diff_type', 'vsum')
        if kwargs['override'] and kwargs['cache']:
            self.cache.pop('wavelet_analysis', None)

        if kwargs['cache'] and 'wavelet_analysis' in self.cache:
            return self.cache['wavelet_analysis']
        roi = kwargs.pop('roi', self.roi())

        resolution = self.resolution()[0]
        if kwargs['diff_type'] == 'vsum':
            v_sum, _, _, _ = self.vector_sum(orientation=orientation, roi=False)
            diff_map = np.empty((2, v_sum.shape[0], v_sum.shape[1]))
            diff_map[0, :, :] = np.real(v_sum)
            diff_map[1, :, :] = np.imag(v_sum)
        else:
            raise NotImplementedError('Only Vector Sumemd Responses are Supported!')

        local_w = np.empty(diff_map.shape)
        local_b = np.empty(diff_map.shape)
        settings = {}
        for i in range(local_w.shape[0]):
            local_w[i, :], local_b[i, :], settings = fit_wavelet(diff_map[i, :, :],
                                                                 resolution,
                                                                 roi=roi,
                                                                 **kwargs)
            logging.info('[%s] Wavelength Difference condition %i: %.2f',
                         self.expt.animal_id, i,   np.nanmedian(local_w[i, :]))

        result = {'wavelength': local_w,
                  'bandedness': local_b,
                  'settings': settings}

        if kwargs['cache']:
            self.cache['wavelet_analysis'] = result

        return result
    # Plotting

    def angle_map_to_hsv(
            self, angle_map: np.ndarray = None, responsivity_map: np.ndarray = None, selectivity_map: np.ndarray = None, **kwargs) -> np.ndarray:
        """Convert RGB image to HSV based on selectivity and responsivity
        TODO:
            * matplotlib v3.2 has better support for invalid values
            * Use masked arrays more consistently
        Args:
            angle_map (np.ndarray): [description]
            responsivity_map (np.ndarray, optional): [description]. Defaults to None.
            selectivity_map (np.ndarray, optional): [description]. Defaults to None.
            clip_pct (float, optional): [description]. Defaults to 99.
            roi (np.ndarray, optional): [description]. Defaults to None.

        Returns:
            (np.ndarray): RGB image base on hsv LUT
        """
        kwargs['clip_pct'] = kwargs.get('clip_pct', 99)
        kwargs['clip_val'] = kwargs.get('clip_val', None)
        kwargs['roi'] = kwargs.get('roi', None)
        kwargs['orientation'] = kwargs.get('orientation', True)
        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['background_color'] = kwargs.get('background_color', True)
        prefix = 'or_' if kwargs['orientation'] else 'dr_'
        if angle_map is None:
            if kwargs['cache'] and prefix+'angle_map' in self.cache.keys():
                logging.info('[%s] Loading angle map from cache', self.expt.animal_id)
                angle_map = self.cache[prefix+'angle_map']
            else:
                raise BaseException('Please Generate an angle map first!')

        norm_img = matplotlib.colors.Normalize(
            vmin=0, vmax=2*np.pi, clip=True)(angle_map)
        norm_img = np.ma.masked_invalid(norm_img)
        if kwargs['roi'] is None:
            kwargs['roi'] = np.ones(angle_map.shape, dtype=bool)
        norm_img.mask = norm_img.mask | np.logical_not(kwargs['roi'])

        color_map = plt.cm.hsv
        color_map.set_bad('k')
        color_img = color_map(norm_img)

        hsv_img = rgb2hsv(color_img[:, :, 0:3])
        if selectivity_map is not None:
            if not np.isnan(kwargs['clip_pct']):
                hsv_img[:, :, 1] = matplotlib.colors.Normalize(vmin=0, vmax=np.nanpercentile(
                    selectivity_map[kwargs['roi']], kwargs['clip_pct']), clip=True)(selectivity_map)
            elif kwargs['clip_val']:
                hsv_img[:, :, 1] = matplotlib.colors.Normalize(
                    vmin=0, vmax=kwargs['clip_val'], clip=True)(selectivity_map)
            else:
                hsv_img[:, :, 1] = selectivity_map
        else:
            hsv_img[:, :, 1] = 1
        if responsivity_map is not None:
            hsv_img[:, :, 2] = matplotlib.colors.Normalize(vmin=0, vmax=np.nanpercentile(
                responsivity_map[kwargs['roi']], kwargs['clip_pct']), clip=True)(responsivity_map)
        else:
            hsv_img[:, :, 2] = 1

        hsv_img[np.isinf(hsv_img)] = 0
        hsv_img[np.isnan(hsv_img)] = 0

        rgb_img = hsv2rgb(hsv_img)
        rgb_img[~kwargs['roi'], 0:3] = 1 if kwargs['background_color'] == 'white' else kwargs['background_color']

        return rgb_img

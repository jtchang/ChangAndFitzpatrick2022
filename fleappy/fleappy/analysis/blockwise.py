import logging
from re import T
import numpy as np
import numba as nb
from fleappy.analysis.twophoton import TwoPhotonAnalysis
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import sem, percentileofscore
from skimage.io import imread
from itertools import combinations


@nb.njit(fastmath=True)
def norm(l):
    s = 0.
    for i in range(l.shape[0]):
        s += l[i]**2
    return np.sqrt(s)


def _template_decoder(single_trials, templates, cell_num, permutations):
    """Template Decoding Algorithm.

    Args:
        single_trials ([type]): [description]
        templates ([type]): [description]
        permutations ([type]): [description]

    Returns:
        [type]: [description]
    """

    total_cells = single_trials.shape[-1]
    similarity_vector = np.empty((single_trials.shape[0], single_trials.shape[1], templates.shape[0], permutations))
    stims = np.transpose(np.tile(np.arange(single_trials.shape[0]),
                                 (single_trials.shape[1], permutations, 1)), (2, 0, 1))

    for permutation in range(permutations):
        cell_vector = np.random.permutation(total_cells)[:cell_num]
        temp_norms = np.empty(templates.shape[0])
        for idx, template in enumerate(templates[:, cell_vector]):
            temp_norms[idx] = norm(template)
        for stim_idx, stim_resps in enumerate(single_trials[:, :, cell_vector]):
            for trial_idx, trial_resp in enumerate(stim_resps):
                trial_norm = norm(trial_resp)
                for temp_idx, template in enumerate(templates[:, cell_vector]):
                    similarity_vector[stim_idx, trial_idx, temp_idx, permutation] = np.sum(template*trial_resp) / \
                        (trial_norm * temp_norms[temp_idx])

    similarity_vector = np.argmax(similarity_vector, axis=2)

    return np.mean(similarity_vector == stims)


class BlockwiseAnalysis(TwoPhotonAnalysis):
    """Class for the handling of blockwise stimuli.

    Class for the analysis of stimuli which can be subdivided into stimulus x trials. This is a subclass of the
    BaseAnalysis class.

    Attributes:
        stim_period (tuple): Stimulus period (start, stop).
        prepad (float): Period before stimulus onset to analyze.
        postpad (float): Period after stimulus offset to analyze.
        analysis_period (tuple): Period during stimulus to analyze (start, stop).
    """

    __slots__ = ['stim_period', 'prepad', 'postpad', 'analysis_period']

    def __init__(self, expt, field, **kwargs):
        super().__init__(expt, field, **kwargs)

        self.analysis_period = kwargs['analysis_period'] if 'analysis_period' in kwargs else (0, -1)
        if self.analysis_period[1] == -1:
            self.analysis_period = (self.analysis_period[0], float(expt.stim_duration()))
        self.stim_period = (0, expt.stim_duration())
        self.prepad = kwargs['prepad'] if 'prepad' in kwargs else 0
        self.postpad = kwargs['postpad'] if 'postapad' in kwargs else 0

    def run(self, **kwargs):
        super().run(**kwargs)
        self.variability(**kwargs)
        # self.template_matching_decoder(**kwargs)
        for corrtype in ['total', 'signal', 'noise']:
            self.correlation_matrix(corrtype=corrtype, **kwargs)

        self.trial_to_trial_matrix(**kwargs)
        self.trial_to_trial_correlation_metric(**kwargs)

    def single_trial_timecourse(self, clip=True) -> np.ndarray:
        """Returns the individual trial time course for the analysis.

        Args:
            clip (bool, optional): Clip Negative Values. Defaults to True.

        Returns:
            np.ndarray: reponses (cell x stim x trial x time)
        """
        frame_rate = self.expt.frame_rate()
        prepad_frames = int(np.round(self.prepad*frame_rate))

        responses, stim_masks = self.expt.get_all_trial_responses(self.field, prepad=self.prepad, postpad=self.postpad)

        if prepad_frames > 0:
            prepad = np.mean(responses[:, :, :, stim_masks[:, 0]], axis=3)
            responses = responses - np.tile(np.expand_dims(prepad, axis=3), (1, 1, 1, responses.shape[3]))
        if clip:
            responses[responses < 0] = 0

        return responses

    def single_trial_responses(self, clip=True) -> np.ndarray:
        """Gets all single trial responses over analysis window.

        Returns:
            nd.array : Trial Responses (roi x # stims x # trials)
        """
        frame_rate = self.expt.frame_rate()
        prepad_frames = int(np.round(self.prepad*frame_rate))
        analysis_start = prepad_frames + int(np.round(self.analysis_period[0]*frame_rate))
        analysis_stop = prepad_frames + int(np.round(self.analysis_period[1]*frame_rate))
        responses = self.single_trial_timecourse(clip=clip)
        responses = np.mean(responses[:, :, :, analysis_start:analysis_stop], axis=3)

        return responses

    def responsive(self, thresholds: np.ndarray = None, **kwargs) -> np.ndarray:
        """Returns all responsive cells as boolean array.

        Thresholds cells to test for responsiveness. Default for thresholds is 2*STD + Mean of the blank condition.
        If there is no blank condition, then all cells are treated as responsive.

        Args:
            thresholds (np.array, optional): [description]. Defaults to None.

        Returns:
            np.array: Boolean array of cells that are responsive.
        """
        responses = self.single_trial_responses()
        responses[np.isnan(responses)] = 0
        if self.expt.do_blank():
            if thresholds is None:
                thresholds = self.threshold(responses=responses)
            with np.errstate(invalid='ignore'):
                if kwargs.get('resp_type', 'max') == 'median':
                    responsive_cells = np.any(np.median(responses[:, :-1, :], axis=2) >
                                              np.tile(thresholds, [1, responses.shape[1]-1]), axis=1)
                elif kwargs.get('resp_type', 'max') == 'mean':
                    responsive_cells = np.any(np.mean(responses[:, :-1, :], axis=2) >
                                              np.tile(thresholds, [1, responses.shape[1]-1]), axis=1)
                else:
                    responsive_cells = np.any(np.max(responses[:, :-1, :], axis=2) >
                                              np.tile(thresholds, [1, responses.shape[1]-1]), axis=1)

            responsive_cells[np.isnan(responsive_cells)] = 0
            return responsive_cells
        UserWarning('Blank condition required for responsiveness test.')
        return np.ones(responses.shape[0]) * np.nan

    def pairwise_correlation(self, seed_cell: int, **kwargs) -> np.ndarray:
        """Compute the pairwise correlation for a given seed cell

        Args:
            seed_cell (int): Cell ID as cell #

        Returns:
            np.array: Correlation coefficients for cells relative to the seed cell.
        """
        kwargs['corrtype'] = kwargs['corrtype'] if 'corrtype' in kwargs else 'total'
        single_trials = self.single_trial_responses()
        if self.expt.do_blank():
            single_trials = single_trials[:, :-1, :]
        correlations = np.empty((single_trials.shape[0],))
        single_trials[single_trials < 0] = 0

        if kwargs['corrtype'] == 'total':
            single_trials = np.reshape(
                single_trials, (single_trials.shape[0], single_trials.shape[1]*single_trials.shape[2]))
        elif kwargs['corrtype'] == 'signal':
            single_trials = np.mean(single_trials, axis=2)
        elif kwargs['corrtype'] == 'noise':
            signal = np.expand_dims(np.mean(single_trials, axis=2), axis=2)
            single_trials = single_trials - np.tile(signal, (1, 1, single_trials.shape[2]))
            single_trials = np.reshape(
                single_trials, (single_trials.shape[0], single_trials.shape[1]*single_trials.shape[2]))

        for cell_idx, resp in enumerate(single_trials):
            correlations[cell_idx] = np.corrcoef(resp,
                                                 single_trials[seed_cell, :])[0][1]

        return correlations

    # Built in analysis functions that run by default with run

    def correlation_matrix(self, **kwargs) -> np.ndarray:
        """Compute cellular response correlation matrix


        kwargs:
            corrtype (str): Correlation Type. Must be in ['total', 'signal', 'noise']. Defaults to total.
            cache (bool): Retrieve/Save result from/to cache. Defaults to True.
            override (bool): Force recomputation of matrix.  Defaults to False.
        Returns:
            np.ndarray: [description]
        """

        kwargs['corrtype'] = kwargs.get('corrtype', 'total')
        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['override'] = kwargs.get('override', False)

        if kwargs['cache'] and not kwargs['override']:

            if kwargs['corrtype'] + 'corr_matrix' in self.cache:
                logging.info('%s: Skipping recomputation of correlation matrix %s',
                             self.expt.animal_id, kwargs['corrtype'])
                return self.cache[kwargs['corrtype']+'corr_matrix']

        single_trials = self.single_trial_responses()
        if self.expt.do_blank():
            single_trials = single_trials[:, :-1, :]

        single_trials[single_trials < 0] = 0
        if kwargs['corrtype'] == 'total':
            single_trials = np.reshape(
                single_trials, (single_trials.shape[0], single_trials.shape[1]*single_trials.shape[2]))
        elif kwargs['corrtype'] == 'signal':
            single_trials = np.mean(single_trials, axis=2)
        elif kwargs['corrtype'] == 'noise':
            signal = np.expand_dims(np.mean(single_trials, axis=2), axis=2)
            single_trials = single_trials - np.tile(signal, (1, 1, single_trials.shape[2]))
            single_trials = np.reshape(
                single_trials, (single_trials.shape[0], single_trials.shape[1]*single_trials.shape[2]))

        correlations = np.empty((single_trials.shape[0], single_trials.shape[0]))
        for cell_a_idx, resp_a in enumerate(single_trials):
            for cell_b_idx, resp_b in enumerate(single_trials):
                correlations[cell_a_idx, cell_b_idx] = np.corrcoef(resp_a, resp_b)[0][1]

        if kwargs['cache']:
            self.cache[kwargs['corrtype']+'corr_matrix'] = correlations

        return correlations

    def bootstrap_correlation_clustering(self, bin_edges=np.arange(0, 500, 100), iterations=1000, **kwargs):

        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['corrtype'] = kwargs.get('corrtype', 'total')
        kwargs['override'] = kwargs.get('override', False)
        kwargs['two_tail'] = kwargs.get('two_tail', True)

        if kwargs['cache'] and not kwargs['override']:
            cache_key = 'bin_bootstraps_'+kwargs['corrtype']
            if cache_key in self.cache and str(bin_edges) in self.cache[cache_key]:
                bin_obs, bin_p_vals = self.cache[cache_key][str(bin_edges)]
                return bin_edges, bin_obs, bin_p_vals

        corr_matrix = self.correlation_matrix(**kwargs)

        corr_matrix[np.identity(corr_matrix.shape[0], dtype=bool)] = np.nan
        resamples = np.full((bin_edges.size, iterations), np.nan)
        for idx in range(iterations):
            dist_matrix = self.expt.distance_matrix(shuffle=True)
            for bin_idx, (min_dist, max_dist) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                resamples[bin_idx, idx] = np.nanmean(corr_matrix[np.logical_and(dist_matrix >= min_dist,
                                                                                dist_matrix < max_dist)])

        bin_obs = np.full((bin_edges.size, ), np.nan)
        bin_p_vals = np.full((bin_edges.size), np.nan)
        dist_matrix = self.expt.distance_matrix()
        for bin_idx, (min_dist, max_dist) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            bin_obs[bin_idx] = np.nanmean(corr_matrix[np.logical_and(dist_matrix >= min_dist,
                                                                     dist_matrix < max_dist)])

            p = percentileofscore(resamples[bin_idx, :], bin_obs[bin_idx]) / 100
            if kwargs['two_tail']:
                if p > 0.5:
                    p = 1-p
                p = 2*p
            bin_p_vals[bin_idx] = p

        if kwargs['cache']:
            cache_key = 'bin_bootstraps_'+kwargs['corrtype']
            if cache_key not in self.cache:
                self.cache[cache_key] = {}

            self.cache[cache_key][str(bin_edges)] = (bin_obs, bin_p_vals)

        return bin_edges, bin_obs, bin_p_vals

    def variability(self, bootstraps: int = 1000, **kwargs) -> None:
        """Compute cellular response variability.

        Computes the average correlation of the response tuning for each cell based on bootstrap resampling.

        kwargs:
            ovverride (bool): Recompute the variability. Defaults to False

        Args:
            bootstraps (int, optional): Total number of bootstraps to run. Defaults to 1000.
        """

        kwargs['override'] = kwargs.get('override',  False)

        if 'variability' not in self.metrics or kwargs['override']:
            responses = self.single_trial_responses()
            if self.expt.do_blank():
                responses = responses[:, :-1, :]
            variability = np.empty((responses.shape[0], bootstraps))
            for r_id, resp in enumerate(responses):

                for bs_id in range(bootstraps):
                    bs_resp_a = np.empty((resp.shape[0], ))
                    bs_resp_b = np.empty((resp.shape[0], ))

                    for stim_id in range(resp.shape[0]):
                        bs_resp_a[stim_id] = resp[stim_id, np.random.randint(0, resp.shape[1])]
                        bs_resp_b[stim_id] = resp[stim_id, np.random.randint(0, resp.shape[1])]

                    variability[r_id, bs_id] = 1 - np.corrcoef(bs_resp_a, bs_resp_b)[0][1]
            self.metrics['variability'] = np.mean(variability, axis=1)
        else:
            logging.info('%s Skipping Variability Recomputation', self.expt.animal_id)

    def template_matching_decoder(self, **kwargs):
        """Compute the accuracy of template matching decoder


        kwargs:
            num_cells (int): number of cells to use
            cache (bool): Save/Retrive result from cache. Defaults to True.
            override (bool): Force recomputation. Defaults to False
            permutations (int): Total number of permutations to run. Defaults to 100.
            orientation (bool): Run decoding in orientation space. Defaults to False.
            return_guesses (bool): Return the matrix of guesses. Defaults to False
        Returns:
            template decoding: Accuracy ofdecoding. If flag for return guesses is true, then all guesses will be returned.
        """

        kwargs['num_cells'] = kwargs.get('num_cells', np.arange(1, 101, 1))
        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['override'] = kwargs.get('override', False)
        kwargs['permutations'] = kwargs.get('permutations', 500)
        kwargs['orientation'] = kwargs.get('orientation', False)
        kwargs['return_guesses'] = kwargs.get('return_guess', False)

        if kwargs['cache'] and not kwargs['override'] and 'template_decoding' in self.cache:
            return self.cache['template_decoding']
        else:
            single_trials = self.single_trial_responses()
            if self.expt.do_blank():
                single_trials = single_trials[:, :-1, :]
            single_trials = single_trials[np.logical_not(
                np.any(np.logical_not(np.isfinite(single_trials)), axis=(1, 2))), :, :]

            if kwargs['orientation']:
                single_trials = np.concatenate((single_trials[:, :single_trials.shape[1]//2, :],
                                                single_trials[:, single_trials.shape[1]//2:, :, ]), axis=2)
                single_trials = single_trials[:, np.random.permutation(
                    np.arange(single_trials.shape[1])), :]  # randomize orientations
            single_trials = np.transpose(single_trials, (1, 2, 0))
            test_trial_num = int(np.ceil(single_trials.shape[1]/2))

            test_set = single_trials[:, test_trial_num:, :]
            templates = np.mean(single_trials[:, :test_trial_num, :], axis=1)
            decoding_efficiency = np.full((len(kwargs['num_cells']), 2), np.nan)

            for cell_step, cell_num in enumerate(kwargs['num_cells']):
                if cell_num > self.expt.num_roi():
                    decoding_efficiency[cell_step, :] = [cell_num, np.nan]
                else:
                    decoding_efficiency[cell_step, :] = [cell_num,
                                                         _template_decoder(test_set, templates, cell_num, kwargs['permutations'])]
            if kwargs['cache']:
                self.cache['template_decoding'] = decoding_efficiency
            return self.cache['template_decoding']

    def template_decoding_shuffle(self, **kwargs):
        kwargs['num_cells'] = kwargs.get('num_cells', np.arange(1, 101, 1))
        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['override'] = kwargs.get('override', False)
        kwargs['permutations'] = kwargs.get('permutations', 500)
        kwargs['shuffles'] = kwargs.get('shuffles', 100)
        kwargs['orientation'] = kwargs.get('orientation', False)

        if kwargs['cache'] and not kwargs['override'] and 'template_decoding_shuffle' in self.cache:
            return self.cache['template_decoding_shuffle']
        else:
            single_trials = self.single_trial_responses()
            if self.expt.do_blank():
                single_trials = single_trials[:, :-1, :]
            single_trials = single_trials[np.logical_not(
                np.any(np.logical_not(np.isfinite(single_trials)), axis=(1, 2))), :, :]

            if kwargs['orientation']:
                single_trials = np.concatenate((single_trials[:, :single_trials.shape[1]//2, :],
                                                single_trials[:, single_trials.shape[1]//2:, :, ]), axis=2)
                single_trials = single_trials[:, np.random.permutation(
                    np.arange(single_trials.shape[1])), :]  # randomize orientations

            single_trials = np.transpose(single_trials, (1, 2, 0))
            test_trial_num = int(np.ceil(single_trials.shape[1]/2))
            all_trials = np.reshape(
                single_trials, (single_trials.shape[0]*single_trials.shape[1], single_trials.shape[2]))

            shuffle_results = np.full((kwargs['num_cells'].size, kwargs['shuffles']), np.nan)
            for shuffle in range(kwargs['shuffles']):
                shuffle_trials = np.reshape(all_trials[np.random.permutation(
                    np.arange(all_trials.shape[0])), :], single_trials.shape)
                shuffle_templates = np.mean(shuffle_trials[:, test_trial_num:, :], axis=1)
                for cell_step, cell_num in enumerate(kwargs['num_cells']):
                    if cell_num > self.expt.num_roi():
                        break
                    else:
                        shuffle_results[cell_step, shuffle] = _template_decoder(
                            shuffle_trials[:, :test_trial_num, :], shuffle_templates, cell_num, kwargs['permutations'])

            if kwargs['cache']:
                self.cache['template_decoding_shuffle'] = (kwargs['num_cells'], np.nanmean(shuffle_results, axis=1))
            return (kwargs['num_cells'], np.mean(shuffle_results, axis=1))

    def trial_to_trial_matrix(self, **kwargs) -> np.ndarray:
        """Compute the trial to trial pattern correlation

        Compute all the pairwise pattern correlations of trials for the experiment.

        kwargs:
            orienation[bool]: Collapse orientation space. Defaults to False
            cache[bool]: Store/retrieve from cache. Defaults to True
            override[bool]: Recompute the matrix without retrieving from cache. Defaults to False.
        Returns:
            np.ndarray: Pattern correlation matrix (stims*trials x stims*trials)
        """
        kwargs['orientation'] = kwargs.get('orientation', False)
        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['override'] = kwargs.get('override', False)

        if not kwargs['override'] and kwargs['cache'] and 'trial_to_trial_corr_matrix' in self.cache:
            logging.info('%s Skipping Recomputation of trial-to-trial correlation matrix', self.expt.animal_id)
            return self.cache['trial_to_trial_corr_matrix']

        single_trials = self.single_trial_responses()  # cells x stims x trials

        if self.expt.do_blank():
            single_trials = single_trials[:, :-1, :]
        _, num_stims, num_trials = single_trials.shape

        if kwargs['orientation']:
            single_trials = np.concatenate(
                (single_trials[:, :num_stims//2, :], single_trials[:, num_stims//2:, :]), axis=2)
            _, num_stims, num_trials = single_trials.shape

        # drop invalid cells (cells with NaN responses)
        invalid_cells = np.any(np.isnan(single_trials), axis=(1, 2))
        single_trials = single_trials[~invalid_cells, :, :]
        single_trials[~np.isfinite(single_trials)] = 0
        single_trials = np.reshape(single_trials, (single_trials.shape[0], num_stims * num_trials))

        corr_matrix = np.ones((single_trials.shape[1], single_trials.shape[1]))

        for idx_a in range(single_trials.shape[1]):
            for idx_b in range(idx_a+1, single_trials.shape[1]):
                corr_matrix[idx_a, idx_b] = np.corrcoef(single_trials[:, idx_a], single_trials[:, idx_b])[0][1]
                corr_matrix[idx_b, idx_a] = corr_matrix[idx_a, idx_b]

        if kwargs['cache']:
            self.cache['trial_to_trial_corr_matrix'] = corr_matrix
        return corr_matrix

    def trial_to_trial_correlation_metric(self, **kwargs):
        """Compute trial to trial correlations.

        Computes the average trial correlation for trials within stimulus conditions

        Returns:
            [type]: [description]
        """
        kwargs['orientation'] = kwargs.get('orientation', False)
        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['override'] = kwargs.get('override', False)
        kwargs['keep_direction'] = kwargs.get('keep_direction', True)
        if not kwargs['override'] and kwargs['cache'] and 'trial_variability_metric' in self.cache:
            if isinstance(self.cache['trial_variability_metric'], tuple):
                return self.cache['trial_variability_metric']

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
            self.cache['trial_variability_metric'] = (matched_metric, orth_metric)
        return matched_metric, orth_metric

    def threshold(self, responses=None):

        if responses is None:
            responses = self.single_trial_responses()

        spon_thresholds = np.expand_dims(
            2 * np.std(responses[:, -1, :], axis=1) + np.mean(responses[:, -1, :], axis=1), axis=1)

        return spon_thresholds

    # Plotting Functions

    def scatter_correlation(self, ax=None, **kwargs) -> None:
        kwargs['seed_cell'] = kwargs.get('seed_cell', 0)
        kwargs['cmap'] = kwargs.get('cmap', 'bwr')
        kwargs['scale_bar'] = kwargs.get('scale_bar', 100)
        kwargs['corrtype'] = kwargs.get('corrtype', 'total')
        kwargs['background'] = kwargs.get('background', '#BBBDC0')

        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        cmap = ScalarMappable(Normalize(-1, 1), cmap=kwargs['cmap'])

        scaling_factor = self.expt.scaling_factor()
        centroids = np.array(self.map_to_roi(lambda x: x.centroid())) * scaling_factor

        correlations = self.pairwise_correlation(kwargs['seed_cell'], corrtype=kwargs['corrtype'])
        correlation_colors = cmap.to_rgba(correlations)
        correlation_colors[kwargs['seed_cell'], :] = [0, 1, 0, 1]

        seed_centroid = centroids[kwargs['seed_cell'], :]

        centroids = np.delete(centroids, kwargs['seed_cell'], axis=0)
        correlation_colors = np.delete(correlation_colors, kwargs['seed_cell'], axis=0)

        ax.scatter(seed_centroid[0],
                   seed_centroid[1],
                   c='#00ff00',
                   s=25,
                   zorder=2, marker='^')

        ax.scatter(centroids[:, 0],
                   centroids[:, 1],
                   c=correlation_colors,
                   s=25,
                   zorder=1
                   )

        self.format_scatter_plot(ax, **kwargs)

    def plot_trial_to_trial_correlation_matrix(self, ax=None, **kwargs) -> None:
        kwargs['cmap'] = kwargs.get('cmap', 'bwr')
        kwargs['showtrials'] = kwargs.get('showtrials', True)
        trial_matrix = self.trial_to_trial_matrix(**kwargs)

        c = ax.imshow(trial_matrix,
                      vmin=-1, vmax=1, cmap=kwargs['cmap'])
        if kwargs.get('colorbar', True):
            cbar = plt.colorbar(c, ax=ax, ticks=[-1, 0, 1])
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

        # Static Functions

    # TIf File Functions

    def get_trial_image(self, **kwargs):

        return self.expt.get_trial_image(**kwargs)

       
    @staticmethod
    def cohens_d(group_a: np.array, group_b: np.array) -> float:
        """Computes Cohen's d

        Args:
            group_a (np.array): Samples in Group A
            group_b (np.array): Samples in Group B

        Returns:
            float: Cohen's d
        """
        n1, n2 = group_a.size, group_b.size
        mean_difference = np.mean(group_a)-np.mean(group_b)

        # pooled SD calculator
        numerator_a = (n1-1) * np.std(group_a)**2
        numerator_b = (n2-1) * np.std(group_b)**2
        denominator = n1+n2-2
        sd_pooled = np.sqrt((numerator_a + numerator_b) / denominator)

        return mean_difference / sd_pooled

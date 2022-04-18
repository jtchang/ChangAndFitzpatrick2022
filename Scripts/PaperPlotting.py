from __future__ import annotations


import numpy as np
import shelve
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import scipy.stats
from collections import defaultdict

from fleappy.experiment.binocularexperiment import BinocularExperiment
from fleappy.analysis.blockwise import BlockwiseAnalysis
from itertools import combinations
from StatisticTesting import *
all_eo_groups = ['Naive', 'Brief', 'Extended', 'BD', 'Recovery']

eo_colors = {'Brief': '#CC2B6B',
             'Extended':  '#FC9D03',
             'Naive': '#174572',
             'BD': '#231F20',
             'Recovery': '#43B549',
             'Long-BD': 'r',
             'Immature': 'r'}
eye_color = {'ipsi': ('#6ABD45', '#6ABD45'),
             'contra': ('#9761A8', '#9761A8'),
             'binoc': ('#f28424', '#f28424')}
skip_expts = []

analysis_field = 'Full'
two_photon_db = r'../db/TwoPhoton.db'


def assign_eo_group(expt, eye='binoc'):
    if isinstance(expt, BinocularExperiment):
        expt = getattr(expt, eye)

    if expt.get_expt_parameter('bd') == 0:
        if expt.get_expt_parameter('eo') < 1 and expt.get_expt_parameter('age') > 26:
            return 'Naive'
        elif expt.get_expt_parameter('eo') < 1 and expt.get_expt_parameter('age') < 27:
            return 'Immature'
        elif expt.get_expt_parameter('eo') > 7:
            return 'Extended'
        else:
            return 'Brief'
    elif expt.get_expt_parameter('eo') <= 0:
        if expt.get_expt_parameter('bd') < 10:
            return 'BD'
        else:
            return 'Long-BD'
    else:
        return 'Recovery'


def make_eo_groups():
    expt_groups = defaultdict(lambda: [])
    with shelve.open(two_photon_db) as db:
        for aid, expt in db.items():
            if expt is None or aid in skip_expts:
                continue
            eo = assign_eo_group(expt) + '-' + expt.get_expt_parameter('virus')
            expt_groups[eo].append(aid)
    return expt_groups


# make EO groups
expt_groups = make_eo_groups()


def make_rgb_transparent(rgb, bg_rgb, alpha):
    return [alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb, bg_rgb)]


def plot_fluorescence_responses(expt, cell_num, ax, **kwargs):
    alpha = kwargs.get('alpha', 0.25)
    prepad = kwargs.get('prepad', 1)
    postpad = kwargs.get('postpad', 1)
    avgtype = kwargs.get('avgtype', 'mean')

    color = eye_color[eye] if 'eye' in 'kwargs' else ('k', [.8, .8, .8])
    eye = kwargs.get('eye', 'binoc')

    kwargs['single_trials'] = kwargs.get('single_trials', True)

    responses, trial_masks = getattr(expt, eye).get_trial_responses(cell_num,
                                                                    'dff',
                                                                    prepad=prepad,
                                                                    postpad=postpad)
    filter_window = int(np.round(.5 * getattr(expt, eye).frame_rate()))

    responses = medfilt(responses, kernel_size=[1, 1, filter_window])
    stim = np.argwhere(trial_masks[:, 1])
    t_length = responses.shape[-1]
    frame_rate = getattr(expt, eye).frame_rate()
    spacing = 10 - t_length/frame_rate

    if avgtype == 'mean':
        mean_resp = np.mean(responses, axis=1)

        sem_resp = scipy.stats.sem(responses, axis=1)

        for idx, (m_resp, e_resp) in enumerate(zip(mean_resp, sem_resp)):
            start = idx*(t_length/frame_rate + spacing)
            trial_time = np.arange(0, t_length)/frame_rate + start
            ax.fill_between(trial_time,
                            m_resp-e_resp,
                            m_resp+e_resp,
                            color=color, alpha=alpha, zorder=1)
            ax.plot(trial_time,
                    m_resp,
                    color=color, zorder=2)
            ax.plot([trial_time[stim[0]], trial_time[stim[len(stim)-1]]],
                    [-.5, -.5],
                    color='k', zorder=3)
    elif avgtype == 'median':

        single_trial_responses = getattr(expt, eye).analysis[analysis_field].single_trial_responses()[cell_num, :, :]
        med_resp = np.nanmedian(single_trial_responses, axis=1)
        median_indices = np.argmin(np.abs(single_trial_responses - np.tile(np.expand_dims(med_resp, axis=1),
                                                                           [1, single_trial_responses.shape[1]])),
                                   axis=1)

        for idx, single_trials in enumerate(responses):
            start = idx*(t_length/frame_rate + spacing)
            trial_time = np.arange(0, t_length)/frame_rate + start
            for trial_idx, resp in enumerate(single_trials):
                if trial_idx == median_indices[idx]:
                    ax.plot(trial_time,
                            resp, color=color[0], zorder=2)
                elif kwargs['single_trials']:
                    ax.plot(trial_time,
                            resp, color=color[1], zorder=1)

            ax.plot([trial_time[stim[0]], trial_time[stim[len(stim)-1]]],
                    [-.5, -.5],
                    color='k', zorder=3)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_tuning_curve(expt, cell_num, ax, **kwargs):
    colors = eye_color[kwargs['eye']] if 'eye' in kwargs else ('r', 'k')
    kwargs['eye'] = kwargs['eye'] if 'eye' in kwargs else 'binoc'
    responses = np.median(getattr(expt, kwargs['eye']).analysis[analysis_field].single_trial_responses()[
        cell_num, :-1, :], axis=1)
    angles = np.arange(responses.shape[0]) * 360 / responses.shape[0]
    fit_angles = np.arange(0, 360, .5)
    ydata = getattr(expt, kwargs['eye']).analysis[analysis_field].tuning_curve(fit_angles, cell_num, orientation=False)
    ax.scatter(angles, responses, color=colors[0])
    ax.plot(fit_angles, ydata, color=colors[1])
    ax.set_xlim((0, 360))
    ax.set_xticks(range(0, 361, 90))


def binocular_cumulative_plot(db, metric_name, ax, **kwargs):
    kwargs['eo_groups'] = kwargs.get('eo_groups', all_eo_groups)
    kwargs['nbins'] = kwargs.get('nbins', 1000)
    kwargs['value_range'] = kwargs.get('value_range', (-1, 1))
    kwargs['virus'] = kwargs.get('virus', 'dlx')
    kwargs['clip'] = kwargs.get('clip', True)
    kwargs['responsive'] = kwargs.get('responsive', True)
    kwargs['rsq'] = -1 if 'rsq' in kwargs and kwargs['rsq'] is None else kwargs.get('rsq', 0.6)
    kwargs['osi_threshold'] = kwargs.get('osi_threshold', 0.3)

    metric_summary = defaultdict(lambda: np.empty((0, kwargs['nbins'])))
    metric_means = defaultdict(lambda: np.empty((0,)))
    for eo in kwargs['eo_groups']:
        eo_group = eo + '-' + kwargs['virus']
        for aid in expt_groups[eo_group]:
            if aid in skip_expts:
                continue
            expt = db[aid]
            if expt is None or expt.contra is None or expt.ipsi is None or analysis_field not in expt.analysis:
                continue
            else:

                metric = getattr(expt.analysis[analysis_field], metric_name)

                responsive = expt.analysis[analysis_field].binocular_responsive(
                ) if kwargs['responsive'] else np.ones((expt.num_roi(),), dtype=bool)

                well_fit = expt.analysis[analysis_field].well_fit(rsq_threshold=kwargs['rsq'],
                                                                  si_threshold=kwargs['osi_threshold'],
                                                                  orientation=True)

                metric = np.abs(metric[np.logical_and(responsive, well_fit)])
                if kwargs['clip']:
                    metric = np.clip(metric, kwargs['value_range'][0], kwargs['value_range'][1])

                cum_stat, _, _, _ = scipy.stats.cumfreq(metric,
                                                        numbins=kwargs['nbins'],
                                                        defaultreallimits=kwargs['value_range'],
                                                        weights=np.ones(metric.shape[0])/metric.shape[0])

                metric_summary[eo_group] = np.concatenate((metric_summary[eo_group],
                                                           np.expand_dims(cum_stat, axis=0)), axis=0)

                metric_means[eo_group] = np.concatenate((metric_means[eo_group], [np.nanmean(metric)]), axis=0)

    bins = np.arange(kwargs['value_range'][0], kwargs['value_range'][1],
                     (kwargs['value_range'][1]-kwargs['value_range'][0])/kwargs['nbins'])
    for eo_group in kwargs['eo_groups']:
        eo = eo_group + '-' + kwargs['virus']

        err = scipy.stats.sem(metric_summary[eo], axis=0)
        avg = np.mean(metric_summary[eo], axis=0)

        ax.fill_between(bins, avg-err, avg+err, zorder=1, alpha=0.25, color=eo_colors[eo_group])
        ax.plot(bins, avg, color=eo_colors[eo_group], zorder=2, label=eo_group)

    ax.set_ylim((0, 1))
    ax.set_xlim(kwargs['value_range'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(metric_name)
    ax.set_ylabel('Probability')
    log_stat_to_file(kwargs['virus'] + ' - ' + metric_name)
    bootstrap_test_dictionary(metric_means)

    return metric_summary


def cumulative_plot(db, metric_name, ax, **kwargs):

    kwargs['virus'] = kwargs.get('virus', 'dlx')
    kwargs['eye'] = kwargs.get('eye', 'binoc')
    kwargs['eo_groups'] = kwargs.get('eo_groups', all_eo_groups)
    kwargs['clip'] = kwargs.get('clip', True)
    kwargs['nbins'] = kwargs.get('nbins', 1000)
    kwargs['rsq_threshold'] = kwargs.get('resq_threshold', 0.6)
    kwargs['value_range'] = kwargs.get('value_range', (0, 1))

    metric_summary = defaultdict(lambda: np.empty((0, kwargs['nbins'])))
    metric_means = defaultdict(lambda: np.empty((0,)))

    well_fit_dict = {'rsq_threshold': kwargs['rsq_threshold'],
                     'si_threshold': None,
                     'pref_type': 'dr_fit'}

    bins = np.arange(kwargs['value_range'][0], kwargs['value_range'][1],
                     (kwargs['value_range'][1]-kwargs['value_range'][0])/kwargs['nbins'])

    for eo in kwargs['eo_groups']:
        eo_group = eo+'-'+kwargs['virus']
        for aid in expt_groups[eo_group]:
            if aid in skip_expts:
                continue
            expt = db[aid]
            if getattr(expt, kwargs['eye']) is None:
                print(f'Skipping {aid}')
                continue

            well_fit_cells = getattr(expt, kwargs['eye']).analysis[analysis_field].well_fit(**well_fit_dict)
            responsive_cells = getattr(expt, kwargs['eye']).analysis[analysis_field].responsive()
            metric = np.stack(getattr(expt, kwargs['eye']).analysis[analysis_field].metrics[metric_name].values)
            metric[np.logical_not(well_fit_cells)] = np.nan
            metric[np.logical_not(responsive_cells)] = np.nan

            metric = np.delete(metric, np.argwhere(np.isnan(metric)))

            if kwargs['clip']:
                np.clip(metric, kwargs['value_range'][0], kwargs['value_range'][1], out=metric)

            if metric.shape[0] == 0:
                continue

            cum_stat, _, _, _ = scipy.stats.cumfreq(metric,
                                                    numbins=kwargs['nbins'],
                                                    defaultreallimits=kwargs['value_range'],
                                                    weights=np.ones(metric.shape[0])/metric.shape[0])

            metric_summary[eo_group] = np.concatenate((metric_summary[eo_group],
                                                       np.expand_dims(cum_stat, axis=0)), axis=0)
            metric_means[eo_group] = np.concatenate((metric_means[eo_group], [np.mean(metric)]))
        err = scipy.stats.sem(metric_summary[eo_group], axis=0)
        avg = np.mean(metric_summary[eo_group], axis=0)

        if eo not in kwargs.get('skip_sem', []):
            ax.fill_between(bins, avg-err, avg+err, zorder=1, alpha=0.25, color=eo_colors[eo])
        ax.plot(bins, avg, color=eo_colors[eo], zorder=2, label=eo)

    ax.set_ylim((0, 1))
    ax.set_xlim(kwargs['value_range'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(metric_name)
    ax.set_ylabel('Probability')

    log_stat_to_file(f'{kwargs["virus"]} - [{kwargs["eye"]}] - {metric_name}')
    bootstrap_test_dictionary(metric_means)


def collect_responsive_tuned_proportions(expt, **kwargs):
    kwargs['or_field'] = kwargs.get('or_field', 'OSI')
    kwargs['dr_field'] = kwargs.get('dr_field', 'DSI')
    kwargs['osi_threshold'] = kwargs.get('osi_thereshold', 0.3)
    kwargs['dsi_threshold'] = kwargs.get('dsi_threshold', 0.3)

    summary = np.zeros((1, 3))  # responsive, orientation tuned, direction tuned

    responsive_cells = expt.analysis[analysis_field].responsive()
    summary[0, 0] = np.sum(responsive_cells)/responsive_cells.shape[0]

    orientation_tuned = np.logical_and(expt.analysis[analysis_field].metrics['OSI_sig'].values,
                                       responsive_cells)
    summary[0, 1] = np.sum(orientation_tuned)/orientation_tuned.shape[0]

    direction_tuned = np.logical_and(expt.analysis[analysis_field].metrics['DSI_sig'],
                                     responsive_cells)
    summary[0, 2] = np.sum(direction_tuned)/direction_tuned.shape[0]

    return summary


def plot_percent_responsive(db, eo_list, virus, ax, **kwargs):
    kwargs['eye'] = kwargs.get('eye', 'binoc')
    kwargs['virus'] = kwargs.get('virus', 'dlx')

    eo_v_list = [eo+'-'+kwargs['virus'] for eo in eo_list]

    responsive = defaultdict(lambda: np.empty((0, 3)))
    for eo in eo_v_list:
        for aid in expt_groups[eo]:
            if aid in skip_expts:
                continue
            expt = db[aid]
            responsive[eo] = np.concatenate((responsive[eo],
                                             collect_responsive_tuned_proportions(getattr(expt, kwargs['eye']))), axis=0)

    for idx, eo in enumerate(eo_v_list):
        ax.bar(idx,
               np.mean(responsive[eo][:, 0]),
               color='white',
               edgecolor='black',
               zorder=0,
               label='Responsive')
        ax.errorbar(idx,
                    np.mean(responsive[eo][:, 0]),
                    yerr=scipy.stats.sem(responsive[eo][:, 0]),
                    color='black',
                    zorder=3)

        ax.bar(idx,
               np.mean(responsive[eo][:, 1]),
               color='white',
               edgecolor='black',
               hatch='//',
               zorder=1,
               label='Orientation Tuned')
        ax.errorbar(idx,
                    np.mean(responsive[eo][:, 1]),
                    yerr=scipy.stats.sem(responsive[eo][:, 1]),
                    color='black',
                    zorder=4)

        ax.bar(idx,
               np.mean(responsive[eo][:, 2]),
               color='gray',
               edgecolor='black',
               zorder=2,
               label='Direction Tuned')
        ax.errorbar(idx,
                    np.mean(responsive[eo][:, 2]),
                    yerr=scipy.stats.sem(responsive[eo][:, 2]),
                    color='black',
                    zorder=5)
    log_stat_to_file(f'Fraction of Responsive Cells ({kwargs["eye"]})')
    bootstrap_responsive_bar(responsive)

    ax.set_ylim((0, 1))
    ax.set_ylabel('Fraction of Cells')
    ax.set_xticks(range(len(eo_list)))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticklabels([x[0] for x in eo_list])

    return responsive


def collect_positions_and_preferences(expt, **kwargs):

    kwargs['osi_thershold'] = kwargs.get('osi_threshold', 0)
    kwargs['rsq_threshold'] = kwargs.get('rsq_threshold', 0)
    kwargs['bin_range'] = kwargs.get('bin_range', (0, 500))
    kwargs['nbins'] = kwargs.get('nbins', 10)

    untuned = np.argwhere(np.logical_or(expt.analysis[analysis_field].metrics.dr_rsq <= kwargs['rsq_threshold'],
                                        expt.analysis[analysis_field].metrics.OSI <= kwargs['osi_threshold']))

    dist_matrix = expt.distance_matrix()
    or_matrix = expt.analysis[analysis_field].pairwise_orientation_matrix('dr_fit', orientation=True)
    dist_matrix = np.delete(np.delete(dist_matrix, untuned, axis=0), untuned, axis=1)
    or_matrix = np.delete(np.delete(or_matrix, untuned, axis=0), untuned, axis=1)

    np.fill_diagonal(or_matrix, np.nan)
    np.fill_diagonal(dist_matrix, np.nan)
    lower_triangle = np.arange(or_matrix.shape[0])[:, None] > np.arange(or_matrix.shape[1])
    or_matrix[lower_triangle] = np.nan
    dist_matrix[lower_triangle] = np.nan

    or_matrix = np.reshape(or_matrix, (or_matrix.size))
    dist_matrix = np.reshape(dist_matrix, (dist_matrix.size))

    valid = np.logical_not(np.logical_or(np.isnan(dist_matrix), np.isnan(or_matrix)))

    stat, edges, binnumber = scipy.stats.binned_statistic(
        dist_matrix[valid], or_matrix[valid], bins=kwargs['nbins'], range=kwargs['bin_range'])

    return stat, edges, binnumber


def collect_positions_and_correlations(expt, **kwargs):

    kwargs['bin_range'] = kwargs.get('bin_range', (0, 500))
    kwargs['nbins'] = kwargs.get('nbins', 10)
    kwargs['corrtype'] = kwargs.get('corrtype', 'total').lower()

    dist_matrix = expt.distance_matrix()
    corr_matrix = expt.analysis[analysis_field].correlation_matrix(corrtype=kwargs['corrtype'])

    np.fill_diagonal(corr_matrix, np.nan)
    np.fill_diagonal(dist_matrix, np.nan)
    lower_triangle = np.arange(corr_matrix.shape[0])[:, None] > np.arange(corr_matrix.shape[1])
    corr_matrix[lower_triangle] = np.nan
    dist_matrix[lower_triangle] = np.nan

    corr_matrix = np.reshape(corr_matrix, (corr_matrix.size))
    dist_matrix = np.reshape(dist_matrix, (dist_matrix.size))

    valid = np.logical_not(np.logical_or(np.isnan(dist_matrix), np.isnan(corr_matrix)))

    stat, edges, binnumber = scipy.stats.binned_statistic(
        dist_matrix[valid], corr_matrix[valid], bins=kwargs['nbins'], range=kwargs['bin_range'])

    return stat, edges, binnumber


def plot_orientation_clustering(db, eo_groups, virus, **kwargs):

    ax = kwargs.get('ax', None)
    nbins = kwargs.get('nbins', 10)
    bin_range = kwargs.get('bin_range', (0, 500))
    kwargs['eye'] = kwargs.get('eye', 'binoc')

    clustering_summary = defaultdict(lambda: np.empty((0, nbins)))

    ax.set_ylim((0, 60))
    ax.set_xlim((bin_range[0], bin_range[1]))
    ax.set_xlabel('Pairwise Difference')
    ax.set_ylabel('|Orientation Preference Difference|')

    for eo in eo_groups:
        eo_group = eo + '-' + 'virus'
        for aid in expt_groups[eo_group]:
            stat, bins, _ = collect_positions_and_preferences(getattr(db[aid], kwargs['eye']),
                                                              osi_threshold=0.3,
                                                              rsq_threshold=0.6,
                                                              nbins=nbins,
                                                              bin_range=bin_range)
            clustering_summary[eo] = np.concatenate((clustering_summary[eo], np.expand_dims(stat, axis=0)), axis=0)

    for eo in eo_groups:
        ax.errorbar(bins[1:],
                    np.mean(clustering_summary[eo], axis=0),
                    scipy.stats.sem(clustering_summary[eo], axis=0),
                    label=eo)


def bar_plot_orientation_clustering(db, eo_list, virus, **kwargs):

    ax = kwargs.get('ax', None)
    nbins = kwargs.get('nbins', 1)
    bin_range = kwargs.get('bin_range', (0, 500))

    clustering_summary = defaultdict(lambda: np.empty((0, len(bin_range))))

    for eo in eo_list:
        for aid in expt_groups[eo + '-' + virus]:
            expt = db[aid]
            if expt is None:
                continue
            stat = np.empty((1, len(bin_range)))
            for br_num, br in enumerate(bin_range):
                stat[:, br_num], _, _ = collect_positions_and_preferences(getattr(expt, kwargs['eye']),
                                                                          osi_threshold=0.3,
                                                                          rsq_threshold=0.6,
                                                                          nbins=nbins, bin_range=br)

            clustering_summary[eo] = np.concatenate((clustering_summary[eo], stat), axis=0)

    ax.set_ylim((0, 60))
    ax.set_ylabel('|Preference Difference|')

    bar_plot_summary(clustering_summary, eo_list, ax, virus=virus, singles=True)


def plot_raster(expt, ax, label):

    responsive = expt.binoc.analysis[analysis_field].responsive()
    responses = expt.binoc.analysis[analysis_field].single_trial_responses()[responsive, :, :]
    orientations = np.argsort(
        expt.binoc.analysis[analysis_field].get_orientation_preferences(orientation=False)[responsive])
    responses = responses[orientations, :, :]

    responses = np.reshape(responses, (responses.shape[0], responses.shape[1]*responses.shape[2]))

    responses = responses / np.repeat(np.expand_dims(np.max(responses, axis=1), axis=1), responses.shape[1], axis=1)

    responses[responses < 0] = 0

    c = ax.imshow(responses, vmin=0, vmax=1, cmap='magma', aspect='auto')
    ax.set_ylabel('Cells')
    ax.set_xlabel('Trials')
    ax.set_title(label)
    ax.set_xticks(np.arange(0, 161, 40))


def plot_single_trials(positions, stim_response, ax, **kwargs):

    ax.set_facecolor([0.5, 0.5, 0.5])
    c = ax.scatter(positions[:, 0],
                   positions[:, 1],
                   c=stim_response,
                   s=kwargs.get('s', 10), vmin=0, vmax=1, cmap='magma'
                   )
    ax.set_ylim((698.88, 0))
    ax.set_xlim((0, 698.88))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def _collect_correlation_clustering(db, eo_list, virus, **kwargs):
    nbins = kwargs.get('nbins', 1)
    bin_range = kwargs.get('bin_range', ((0, 500)))
    kwargs['eye'] = kwargs.get('eye', 'binoc')
    kwargs['corrtype'] = kwargs['corrtype'] if 'corrtype' in kwargs else 'total'

    clustering_summary = defaultdict(lambda: np.empty((0, len(bin_range))))

    for eo in eo_list:
        for aid in expt_groups[eo + '-' + virus]:
            if aid in skip_expts:
                continue
            expt = db[aid]
            if expt is None:
                continue
            stat = np.empty((1, len(bin_range)))
            for br_num, br in enumerate(bin_range):
                stat[:, br_num], _, _ = collect_positions_and_correlations(
                    getattr(expt, kwargs['eye']), nbins=nbins, bin_range=br, corrtype=kwargs['corrtype'])

            clustering_summary[eo] = np.concatenate((clustering_summary[eo], stat), axis=0)

    return clustering_summary


def _bar_plot_correlation_grouped(eo_list, clustering_summary, **kwargs):
    ax = kwargs.get('ax', None)

    num_eos = len(eo_list)
    bar_width = 1/num_eos
    offsets = np.linspace(-0.5 + bar_width/2, 0.5-bar_width/2, num_eos)

    for offset, eo in zip(offsets, eo_list):
        for bin_idx, bin_vals in enumerate(np.transpose(clustering_summary[eo])):
            ax.bar(bin_idx*1.5 + offset,
                   np.nanmean(bin_vals),
                   color=eo_colors[eo],
                   label=eo,
                   zorder=1)
            ax.errorbar(bin_idx*1.5 + offset,
                        np.nanmean(bin_vals),
                        sem(bin_vals, nan_policy='omit'),
                        color='k',
                        zorder=2)


def bar_plot_correlation_clustering(db, eo_list, virus, **kwargs):
    ax = kwargs.get('ax', None)
    kwargs['grouped'] = kwargs.get('grouped', True)
    kwargs['stat_test'] = kwargs.get('stat_test')

    ax.set_ylim((0, 1))
    ax.set_xlim((-1, (len(eo_list)-1)*3+1))
    ax.set_ylabel('Response Correlation')

    clustering_summary = kwargs.get('clustering_summary', _collect_correlation_clustering(db, eo_list, virus, **kwargs))

    if kwargs['grouped']:
        _bar_plot_correlation_grouped(eo_list, clustering_summary, **kwargs)
    else:
        bar_plot_summary(clustering_summary, eo_list, ax, virus=virus, ylim=kwargs.get('ylim', None))

        if kwargs['stat_test']:
            log_stat_to_file(f'Cellular Response CorrelationClustering {virus} Within Condition')
            _ = test_dictionary_clustering(clustering_summary)

            log_stat_to_file(f'Cellular Response Correlation Clustering {virus} Across Conditions')
            p_values = ttest_dictionary_clustering_difference(clustering_summary)
            x_range = list(combinations(np.arange(0, len(eo_list)*3, 3), 2))

            for idx, (p, x) in enumerate(zip(p_values, x_range)):
                annotate_significance(ax, [x[0], x[1]], 0.9 - idx*.05, p)

    clustering_within_bin_testing_diff(clustering_summary)

    return clustering_summary


def bar_plot_summary(clustering_summary, eo_list, ax, virus=None, singles=False, **kwargs):
    for e_idx, eo in enumerate(eo_list):
        x = [(e_idx*3), (e_idx*3)+1]
        yvals = []
        for x_val, values, in zip(x, np.transpose(clustering_summary[eo])):
            yvals.append(np.nanmean(values))
            ax.errorbar(x_val,
                        np.nanmean(values, axis=0),
                        scipy.stats.sem(values[np.logical_not(np.isnan(values))], axis=0),
                        label=eo,
                        elinewidth=1,
                        linewidth=0,
                        color='k', zorder=3, marker='.')
        ax.plot(x, yvals, color='k', zorder=2)
        for values in clustering_summary[eo]:
            ax.plot(x, values, color='gainsboro', zorder=1)

    if isinstance(kwargs.get('ylim', None), tuple):
        ax.set_ylim(kwargs.get('ylim'))

    ax.set_xlim([-1, (len(eo_list)-1)*3+2])
    ax.set_xticks(np.arange(0.5, len(eo_list)*3, 3))
    ax.set_xticklabels([eo[0] for eo in eo_list])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_trial_variability_metric(db, eo_list, virus, **kwargs):
    kwargs['eye'] = kwargs.get('eye', 'binoc')
    kwargs['plot_ortho'] = kwargs.get('plot_ortho', True)

    matched_offset = 0
    orth_offset = 0.5
    if kwargs['plot_ortho']:
        matched_single_color = '#EDB1B4'
        matched_color = '#ED1C24'
        matched_offset = -0.5
        orth_single_color = '#B3DEEF'
        orth_color = '#00ADEF'
    else:
        matched_single_color = 'silver'
        matched_color = 'black'

    matched_var = defaultdict(lambda: np.empty((0,)))
    orth_var = defaultdict(lambda: np.empty((0,)))

    for idx, eo in enumerate(eo_list):
        for aid in expt_groups[eo + '-' + virus]:
            if aid in skip_expts:
                continue
            matched_stat, orth_stat = getattr(
                db[aid], kwargs['eye']).analysis[analysis_field].trial_to_trial_correlation_metric(keep_direction=True)
            matched_var[eo] = np.append(matched_var[eo], [matched_stat])
            orth_var[eo] = np.append(orth_var[eo], [orth_stat])
            kwargs['ax'].scatter(idx*3+matched_offset,
                                 matched_stat,
                                 c=matched_single_color,
                                 marker='.',
                                 zorder=2)

            if kwargs['plot_ortho']:
                kwargs['ax'].scatter(idx*3+orth_offset,
                                     orth_stat,
                                     c=orth_single_color,
                                     marker='.', zorder=2)
        kwargs['ax'].errorbar(idx*3+matched_offset,
                              np.mean(matched_var[eo]),
                              scipy.stats.sem(matched_var[eo]),
                              marker='.', color=matched_color, zorder=4)

        if kwargs['plot_ortho']:
            for matched, orth in zip(matched_var[eo], orth_var[eo]):
                kwargs['ax'].plot([idx*3+matched_offset, idx*3+orth_offset],
                                  [matched, orth],
                                  color='silver',
                                  zorder=1
                                  )

            kwargs['ax'].plot([idx*3+matched_offset, idx*3+orth_offset],
                              [np.nanmean(matched_var[eo]), np.nanmean(orth_var[eo])],
                              marker='', color='k', zorder=3)

            kwargs['ax'].errorbar(idx*3+orth_offset,
                                  np.mean(orth_var[eo]),
                                  scipy.stats.sem(orth_var[eo]),
                                  marker='.', color=orth_color, zorder=4)

    kwargs['ax'].set_xticks(np.arange(0, len(eo_list)*3, 3))
    kwargs['ax'].set_xticklabels([eo[0] for eo in eo_list])
    kwargs['ax'].set_xlim((-1, (len(eo_list)-1)*3+1))
    kwargs['ax'].set_ylim((-0.2, 1))
    kwargs['ax'].axhline(0, color='k', linewidth=0.5)
    kwargs['ax'].set_yticks(np.arange(-0.2, 1.1, .2))
    kwargs['ax'].set_ylabel('Trial-Trial Correlation')
    kwargs['ax'].spines['right'].set_visible(False)
    kwargs['ax'].spines['top'].set_visible(False)

    log_stat_to_file('Two-Photon Pattern Variability')
    p_values = ttest_trial_variability_matched_ortho(eo_list,
                                                     matched_var,
                                                     orth_var)
    for idx, pval in enumerate(p_values):
        annotate_significance(kwargs['ax'], [3*idx+matched_offset, 3*idx+orth_offset], 0.9, pval)

    log_stat_to_file('Two Photon Pattern Variability - Matched')
    _ = bootstrap_test_dictionary(matched_var)

    log_stat_to_file('Two Photon Pattern Variability - Orthogonal')
    _ = bootstrap_test_dictionary(orth_var)

    return matched_var


def plot_template_decoding(db, eo_list, axis_list, **kwargs):
    kwargs['eye'] = kwargs.get('eye', 'binoc')
    kwargs['virus'] = kwargs.get('virus', 'dlx')
    max_cells = 101
    num_cells = np.arange(1, max_cells, 1)
    accuracy = {}
    shuffle_accuracy = {}
    eo_v_list = [eo + '-' + kwargs['virus'] for eo in eo_list]

    for idx, eo in enumerate(eo_v_list):
        accuracy[eo] = np.full((num_cells.size, len(expt_groups[eo])), np.nan)
        shuffle_accuracy[eo] = np.full((num_cells.size, len(expt_groups[eo])), np.nan)
        for a_idx, aid in enumerate(expt_groups[eo]):
            if aid in skip_expts:
                continue
            decoding = getattr(db[aid], kwargs['eye']).analysis[analysis_field].template_matching_decoder()
            axis_list[idx].plot(decoding[:, 0], decoding[:, 1],  color='silver', zorder=1, label=aid)

            decoding[np.isnan(decoding[:, 1]), 1] = decoding[np.max(np.where(~np.isnan(decoding[:, 1]))), 1]
            accuracy[eo][:, a_idx] = decoding[:, 1]
            _, shuffle = getattr(db[aid], kwargs['eye']).analysis[analysis_field].template_decoding_shuffle()
            shuffle[np.isnan(shuffle)] = shuffle[np.max(np.where(~np.isnan(shuffle)))]

            shuffle_accuracy[eo][:, a_idx] = shuffle

            axis_list[idx].plot(num_cells, shuffle, color='lightcoral', zorder=1)

        axis_list[idx].plot(num_cells, np.nanmean(accuracy[eo], axis=1), color='k', zorder=2)
        axis_list[idx].plot(num_cells, np.nanmean(shuffle_accuracy[eo], axis=1), color='r', zorder=2)

    for idx, ax in enumerate(axis_list):
        ax.set_xlim(0, max_cells)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Fraction Correct')
        ax.set_xlabel('Number of Cells')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(eo_list[idx])

    statistic_test_decoding_accuracy(accuracy, shuffle_accuracy)
    statistic_test_decoding_comparison(accuracy)

    return accuracy, shuffle_accuracy


def plot_decoding_derivative(db, eo_list, ax, **kwargs):
    kwargs['eye'] = kwargs.get('eye', 'binoc')
    kwargs['virus'] = kwargs.get('virus', 'dlx')

    accuracy_diffs = {}

    cells = np.arange(2, 101, 1)
    for _, eo in enumerate(eo_list):
        eo_group = eo + '-' + kwargs['virus']
        accuracy_diffs[eo] = np.full((cells.size, len(expt_groups[eo_group])), np.nan)
        for a_idx, aid in enumerate(expt_groups[eo_group]):
            if aid in skip_expts:
                continue

            accuracy_diffs[eo][:, a_idx] = np.diff(
                getattr(db[aid], kwargs['eye']).analysis[analysis_field].template_matching_decoder()[:, 1])

            # ax.plot(cells, accuracy_diffs[:, a_idx], color=eo_colors[eo], alpha=0.25, zorder=1),
        accuracy_sem = scipy.stats.sem(accuracy_diffs[eo], axis=1, nan_policy='omit')
        accuracy_mean = np.nanmean(accuracy_diffs[eo], axis=1)

        ax.fill_between(cells, accuracy_mean + accuracy_sem, accuracy_mean -
                        accuracy_sem, color=eo_colors[eo], zorder=1, alpha=0.25)
        ax.plot(cells, accuracy_mean, color=eo_colors[eo], zorder=2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Decoder Cell Number')
    ax.set_ylabel('Delta Accuracy')
    ax.set_ylim(-0.01, 0.1)
    ax.set_xlim(0, 100)

    log_stat_to_file('Decoding Accuracy Difference')

    statistic_test_decoding_diff(accuracy_diffs)

    return accuracy_diffs


def violin_plot_cohens(cohens_dict, ax, eos=['Naive-dlx', 'Brief-dlx', 'Extended-dlx']):
    log_stat_to_file('Blank Cohen''s d')
    colors = ['k', 'r']
    for idx, eo in enumerate(eos):
        for tuned_idx, tuning in enumerate(['-tuned']):
            parts = ax.violinplot(cohens_dict[eo+tuning], [idx + tuned_idx], showextrema=False)
            ax.errorbar(idx + tuned_idx,
                        np.mean(cohens_dict[eo+tuning]),
                        np.std(cohens_dict[eo+tuning]),
                        color=colors[tuned_idx],
                        marker='o',
                        zorder=3)

            t_stat, p = scipy.stats.ttest_1samp(cohens_dict[eo+tuning], 0)
            log_stat_to_file(f' * {eo+tuning}: {t_stat:0.4f} (p = {p:0.4f}, n= {len(cohens_dict[eo+tuning])})')
            if p < 0.0005:
                ax.text(idx+tuned_idx, 6, '***', ha='center')
            elif p < 0.005:
                ax.text(idx+tuned_idx, 6, '**', ha='center')
            elif p < 0.05:
                ax.text(idx+tuned_idx, 6, '*', ha='center')
            else:
                ax.text(idx+tuned_idx, 6, 'ns', ha='center')
            for pc in parts['bodies']:
                pc.set_facecolor(colors[tuned_idx])
                pc.set_alpha(.5)

    ax.set_xticks(np.arange(0, 4, 1))
    ax.set_xticklabels([eo[0] for eo in eos])
    ax.axhline(ls='--', lw=1, color='k')
    ax.set_ylim([-2, 6])
    ax.set_ylabel('Orthogonal-Blank Cohen''s d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    bootstrap_test_dictionary(cohens_dict)
    return(cohens_dict)


def compare_cohens(expt):

    responses = expt.binoc.analysis['Full'].single_trial_responses()
    pref, orth = expt.binoc.analysis['Full']._pref_ortho_responses()
    blanks = responses[:, -1, :]

    cohens_d_orth = np.empty(pref.shape[0])
    cohens_d = np.empty(pref.shape[0])
    cohens_d_pref = np.empty(pref.shape[0])
    for idx, (pref_resp, orth_resp, blank_resp) in enumerate(zip(pref, orth, blanks)):
        cohens_d_orth[idx] = BlockwiseAnalysis.cohens_d(orth_resp, blank_resp)
        cohens_d[idx] = BlockwiseAnalysis.cohens_d(pref_resp, orth_resp)
        cohens_d_pref[idx] = BlockwiseAnalysis.cohens_d(pref_resp, blank_resp)
    return cohens_d, cohens_d_orth, cohens_d_pref

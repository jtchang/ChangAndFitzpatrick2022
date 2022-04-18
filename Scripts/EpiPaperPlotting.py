from collections import defaultdict
import shelve

import numpy as np
from scipy.ndimage import shift
from scipy.stats import sem, ttest_rel
from fleappy.analysis.epimapstructure import fit_wavelet
from itertools import combinations
import PaperPlotting
from StatisticTesting import *
from matplotlib.pyplot import colorbar
all_eo_groups = PaperPlotting.all_eo_groups
epi_db = r'..\db\Epi.db'
summary_db_path = r'..\db\Epi_Summary.db'
analysis_field = 'Full'
skip_expts = []
eo_colors = PaperPlotting.eo_colors


def assign_eo_group(expt):
    return PaperPlotting.assign_eo_group(expt)


def make_eo_groups():
    expt_groups = defaultdict(lambda: [])
    with shelve.open(epi_db) as db:
        for aid, expt in db.items():
            if expt is None or aid in skip_expts:
                continue
            eo = assign_eo_group(expt) + '-' + expt.get_expt_parameter('virus')
            expt_groups[eo].append(aid)
    return expt_groups


# make EO groups
expt_groups = make_eo_groups()


def plot_difference_map(expt, axis_list, **kwargs):
    fov = kwargs.pop('fov', ((270, 0), (0, 320)))
    offset = kwargs.pop('offset', 0)
    kwargs['scalebar'] = kwargs.get('scalebar', None)  # length of scalebar in microns
    kwargs['override'] = kwargs.get('override', False)
    kwargs['zrange'] = kwargs.get('zrange', (-1, 1))

    eo = expt.get_expt_parameter('eo')
    age = expt.get_expt_parameter('age')
    axis_list[0].set_title(f'{expt.animal_id} (EO+{eo}/p{age})')
    diff_maps = expt.analysis[analysis_field].difference_maps(
        orientation=True, normalize=True, cache=True, override=kwargs['override'])
    for idx, ax in enumerate(axis_list[0:2]):
        stim = (offset + (idx * diff_maps.shape[0]//2)) % diff_maps.shape[0]
        ax.imshow(diff_maps[stim, :, :], vmin=kwargs['zrange'][0], vmax=kwargs['zrange'][1], cmap='gray')

        angle = np.rad2deg(expt.analysis[analysis_field].angles(orientation=True)[stim])
        title = f'{angle : 0.1f}°'
        ax.set_xlabel(title)

        configure_image_axis(ax, scalebar=kwargs['scalebar'], fov=fov,
                             resolution=expt.analysis[analysis_field].resolution()[0])

    _, angle_map, _, selectivity_map = expt.analysis[analysis_field].vector_sum(
        orientation=True)

    roi = expt.roi(expt.analysis[analysis_field].field)
    hsv_img = expt.analysis[analysis_field].angle_map_to_hsv(
        angle_map, selectivity_map=selectivity_map, roi=roi, clip_val=.3, clip_pct=np.nan, background_color=[.9, .9, .9])

    axis_list[2].imshow(hsv_img, aspect='equal')
    axis_list[2].set_xlabel('Orientation Preference Map')

    configure_image_axis(axis_list[2], scalebar=kwargs['scalebar'], fov=fov,
                         resolution=expt.analysis[analysis_field].resolution()[0])


def configure_image_axis(ax, scalebar=None, plot_cbar=True, fov=None, c_object=None, resolution=None):
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


def trial_to_trial_variability(db, ax, virus='dlx', eo_list=all_eo_groups, **kwargs):

    kwargs['plot_ortho'] = kwargs.get('plot_ortho', True)

    matched_single_color = 'silver'
    matched_color = 'black'

    orth_single_color = '#B3DEEF'
    orth_color = '#00ADEF'
    matched_offset = 0
    orth_offset = 0.5
    if kwargs['plot_ortho']:
        matched_single_color = '#EDB1B4'
        matched_color = '#ED1C24'
        matched_offset = -0.5

    eo_list_v = [eo+'-'+virus for eo in eo_list]
    matched_var = defaultdict(lambda: np.empty((0,)))
    orth_var = defaultdict(lambda: np.empty((0,)))
    for eo_idx, eo_group in enumerate(eo_list_v):
        for aid in expt_groups[eo_group]:
            matched_stat, orth_stat = db[aid].analysis[analysis_field].pattern_variability()
            if np.isnan(matched_stat) or np.isnan(orth_stat):
                print(aid)
            matched_var[eo_group] = np.append(matched_var[eo_group],
                                              [matched_stat])

            orth_var[eo_group] = np.append(orth_var[eo_group],
                                           [orth_stat])

    for eo_idx, eo in enumerate(eo_list_v):
        matched = matched_var[eo]
        orth = orth_var[eo]

        if kwargs['plot_ortho']:
            for matched_val, orth_val in zip(matched, orth):
                ax.plot([3*eo_idx+matched_offset, 3*eo_idx + orth_offset],
                        [matched_val, orth_val], color='silver', zorder=0)

            ax.scatter(np.ones(orth.size) * 3*eo_idx + orth_offset,
                       orth, color=orth_single_color, marker='.', zorder=1)

            ax.plot([3*eo_idx+matched_offset, 3*eo_idx + orth_offset],
                    [np.nanmean(matched), np.nanmean(orth)], color='k', zorder=2)

            ax.errorbar(3*eo_idx + orth_offset,
                        np.nanmean(orth), sem(orth), color=orth_color, marker='.', zorder=3)

        ax.scatter(np.ones(matched.size) * 3*eo_idx + matched_offset,
                   matched, color=matched_single_color, marker='.', zorder=1)

        ax.errorbar(3*eo_idx + matched_offset,
                    np.nanmean(matched), sem(matched), color=matched_color, marker='.', zorder=3)

    ax.set_ylim((-0.2, 1.1))
    ax.set_xlim((-1, 3*(len(eo_list)-1)+1))
    ax.set_xticks(3 * np.arange(len(eo_list)))
    ax.set_yticks(np.arange(0, 1.1, 0.25))
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xticklabels([eo[0] for eo in eo_list])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Trial-to-Trial Correlation')

    log_stat_to_file('Widefield Pattern Variability')

    p_values = ttest_trial_variability_matched_ortho(eo_list_v, matched_var, orth_var)

    for idx, pval in enumerate(p_values):
        annotate_significance(ax, [3*idx+matched_offset, 3*idx+orth_offset], 0.9, pval)

    log_stat_to_file('Epi Pattern Variability - Matched')
    _ = bootstrap_test_dictionary(matched_var)

    log_stat_to_file('Epi Pattern Variability - Orthogonal')
    _ = bootstrap_test_dictionary(orth_var)
    return matched_var


def plot_binned_correlations(db, ax, virus='dlx', eo_list=all_eo_groups, field='Filtered', **kwargs):
    kwargs['corr_type'] = kwargs.get('corr_type', 'total')

    eo_v_list = [eo + '-' + virus for eo in eo_list]

    summary = {}

    for eo in eo_v_list:
        summary[eo] = np.full((20, len(expt_groups[eo])), np.nan)

        for idx, aid in enumerate(expt_groups[eo]):

            _, correlations = db[aid].analysis[field].cache[kwargs['corr_type']+'_binned_data']

            if correlations.shape[0] == 20:
                summary[eo][:, idx] = correlations
            else:
                print(f'Not 20 long {aid}')

    distances = np.arange(0.1, 1.6, 0.1)
    for idx, (eo, eo_v) in enumerate(zip(eo_list, eo_v_list)):
        eo_mean = np.mean(summary[eo_v][:len(distances), :], axis=1)
        eo_sem = sem(summary[eo_v][:len(distances), :], axis=1, nan_policy='omit')
        ax.plot(distances,
                eo_mean,
                color=eo_colors[eo],
                zorder=1,
                label=eo
                )
        ax.fill_between(distances,
                        eo_mean-eo_sem,
                        eo_mean+eo_sem,
                        color=eo_colors[eo],
                        alpha=0.25,
                        linewidth=0,
                        zorder=0)

    log_stat_to_file('Epi Correlation Distance Linearity')
    correlation_distance_linearity(summary, distances)

    ax.set_ylabel('Average Pixelwise Total Correlation')
    ax.set_xlabel('Distance (mm)')
    ax.set_xlim(distances[0], distances[-1])
    ax.set_ylim(-0.2, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return summary, distances


def fraction_of_tuned_pixels(db, ax, virus='dlx', eo_list=all_eo_groups, **kwargs):

    eo_v_list = [eo + '-' + virus for eo in eo_list]

    sig_tuned_summary = {}

    for eo_idx, eo in enumerate(eo_v_list):
        sig_tuned_summary[eo] = np.empty((len(expt_groups[eo])))
        for a_idx, aid in enumerate(expt_groups[eo]):
            expt = db[aid]
            responsive = expt.analysis[analysis_field].significantly_responsive_pixels()
            tuned = expt.analysis[analysis_field].significant_tuned_pixels()
            roi = expt.analysis[analysis_field].roi()
            percentage = np.nansum(responsive[roi] & tuned[roi]) / np.nansum(responsive[roi])
            sig_tuned_summary[eo][a_idx] = percentage

        ax.bar(eo_idx,
               np.nanmean(sig_tuned_summary[eo]),
               facecolor='w',
               edgecolor='black',
               zorder=0)

        ax.errorbar(eo_idx,
                    np.nanmean(sig_tuned_summary[eo]),
                    sem(sig_tuned_summary[eo], nan_policy='omit'),
                    color='k',
                    zorder=2)

        ax.scatter(eo_idx * np.ones(len(sig_tuned_summary[eo])),
                   sig_tuned_summary[eo],
                   color='silver',
                   marker='.',
                   zorder=1)

    ax.set_xticks(range(len(eo_list)))
    ax.set_xticklabels([eo[0] for eo in eo_list])
    ax.set_ylim(0, 1.25)
    ax.set_yticks(np.arange(0, 1.25, .25))
    ax.set_ylabel('Fraction of Tuned Pixels')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    log_stat_to_file('EPI Fraction Tuned')
    p_values = bootstrap_test_dictionary(sig_tuned_summary)

    xrange = list(combinations(np.arange(0, len(eo_list), 1), 2))

    for idx, (key_a, key_b) in enumerate(combinations(eo_v_list, 2)):
        p = p_values[key_a + '-'+key_b][1]
        annotate_significance(ax, [xrange[idx][0], xrange[idx][1]], 1.24 - idx*.1, p)
    with shelve.open(summary_db_path) as db:
        db['FractionTunedPixels'] = sig_tuned_summary
    return sig_tuned_summary


def wavelength_plot(db, ax, virus=['dlx', 'hsyn'], eo_list=all_eo_groups, **kwargs):
    virus_colors = {'dlx': 'red',
                    'hsyn': 'green'}
    x_offsets = 0 if len(virus) == 1 else [-0.25, 0.25]

    with shelve.open(summary_db_path) as summary_db:
        if 'wavelengths' not in summary_db:
            wavelengths = collect_wavelengths(db)
            summary_db['wavelengths'] = wavelengths
        else:
            wavelengths = summary_db['wavelengths']

    for v_idx, virus_name in enumerate(virus):
        eo_virus_list = [f'{eo}-{virus_name}' for eo in eo_list]
        for eo_idx, eo in enumerate(eo_virus_list):

            ax.scatter(np.ones(len(wavelengths[eo]))*(x_offsets[v_idx] + 2 * eo_idx + 1),
                       np.array(wavelengths[eo])*1e-3,
                       marker='.',
                       color='lightgray')

            ax.errorbar(x_offsets[v_idx] + 2*eo_idx + 1,
                        np.mean(wavelengths[eo])*1e-3,
                        yerr=sem(wavelengths[eo])*1e-3,
                        marker='.',
                        color=virus_colors[virus_name])

    ax.set_xlim([0, 2*len(eo_list)+1])
    ax.set_xticks(range(1, 2*len(eo_list)+1, 2))
    ax.set_xticklabels([x[0] for x in eo_list])
    ax.set_ylabel('Wavelength (mm)')
    ax.set_ylim([500*1e-3, 1050*1e-3])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return wavelengths


def collect_wavelengths(db):
    wavelengths = {}

    for eo_group, aids in expt_groups.items():
        wavelengths[eo_group] = []
        for aid in aids:
            expt = db[str(aid)]
            wavelengths[eo_group].append(get_avg_wavelength(expt))
    return wavelengths


def get_avg_wavelength(expt):

    wavelengths = []

    for seed_pt, wv_map in expt.analysis[analysis_field].cache['wavelet_analysis'].items():
        wavelengths.append(np.nanmean(wv_map[0]))

    return np.nanmean(wavelengths)


def compare_wavelengths(db):
    wavelengths = {}
    uniform = {}
    for eo_group, aids in expt_groups.items():
        wavelengths[eo_group] = []
        uniform[eo_group] = []

        for aid in aids:
            expt = db[str(aid)]
            if 'wavelet_analysis_uniform' not in expt.analysis[analysis_field].cache:
                continue
            wavelengths[eo_group].append(get_avg_wavelength(expt))
            uniform[eo_group].append(np.nanmean(expt.analysis[analysis_field].cache['wavelet_analysis_uniform'][0]))

        t, p = ttest_rel(wavelengths[eo_group], uniform[eo_group])

        result_string = f'[{eo_group}]: {np.mean(wavelengths[eo_group]):0.2f} +/-  {sem(wavelengths[eo_group]):0.2f}'
        result_string += f'vs {np.mean(uniform[eo_group]):0.2f} +/-  {sem(uniform[eo_group]):0.2f}'
        result_string += f'(p = {p:0.4f}, n = {len(wavelengths[eo_group])})'
        print(result_string)
    return wavelengths, uniform


def plot_single_trial_response(expt, ax_list, trial_list, stim_id, boundaries):
    responses = expt.analysis[analysis_field].single_trial_responses()
    roi = expt.analysis[analysis_field].roi()
    responses = responses / np.nanpercentile(responses, 99.5)
    responses[~np.isfinite(responses)] = 0
    responses[responses < 0] = 0
    responses[:, :, ~roi] = 1
    resolution = expt.analysis[analysis_field].resolution()
    for trial, ax in zip(trial_list, ax_list):
        expt.analysis[analysis_field].plot_single_trial(ax,
                                                        stim_id,
                                                        trial,
                                                        scalebar=1000,
                                                        fov=boundaries,
                                                        colorbar=False,
                                                        resolution=resolution,
                                                        responses=responses,
                                                        vmin=0,
                                                        vmax=1)


def roi_bounding(roi):
    pts = np.where(roi)
    return np.min(pts[0]), np.max(pts[0])+1, np.min(pts[1]), np.max(pts[1])+1


def cropped_roi(db, ax1, ax2):

    # load ROIs and Correlation Maps
    dlx_young = db['F2334']
    dlx_old = db['F2312']

    dlx_young_roi = dlx_young.analysis['Full'].roi()

    seed_point = list(dlx_old.analysis['Full'].cache['total_corr_matrix'].keys())[0]
    dlx_old_corr = dlx_old.analysis['Full'].cache['total_corr_matrix'][seed_point]
    dlx_old_wavelength = dlx_old.analysis['Full'].cache['wavelet_analysis'][seed_point][0]

    # Shift young ROI
    dlx_young_centroid = dlx_young.analysis['Full'].roi_centroid()
    x_shift = int(dlx_young_centroid[0]-seed_point[0])
    y_shift = int(dlx_young_centroid[1]-seed_point[1])
    dlx_roi_shifted = shift(dlx_young_roi, [-x_shift, -y_shift])

    # Make Copies, mask, and compute new wavelength
    resolution = dlx_old.analysis['Full'].resolution()[0]
    min_x, max_x, min_y, max_y = roi_bounding(dlx_roi_shifted)
    trimmed_roi = dlx_young_roi[min_x:max_x, min_y:max_y]

    trimmed_img = dlx_old_corr[min_x:max_x, min_y:max_y].copy()

    trimmed_roi_img = dlx_old_corr.copy()
    trimmed_roi_img[~dlx_roi_shifted] = 0

    trimmed_img = trimmed_img - np.mean(trimmed_img[trimmed_roi])
    trimmed_img[~trimmed_roi] = 0
    result = fit_wavelet(trimmed_img,
                         resolution,
                         roi=trimmed_roi,
                         max_wavelength=1000,
                         min_wavelength=500,
                         k_base=7,
                         wavelength_step=5,
                         interp_factor=1)

    c = ax1.imshow(dlx_old_corr, vmin=-1, vmax=1, cmap='bwr')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel(f'{np.nanmean(dlx_old_wavelength)/1000:0.3f} mm')
    ax1.scatter(seed_point[1], seed_point[0], s=10, marker='s', color='g')
    ax1.set_title('Measured ROI')

    ax2.imshow(trimmed_roi_img, vmin=-1, vmax=1, cmap='bwr')
    ax2.set_ylim((270, 0))
    ax2.set_xlim((0, 320))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel(f'{np.nanmean(result[0])/1000:0.3f} mm')
    ax2.set_title('Cropped ROI')
    ax2.scatter(seed_point[1], seed_point[0], s=10, marker='s', color='g')
    ax2.plot((200, 200 + 1000/resolution), [250, 250], color='k')  # scalebar


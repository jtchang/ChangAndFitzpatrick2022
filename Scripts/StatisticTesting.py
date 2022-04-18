import numpy as np
from scipy.stats import percentileofscore, sem, ttest_1samp, ttest_ind
from itertools import combinations
from sklearn.linear_model import LinearRegression
import datetime
stats_file = rf'Figures/stats_{datetime.datetime.now().strftime("%Y%m%d")}.txt'


def bootstrap_test_dictionary(library, iterations=10000):
    p_values = {}
    for key_a, key_b in combinations(library.keys(), 2):

        p, observed_diff = bootstrap_test(library[key_a], library[key_b], iterations=iterations)
        n_a = np.array(library[key_a]).size
        n_b = np.array(library[key_b]).size
        key_a_stats = f'{np.mean(library[key_a]):0.4f} +/- {sem(library[key_a]):0.4f}'
        key_b_stats = f'{np.mean(library[key_b]):0.4f} +/- {sem(library[key_b]):0.4f}'
        str_to_print = f'[bootstrap]{key_a} vs {key_b}: {key_a_stats} vs {key_b_stats}(p={p: 0.4f}, {n_a} vs {n_b})'
        log_stat_to_file(str_to_print)
        p_values[key_a+'-'+key_b] = (observed_diff, p, n_a, n_b)

    return p_values


def bootstrap_responsive_bar(library, iterations=10000):

    for idx, metric in enumerate(['responsive', 'osi', 'dsi']):
        for key_a, key_b in combinations(library.keys(), 2):
            p, _ = bootstrap_test(library[key_a][:, idx], library[key_b][:, idx], iterations=iterations)
            n_a = library[key_a][:, idx].size
            n_b = library[key_b][:, idx].size
            key_a_stats = f'{np.mean(library[key_a][:, idx]):0.4f} ± {sem(library[key_a][:, idx]):0.4f}'
            key_b_stats = f'{np.mean(library[key_b][:, idx]):0.4f} ± {sem(library[key_b][:, idx]):0.4f}'
            str_to_print = f'[bootstrap] ({metric}){key_a} vs {key_b}: {key_a_stats} vs {key_b_stats}(p={p: 0.4f}, {n_a} vs {n_b})'
            log_stat_to_file(str_to_print)


def ttest_dictionary(library):
    p_values = {}

    for key_a, key_b in combinations(library.keys(), 2):

        t_stat, p_values[key_a+'-'+key_b] = ttest_ind(library[key_a], library[key_b])
        obs_diff = np.mean(library[key_a]) - np.mean(library[key_b])
        p = p_values[key_a+'-'+key_b]
        str_to_print = f'[t-test] {key_a} vs {key_b}: {np.mean(library[key_a]):0.4f} vs {np.mean(library[key_b]):0.4f} (p={p:0.4f}| {t_stat:0.4f})'
        log_stat_to_file(str_to_print)
    return p_values


def test_dictionary_clustering(library):
    p_values = np.empty((len(library.keys()), ))
    for idx, (key, vals) in enumerate(library.items()):
        for bin_a, bin_b in combinations(np.arange(vals.shape[1]), 2):
            differences = vals[:, bin_a] - vals[:, bin_b]
            t_stat, p_values[idx] = ttest_1samp(differences, 0)
            str_to_print = f'[t-test] {key}- bin{bin_a} vs bin{bin_b}: ' + \
                f'{np.mean(vals[:,bin_a]):0.4f} ± {sem(vals[:,bin_a]):0.4f}vs ' + \
                f'{np.mean(vals[:,bin_b]):0.4f} ± {sem(vals[:,bin_b]):0.4f}' + \
                f'(p={p_values[idx]:0.4f}/ {t_stat:0.4f} | n = {vals.shape[0]})'
            log_stat_to_file(str_to_print)
    return p_values


def ttest_dictionary_clustering_difference(library):

    p_values = np.empty(len(list(combinations(list(library.keys()), 2))))
    for idx, (key_a, key_b) in enumerate(combinations(list(library.keys()), 2)):
        diff_a = library[key_a][:, 0] - library[key_a][:, 1]
        diff_b = library[key_b][:, 0] - library[key_b][:, 1]

        p_values[idx], _ = bootstrap_test(diff_a, diff_b)
        str_to_print = f'[bootstrap] {key_a} - {key_b} : ' + \
            f'{np.mean(diff_a):0.4f} ± {sem(diff_a):0.4f} vs {np.mean(diff_b):0.4f} ± {sem(diff_b):0.4f}' + \
            f' (p={p_values[idx]:0.4f} | {diff_a.size} vs {diff_b.size})'
        log_stat_to_file(str_to_print)

    return p_values


def clustering_within_bin_testing_diff(library):

    for idx in range(2):
        log_stat_to_file(f'Bin{idx} clustering comparison')
        bin_library = {}
        for key, val in library.items():
            bin_library[key] = val[:, idx]

        bootstrap_test_dictionary(bin_library)


def bootstrap_test(set_a, set_b, iterations=10000, two_tail=True):

    observed_diff = np.mean(set_a) - np.mean(set_b)

    pooled_data = np.concatenate((set_a, set_b))

    rs_stats = np.empty((iterations,))

    for iter in range(iterations):
        rs_a = np.random.randint(0, pooled_data.size, np.array(set_a).size)
        rs_b = np.random.randint(0, pooled_data.size, np.array(set_b).size)
        rs_stats[iter] = np.mean(pooled_data[rs_a]) - np.mean(pooled_data[rs_b])

    bootstrap_stat = percentileofscore(rs_stats, observed_diff) / 100

    if two_tail:
        if bootstrap_stat > 0.5:
            bootstrap_stat = 1 - bootstrap_stat
        bootstrap_stat = 2*bootstrap_stat

    return bootstrap_stat, observed_diff


def log_stat_to_file(str_to_print: str):
    filehandle = open(stats_file, 'a')
    filehandle.write(str_to_print+'\n')
    filehandle.close()


def annotate_significance(ax, x, y, p):
    props = {'arrowstyle': '-'}

    if p < 0.0005:
        text = '***'
    elif p < 0.005:
        text = '**'
    elif p < 0.05:
        text = '*'
    else:
        text = 'ns'

    if isinstance(x, float):
        ax.annotate(text, xy=(x, y), ha='center', fontsize=8)
    else:
        ax.annotate(text, xy=(np.mean(x), y + ax.get_ylim()[1]*.01), ha='center', fontsize=8)
        ax.annotate('', xy=(x[0], y), xytext=(x[1], y), arrowprops=props)


def ttest_trial_variability_matched_ortho(eo_list, matched, ortho):
    p_values = np.empty((len(eo_list)))
    for eo_idx, eo in enumerate(eo_list):
        differences = np.array(matched[eo]) - np.array(ortho[eo])
        t_stat, p_values[eo_idx] = ttest_1samp(differences, 0)

        str_to_print = f'[t-test] {eo}: ' + \
                       f'Matched {np.nanmean(matched[eo]):0.4f} {sem(matched[eo]):0.4f} vs ' + \
                       f'Orthogonal {np.nanmean(ortho[eo]):0.4f} {sem(ortho[eo]):0.4f} ' + \
                       f'(p={p_values[eo_idx]:0.4f}| tstat = {t_stat:0.4f} )| n = {ortho[eo].size}'

        log_stat_to_file(str_to_print)
    return p_values


def correlation_distance_linearity(library, bins):

    for eo, all_values in library.items():
        r_squared = np.full((all_values.shape[1]), np.nan)
        for a_idx, values in enumerate(np.transpose(all_values)):
            values = values[:bins.size]
            cell_bins = bins[np.isfinite(values)].reshape(-1, 1)
            values = values[np.isfinite(values)].reshape(-1, 1)

            r_squared[a_idx] = LinearRegression().fit(cell_bins, values
                                                      ).score(cell_bins, values)
        log_stat_to_file(f'{eo}: R^2 = {np.nanmean(r_squared):0.4f} +/- {sem(r_squared, nan_policy="omit"):0.4f}')


def statistic_test_decoding_accuracy(accuracy, shuffles, num_cells=40):
    log_stat_to_file('Decoding Accuracy vs Chance')
    eo_list = list(accuracy.keys())

    p_values = {}
    for eo in eo_list:
        eo_accuracy = accuracy[eo][num_cells-1, :]
        eo_shuffle = shuffles[eo][num_cells-1, :]

        t_stat, p_values[eo] = ttest_1samp(eo_accuracy-eo_shuffle, 0)

        str_to_print = f'[t-test] {eo}: ' +\
            f'Accuracy {np.nanmean(eo_accuracy):0.4f} +/- {sem(eo_accuracy, nan_policy="omit"):0.4f} vs ' +\
            f'Shuffle {np.nanmean(eo_shuffle):0.4f} +/- {sem(eo_shuffle, nan_policy="omit"):0.4f}' +\
            f'(p={p_values[eo]:0.4f} | tstat = {t_stat:0.4f}| n={eo_accuracy.size})'

        log_stat_to_file(str_to_print)


def statistic_test_decoding_comparison(accuracy, num_cells=40):
    log_stat_to_file('Decoding Accuracy Comparisons')
    cell_accuracy = {}
    for eo, eo_accuracy in accuracy.items():
        cell_accuracy[eo] = eo_accuracy[num_cells-1, :]

    return bootstrap_test_dictionary(cell_accuracy)


def statistic_test_decoding_diff(accuracy_diffs):
    accuracy = {}

    for eo, acc_array in accuracy_diffs.items():
        accuracy[eo] = acc_array[0, :]

    return bootstrap_test_dictionary(accuracy)

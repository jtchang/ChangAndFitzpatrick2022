import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.signal import medfilt
from fleappy.analysis.twophoton import TwoPhotonAnalysis


class SpontaneousAnalysis(TwoPhotonAnalysis):

    __slots__ = ['events', 'params', 'thresholds']

    def __init__(self, expt, field, params={}, **kwargs) -> None:
        super().__init__(expt,  field, **kwargs)

        self.params = {}
        self.params['threshold'] = params.get('threshold', 2)  # standard deviations above mean
        self.params['minimum_length'] = params.get('minimum_length', 1/self.expt.frame_rate())  # seconds
        self.params['minimum_iei'] = params.get('minimum_iei', .25)  # seconds
        self.params['minimum_participation'] = params.get(
            'minimum_participation', self.expt.num_roi()//10)  # of cells to qualify as an event
        self.params['median_filter'] = params.get('median_filter', .5)  # number of seconds to median filter
        self.events = []
        self.thresholds = np.full((self.expt.num_roi(),), np.nan)

    def run(self, **kwargs) -> None:
        """Run all associated analyses
        """
        super().run(**kwargs)
        self.identify_events()
        self.event_correlations(**kwargs)
        self.cellular_correlations(**kwargs)

    def event_lengths(self):
        """Get length of spontaneous events in seconds.

        Returns:
            np.ndarray: Event durations in seconds
        """
        return np.array([(stop-1)-start for start, stop, _ in self.events]) / self.expt.frame_rate()

    def event_frequency(self):
        times = self.expt.get_frame_times()
        period = (times[-1] - times[0]) / 60  # minutes recorded

        return len(self.events) / period  # events/minute

    def identify_events(self):
        """Return Spontaneous event frames

            Sets events property to a list of tuples. The first two elements of the tuple are [start idx, stop idx).
            The last element of the tuple is a boolean array of cells that were active over that time period.
        """
        minimum_duration = int(np.round(self.params['minimum_length']*self.expt.frame_rate()))
        minimum_iei = int(np.round(self.params['minimum_iei'] * self.expt.frame_rate()))

        times, ts_data = self.expt.get_all_tseries(self.field)

        if self.params['median_filter'] is not None:
            filter_window = round_to_odd(self.params['median_filter'] * self.expt.frame_rate())
            if filter_window > 1:
                ts_data = medfilt(ts_data, (1, filter_window))

        self.thresholds = find_thresholds(ts_data, self.params['threshold'])

        valid_cells = ~np.any(np.isnan(ts_data), axis=1)
        ts_data = ts_data[valid_cells]
        thresholds = self.thresholds[valid_cells]

        cells_active_per_frame = ts_data > np.tile(np.expand_dims(thresholds, axis=1), (1, times.shape[0]))
        active_frames = np.where(np.sum(cells_active_per_frame, axis=0) > self.params['minimum_participation'])[0]

        start_idx = active_frames[0]
        for idx_a, idx_b in zip(active_frames[:-1], active_frames[1:]):

            if idx_b == active_frames[-1]:
                if idx_b-start_idx > minimum_duration:
                    active_cells = np.any(cells_active_per_frame[:, start_idx:idx_b+1], axis=1)
                    self.events.append((start_idx, idx_b+1, active_cells))
            elif idx_b - idx_a > minimum_iei:
                if idx_a-start_idx > minimum_duration:
                    active_cells = np.any(cells_active_per_frame[:, start_idx:idx_a+1], axis=1)
                    self.events.append((start_idx, idx_a+1, active_cells))
                start_idx = idx_b

    def event_frames(self, **kwargs):
        kwargs['invalids'] = kwargs.get('invalids', False)

        _, ts_data = self.expt.get_all_tseries(self.field)

        if not kwargs['invalids']:
            valid_cells = ~np.any(~np.isfinite(ts_data), axis=1)
            ts_data = ts_data[valid_cells]

        frames = np.empty((len(self.events), ts_data.shape[0]))
        for idx, (start_idx, stop_idx, _) in enumerate(self.events):
            frames[idx, :] = np.mean(ts_data[:, start_idx:stop_idx], axis=1)

        return frames

    def event_participation(self)->np.ndarray:
        """Get the fraction of active cells during events.

        Returns:
            np.ndarray: Array of fraction of active cells
        """
        return np.array([np.sum(x)/len(x) for _, _, x in self.events])

    def load_events(self):
        """Load Events

        Returns:
            [type]: [description]
        """
        if len(self.events) == 0:
            self.identify_events()

        events = np.empty((len(self.events), self.expt.num_roi()))
        if len(events > 0):
            _, ts_data = self.expt.get_all_tseries(self.field)

            for e_idx, (start_idx, stop_idx) in enumerate(self.events):
                events[e_idx] = np.mean(ts_data[:, start_idx, stop_idx], axis=1)

        return events

    def event_correlations(self, **kwargs):
        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['override'] = kwargs.get('override', False)
        if 'event_event_corr_matrix' in self.cache and not kwargs['override']:
            corr_matrix = self.cache['event_event_corr_matrix']

        else:
            events = self.event_frames(invalids=False)   # by default event correlations must drop invalid cells
            corr_matrix = np.empty((events.shape[0], events.shape[0]))

            for idx_a in range(events.shape[0]):
                for idx_b in range(idx_a, events.shape[0]):
                    corr_matrix[idx_a, idx_b] = np.corrcoef(events[idx_a], events[idx_b])[0][1]
                    corr_matrix[idx_b, idx_a] = corr_matrix[idx_a, idx_b]

            if kwargs['cache']:
                self.cache['event_event_corr_matrix'] = corr_matrix

        return corr_matrix

    def event_correlation_metric(self, **kwargs):
        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['override'] = kwargs.get('override', False)

        corr_matrix = self.event_correlations(**kwargs)
        identity_matrix = np.identity(corr_matrix.shape[0], dtype=bool)

        metric = np.mean(corr_matrix[~identity_matrix])

        if kwargs['cache']:
            self.cache['event_event_corrmetric'] = metric

        return metric

    def cellular_correlations(self, seed_cell=None, **kwargs):
        kwargs['cache'] = kwargs.get('cache', True)
        kwargs['override'] = kwargs.get('override', False)
        cellular_correlations = None
        if not kwargs['override'] and kwargs['cache'] and 'cellular_correlations' in self.cache:
            cellular_correlations = self.cache['cellular_correlations']
        else:
            events = np.transpose(self.event_frames(invalids=True), (1, 0))

            cellular_correlations = np.full((events.shape[0], events.shape[0]), np.nan)
            for seed_idx, seed_resp in enumerate(events):
                for cell_idx, resp in enumerate(events[seed_idx:, :]):
                    cellular_correlations[seed_idx, cell_idx+seed_idx] = np.corrcoef(seed_resp, resp)[0][1]
                    cellular_correlations[cell_idx+seed_idx,
                                          seed_idx] = cellular_correlations[seed_idx, cell_idx+seed_idx]

            if kwargs['cache']:
                self.cache['cellular_correlations'] = cellular_correlations
        if seed_cell is not None:
            return cellular_correlations[seed_cell, :]
        return cellular_correlations

    def plot_scatter_correlations(self, seed_cell, **kwargs):
        kwargs['ax'] = kwargs.get('ax', None)
        kwargs['cmap'] = kwargs.get('cmap', 'bwr')
        kwargs['vmin'] = kwargs.get('vmin', -1)
        kwargs['vmax'] = kwargs.get('vmax', 1)
        kwargs['colorbar'] = kwargs.get('colorbar', True)
        kwargs['scale_bar'] = kwargs.get('scale_bar', 100)

        if kwargs['ax'] is None:
            fig = plt.figure()
            kwargs['ax'] = fig.add_subplot(111)

        centroids = self.expt.roi_positions()
        correlations = self.cellular_correlations(seed_cell=seed_cell)

        seed_pos = centroids[seed_cell, :]

        centroids = np.delete(centroids, seed_cell, axis=0)
        correlations = np.delete(correlations, seed_cell, axis=0)

        scatter_mappable = kwargs['ax'].scatter(centroids[:, 0],
                                                centroids[:, 1],
                                                c=correlations,
                                                vmin=kwargs['vmin'], vmax=kwargs['vmax'], cmap=kwargs['cmap']
                                                )

        kwargs['ax'].scatter(seed_pos[0], seed_pos[1], marker='^', zorder=2, color='lime')
        kwargs['ax'].set_facecolor()
        self.format_scatter_plot(**kwargs)

        if kwargs['colorbar']:
            cbar = plt.colorbar(scatter_mappable, ax=kwargs['ax'])
            cbar.set_label('Response Correlation', rotation=270)
        return scatter_mappable

    def plot_events_with_active_frames(self):
        """[summary]
        """

        times, ts_data = self.expt.get_all_tseries(self.field)
        times = times - times[0]

        for start, stop, _ in self.events:
            plt.plot([times[start], times[stop-1]], [0, 0])

        if self.params['median_filter'] is not None:
            filter_window = round_to_odd(self.params['median_filter'] * self.expt.frame_rate())
            if filter_window > 1:
                ts_data = medfilt(ts_data, (1, filter_window))

        valid_cells = ~np.any(np.isnan(ts_data), axis=1)
        ts_data = ts_data[valid_cells]
        thresholds = self.thresholds[valid_cells]

        cells_active_per_frame = ts_data > np.tile(np.expand_dims(thresholds, axis=1), (1, times.shape[0]))
        active_frames = np.sum(cells_active_per_frame, axis=0) > self.params['minimum_participation']

        plt.plot(times, active_frames)

    def plot_raw_fluorescence_traces(self, ax=None, **kwargs):

        kwargs['zscore_f'] = kwargs.get('zscore_f', False)
        kwargs['normalize_f'] = kwargs.get('normalize_f', False)
        kwargs['colorbar'] = kwargs.get('colorbar', False)
        kwargs['cmap'] = kwargs.get('cmap', 'magma')

        times, ts_data = self.expt.get_all_tseries(self.field)
        valid_cells = ~np.any(np.isnan(ts_data), axis=1)
        ts_data = ts_data[valid_cells]

        times = times - times[0]
        if kwargs['zscore_f']:
            ts_data = zscore(ts_data, axis=1)
        elif kwargs['normalize_f']:
            ts_data = ts_data / np.tile(np.expand_dims(np.nanmax(ts_data, axis=1), axis=1), (1, times.shape[0]))
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        fig_mappable = ax.imshow(ts_data,
                                 vmin=np.nanmin(ts_data),
                                 vmax=np.nanmax(ts_data),
                                 cmap=kwargs['cmap'],
                                 aspect='auto',
                                 extent=[times[0], times[-1], ts_data.shape[0]-1, 0])
        if kwargs['colorbar']:
            cb = plt.colorbar(fig_mappable, ax=ax)
            if kwargs['zscore_f']:
                cb.set_label('Z-Score ΔF/F', rotation=270)
            if kwargs['normalize_f']:
                cb.set_label('Normalized ΔF/F', rotation=270)
            else:
                cb.set_label('ΔF/F', rotation=270)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cell Number')
        return fig_mappable


def find_thresholds(ts_data, num_sds):

    return np.nanmean(ts_data, axis=1) + num_sds * np.nanstd(ts_data, axis=1)


def round_to_odd(number):
    return int(np.floor(number) + (number % 2 < 1))

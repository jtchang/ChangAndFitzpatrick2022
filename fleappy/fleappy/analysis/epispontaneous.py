import logging
import numpy as np
from scipy.ndimage import median_filter
import scipy.sparse
from typing import List, Set, Dict, Tuple, Optional
from fleappy.analysis.epi import EpiAnalysis
from fleappy.analysis.epimapstructure import fit_wavelet
from fleappy.filter import fermi


class EpiSpontaneousAnalysis(EpiAnalysis):
    def __init__(self, expt, field, **kwargs):
        super().__init__(expt, field, **kwargs)

    def run(self, **kwargs):
        """Run all spontaneous analyses.
        """
        super().run()
        self.compute_correlation_structure()
        # self.wavelet_analysis()

    def identify_events(self, frame_thresholds: float = 0.05, pixel_thresholds: np.ndarray = None,
                        filter_windows: Tuple[float, float] = (.3, 60),
                        minimum_iei: float = 0.3, minimum_duration: float = 0.5, cache: bool = True) -> np.ndarray:
        """Identify spontaneous events.

        Args:
            frame_thresholds (float, optional): [description]. Defaults to 0.05.
            pixel_thresholds (np.ndarray, optional): [description]. Defaults to None.
            filter_windows (Tuple[float, float], optional): [description]. Defaults to (.3, 60).
            minimum_iei (float, optional): [description]. Defaults to 0.3.
            minimum_duration (float, optional): [description]. Defaults to 0.5.
            cache (bool, optional): [description]. Defaults to True.

        Cached Values:

        Returns:
            np.ndarray: list of frame pairs of start and stop index of events (n,2)
        """
        batch_size = 100
        if pixel_thresholds is None:
            frame_mean, frame_std = self.compute_thresholds()
            pixel_thresholds = 2*frame_std + frame_mean
        roi = self.expt.roi(self.field)
        file_reader = self.expt._access_files(series_type=self.field)
        total_frames = file_reader.sizes['t']

        active_pixels = np.empty((total_frames,), dtype=int)
        for f_idx in range(0, total_frames, batch_size):
            load_frames = self.expt.get_frames(
                f_idx, f_idx + batch_size, file_reader=file_reader)
            for idx, frame in enumerate(load_frames):
                # > active_pixels
                active_pixels[idx +
                              f_idx] = np.sum(frame[roi] > pixel_thresholds[roi])

        # filter and detrend our active pixel data

        filt_size = np.ceil(np.array(filter_windows) *
                            self.expt.metadata.frame_rate()) // 2 * 2 + 1
        filtered_active_pixels = median_filter(active_pixels, filt_size.astype(
            int)[0]) - median_filter(active_pixels, size=(filt_size.astype(int)[1],))

        frame_pixel_threshold = frame_thresholds * np.sum(roi)

        active_frames = np.where(
            filtered_active_pixels > frame_pixel_threshold)

        event_list = np.ndarray((0, 2), dtype=int)
        event_count = 0
        last_frame = np.inf

        for frame in active_frames[0]:
            if event_list.shape[0] == 0:
                event_list = np.append(event_list, np.expand_dims(
                    np.array([frame, frame], dtype=int), 0), axis=0)

            elif frame-last_frame > (minimum_iei*self.expt.metadata.frame_rate()):
                event_list[event_count, 1] = last_frame
                event_count = event_count+1
                event_list = np.append(event_list, np.expand_dims(
                    np.array([frame, frame], dtype=int), 0), axis=0)
            last_frame = frame

        event_list = np.array([x for x in event_list if (x[1]-x[0]) >=
                               (minimum_duration*self.expt.metadata.frame_rate())])

        if cache:
            self.cache['event_list'] = event_list
            self.cache['active_pixels'] = active_pixels
            self.cache['filtered_active_pixels'] = filtered_active_pixels

        return event_list

    def collect_event_frames(self, cache=True) -> np.ndarray:
        """Collapse all spontaneous events collapsed in time.

        Args:
            cache (bool, optional): [description]. Defaults to True.

        Returns:
            np.ndarray: Event frames (n,y,x)
        """
        if 'event_list' in self.cache.keys():
            event_list = self.cache['event_list']
        else:
            logging.info('[%s] Identifying Events!', self.expt.animal_id)
            event_list = self.identify_events(cache=cache)
        file_reader = self.expt._access_files(self.field)
        roi = self.roi()

        all_event_frames = np.empty(
            (len(event_list), roi.shape[0], roi.shape[1]))

        for idx, (start, stop) in enumerate(event_list):
            frame_event = np.nanmean(self.expt.get_frames(
                start, stop, file_reader=file_reader), axis=0)
            frame_event[np.isinf(frame_event) | np.isnan(frame_event)] = 0

            frame_event = fermi.filter(frame_event,
                                       cutoff=self.filter_params['cutoffs'], resolution=self.filter_params['resolution'])
            frame_event[np.logical_not(roi)] = 0
            frame_event[frame_event < 0] = 0
            all_event_frames[idx, :, :] = frame_event

        if cache:
            self.cache['event_frames'] = all_event_frames

        file_reader.close()

        return all_event_frames

    def compute_thresholds(self, cache=True, override=False):
        """[summary]

        Args:
            cache (bool, optional): Store value in cache. Defaults to True.
            override (bool, optional): Bypass cache and recompute. Defaults to False.

        Returns:
            [np.ndarray]: Average Pixel Values
            [np.ndarray]: Standard Deviation of Pixel Value.
        """
        batch_size = 250

        if cache and not override and 'frame_mean' in self.cache.keys() and 'frame_std' in self.cache.keys():
            return self.cache['frame_mean'], self.cache['frame_std']

        file_reader = self.expt._access_files(series_type=self.field)

        total_frames = file_reader.sizes['t']

        all_means = np.empty((int(np.ceil(total_frames/batch_size)),
                              file_reader.sizes['y'], file_reader.sizes['x']))
        all_vars = np.empty((int(np.ceil(total_frames/batch_size)),
                             file_reader.sizes['y'], file_reader.sizes['x']))
        frame_counts = np.empty((int(np.ceil(
            total_frames/batch_size)), file_reader.sizes['y'], file_reader.sizes['x']))
        for idx, f_idx in enumerate(range(0, total_frames, batch_size)):
            load_frames = self.expt.get_frames(
                f_idx, f_idx+batch_size, file_reader=file_reader)

            all_means[idx, :, :] = np.nanmean(load_frames, axis=0)
            all_vars[idx, :, :] = np.nanvar(load_frames, axis=0)
            frame_counts[idx, :, :] = load_frames.shape[0]

        frame_mean = np.sum(all_means * frame_counts, axis=0) / total_frames

        frame_std = np.sqrt(np.sum(
            frame_counts * (all_vars + (all_means-frame_mean)**2), axis=0)/total_frames)

        if cache:
            self.cache['frame_mean'] = frame_mean
            self.cache['frame_std'] = frame_std

        return frame_mean, frame_std

    def compute_correlation_structure(self, cache=True, override=False, **kwargs):
        """Compute pixelwise correlation maps

        Args:
            cache (bool, optional): Read and/or save from cache. Defaults to True.
            override (bool, optional): For recompute. Defaults to False.

        Returns:
            [dict]: Correlation maps {seed point: map}
        """
        correlation_pts = np.concatenate([[self.roi_centroid()],
                                          self.grid_roi()], axis=0)
        roi = self.roi()
        events = self.collect_event_frames()
        if cache and 'total_corr_matrix' in self.cache and not override:
            return self.cache['total_corr_matrix']
        self.cache['total_corr_matrix'] = {}
        logging.info('[%s] Computing Correlation Maps...', self.expt.animal_id)
        correlation_matrix = {}

        for seed_point in correlation_pts:
            correlation_matrix[tuple(seed_point)] = self.correlation_map(seed_point,
                                                                         events=events,
                                                                         roi=roi)
        if cache:
            self.cache['total_corr_matrix'] = correlation_matrix

        return correlation_matrix

    def correlation_map(self, seed_point, cache=True, override=False, **kwargs):
        """Compute the total correlation map for spontaneous events

        Args:
            seed_point ([type]): Seed point for the correlation map.
            cache (bool, optional): Get the map from cached results?. Defaults to True.
            override (bool, optional): For the recomputation of the correlation map. Defaults to False.

        kwargs:
            roi (np.ndarray): Boolean mask for region of interest.
            events (np.ndarray): Identified events.

        Returns:
            [np.ndarray]: Tootal Correlation Map
        """

        if cache and not override and 'total_corr_matrix' in self.cache:
            if 'total_corr_matrix' in self.cache and isinstance(self.cache['total_corr_matrix'], dict):
                if tuple(seed_point) in self.cache['total_corr_matrix']:
                    return self.cache['total_corr_matrix'][seed_point]

        roi = kwargs.get('roi', self.roi())
        events = kwargs.get('events', self.collect_event_frames)

        correlation_matrix = np.full(roi.shape, np.nan)
        seed_pt_responses = events[:, seed_point[0], seed_point[1]]
        for roi_x, roi_y in np.argwhere(roi):
            correlation_matrix[roi_x, roi_y] = np.corrcoef(seed_pt_responses,
                                                           events[:, roi_x, roi_y])[0][1]

        if cache:
            if 'total_corr_matrix' not in self.cache:
                self.cache['total_corr_matrix'] = {}
            self.cache['total_corr_matrix'][tuple(seed_point)] = correlation_matrix

        return correlation_matrix

    def wavelet_analysis(self,  override=False, **kwargs):
        """Wavelet analysis for correlation maps

        Args:
            override (bool, optional): Force recompuation. Defaults to False.

        Returns:
            [dict]: Wavelet analysi results {seed point: [wavelet analysis information]}
        """
        kwargs['seed_points'] = kwargs.get('seed_points', self.cache[f'total_corr_matrix'].keys())

        if 'wavelet_analysis' not in self.cache or not isinstance(self.cache['wavelet_analysis'], dict):
            self.cache['wavelet_analysis'] = {}

        roi = self.roi()
        resolution = self.resolution()[0]

        min_x, max_x, min_y, max_y = self._roi_bounding()
        trimmed_roi = roi[min_x:max_x, min_y:max_y]

        for key in kwargs['seed_points']:
            if key in self.cache['wavelet_analysis'] and not override:
                continue
            logging.info('[%s] Processing %s...', self.expt.animal_id, str(key))

            corr_img = self.correlation_map(key,  **kwargs)
            trimmed_img = corr_img[min_x:max_x, min_y:max_y] - np.mean(corr_img[roi])

            trimmed_img[~trimmed_roi] = 0

            self.cache['wavelet_analysis'][key] = fit_wavelet(trimmed_img,
                                                              resolution,
                                                              roi=trimmed_roi,
                                                              k_base=7,
                                                              min_wavelength=400,
                                                              max_wavelength=1300)

        return self.cache['wavelet_analysis']

    def collect_single_event_frame(self, start, stop, file_reader=None, **kwargs):
        if file_reader is None:
            freader = self.expt._access_files(self.field)

        frame_event = self.expt.get_frames(start, stop, file_reader=freader)
        frame_event = fermi.filter(frame_event,
                                   cutoffs=self.filter_params['cutoffs'],
                                   resolution=self.filter_params['resolution'])

        # convert data to uint8
        if kwargs.get('roi', True):
            roi = self.roi()
            frame_event[:, ~roi] = 0

        if file_reader is None:
            freader.close()

        return frame_event

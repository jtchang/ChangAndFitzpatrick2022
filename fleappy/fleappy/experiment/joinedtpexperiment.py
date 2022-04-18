
from typing import Any
import numpy as np
from fleappy.experiment.tpwrapper import TPWrapper
import fleappy.experiment.baselinefunctions as baselinefunctions
import pims


class JoinedTPExperiment(TPWrapper):

    __slots__ = ['expts']

    def __init__(self, expts) -> None:
        self.expts = expts

        super().__init__(animal_id=expts[0].animal_id)

    def __str__(self, short=False):
        str_ret = f'{self.animal_id}: {self.__class__.__name__}'

        if not short:
            for expt in self.expts:
                str_ret = str_ret + expt.__str__(short=True)
        return str_ret

    def load_roi(self):
        for expt in self.expts:
            expt.load_roi()

    def load_ts_data(self, override: bool = False):
        for expt in self.expts:
            expt.load_ts_data(override=override)

    def baseline_roi(self, field: str, target_field: str, **kwargs):
        for expt in self.expts:
            expt.baseline_roi(field, target_field, **kwargs)

    def compute_dff(self, signal: str, baseline: str, dff_name: str, **kwargs):
        for expt in self.expts:
            expt.compute_dff(signal, baseline, dff_name, **kwargs)

    def update_path(self, path: str):
        super().update_path(path)
        for expt in self.expts:
            expt.update_path(path)

    def map_to_roi(self, func, **kwargs):
        """Applies a function to each roi in the associated experiment.

        Args:
            func ([type]): [description]
        """
        kwargs['append_dim'] = kwargs['append_dim'] if 'append_dim' in kwargs else -1

        results = self.expts[0].map_to_roi(func)
        while results.ndim < kwargs['append_dim']:
            results = np.expand_dims(results, results.ndim)
        if len(self.expts) > 1:
            for expt in self.expts[1:]:
                e_results = expt.map_to_roi(func)
                results = np.concatenate((results, e_results), kwargs['append_dim'])
        return results

    # Accessor Methods

    def get_path(self):
        return self.expts[0].get_path()

    def get_expt_parameter(self, field: str, **kwargs) -> Any:
        """Get Experimental Parameter.

        Using experimental parameter assigned to the first piezo stack, retrieves experiment information.

        Args:
            field (str): Field for the information

        Returns:
            Any: Information value
        """
        param = super().get_expt_parameter(field)
        if param is None:
            param = self.expts[0].get_expt_parameter(field)
        return param

    def get_trial_responses(self, roi_id: int, field: str, prepad: float = 0, postpad: float = 0) -> np.ndarray:
        """Returns single trial responses for a specified ROI.

        Args:
            roi_id (int): Desired roi # (0-N)
            field (str): Desired time series to chop into trial responses
            prepad (float, optional): Defaults to 0. Time to pad response before trial start
            postpad (float, optional): Defaults to 0. Time to pad response after trial end

        Returns:
            Tuple[np.ndarray, np.ndarray]: Trial Responses (# stims x # trials x time), Trial masks (time x stim portion [pre, stim, post])
        """

        trial_responses, trial_masks = self.expts[0].get_trial_responses(roi_id, field, prepad=prepad, postpad=postpad)
        if len(self.expts) > 1:
            for expt in self.expts[1:]:
                e_trials, _ = expt.get_trial_responses(roi_id, field, prepad=prepad, postpad=postpad)
                trial_responses = np.concatenate((trial_responses, e_trials), axis=1)

        return trial_responses, trial_masks

    def get_all_trial_responses(self, field: str, prepad: float = 0, postpad: float = 0):
        """ Returns single trial responses for all ROI

        Args:
            field (str): Desired time series to chop into trial responses
            prepad (float, optional): Defaults to 0. Time to pad response before trial start
            postpad (float, optional): Defaults to 0. Time to pad response after trial end

        Returns:
            numpy.array : Trial Responses ( # roi x # stims x # trials x time)
            numpy.array : Trial Masks (prestim, stim, post stim)
        """
        roi_responses, trial_masks = self.get_trial_responses(0, field, prepad, postpad)

        trial_responses = np.empty((self.num_roi(), *roi_responses.shape))
        trial_responses[0, :, :, :] = roi_responses

        for roi_idx in range(1, self.num_roi()):
            trial_responses[roi_idx, :, :, :], _ = self.get_trial_responses(
                roi_idx, field, prepad=prepad, postpad=postpad)

        return trial_responses, trial_masks

    def get_tseries(self, roi_id: int, field: str):
        """Returns the time series labeled with field for a roi.

        Args:
            roi_id (int): Desired roi # (0-N).
            field (str): Desired time series field.

        Returns:
            numpy.ndarray, numpy.ndarray: frame times, time series data
        """
        _, t_series = self.expts[0].get_tseries(roi_id, field)

        if len(self.expts) > 1:
            for expt in self.expts[1:]:
                _, expt_series = expt.get_tseries(roi_id, field)

                t_series = np.concatenate((t_series, expt_series), axis=0)

        return self.get_frame_times(), t_series

    def get_all_tseries(self, field: str):
        """ Returns single trial responses for all ROI

        Args:
            field (str): Desired time series to chop into trial responses
            prepad (float, optional): Defaults to 0. Time to pad response before trial start
            postpad (float, optional): Defaults to 0. Time to pad response after trial end

        Returns:
            numpy.array : Trial Responses ( # roi x # stims x # trials x time)
            numpy.array : Trial Masks (prestim, stim, post stim)
        """
        if self.num_roi() == 0:
            return None
        times, roi_resp = self.get_tseries(0, field)
        responses = np.empty((self.num_roi(), roi_resp.shape[0]))
        responses[0, :] = roi_resp

        for roi_id in range(1, self.num_roi()):
            _, responses[roi_id, :] = self.get_tseries(roi_id, field)
        return times, responses

    def get_frame_times(self):

        frame_times = self.expts[0].get_frame_times()

        if len(self.expts) > 1:
            for expt in self.expts[1:]:
                expt_frames = expt.get_frame_times()
                expt_frames = expt_frames - expt_frames[0]
                expt_frames = expt_frames + frame_times[-1] + expt_frames[1]
                frame_times = np.concatenate((frame_times, expt_frames), axis=0)

    def num_roi(self, **kwargs):
        return self.expts[0].num_roi(**kwargs)

    def stim_type(self):
        return self.expts[0].stim_type()

    def num_stims(self):
        return self.expts[0].num_stims()

    def num_trials(self):
        num_trials = 0
        for expt in self.expts:
            num_trials = num_trials + expt.num_trials()

        return num_trials

    def frame_rate(self):
        return self.expts[0].frame_rate()

    def do_blank(self):
        return self.expts[0].do_blank()

    def pixel_frame_size(self):
        return self.expts[0].pixel_frame_size()

    def scaling_factor(self):
        return self.expts[0].scaling_factor()

    def stim_duration(self):
        return self.expts[0].stim_duration()

    def get_roi(self):
        return self.expts[0].roi

    def roi_positions(self, include_z: bool = False) -> np.ndarray:
        return self.expts[0].roi_positions(include_z=include_z)

    def distance_matrix(self, include_z: bool = False) -> np.ndarray:
        return self.expts[0].distance_matrix(include_z=include_z)

    def get_frame_indices(self, **kwargs):

        frame_indices, masks = self.expts[0].get_frame_indices(**kwargs)

        if len(self.expts) > 1:
            aggregate_frames = 0
            for expt in self.expts[1:]:
                expt_indices, _ = expt.get_frame_indices(**kwargs)
                frame_indices = np.concatenate((frame_indices,
                                                expt_indices + aggregate_frames))
                aggregate_frames += expt.metadata.imaging['times'].size

        return frame_indices, masks
    # tif functions

    def get_trial_image(self, **kwargs):

        trial_responses = self.expts[0].get_trial_image(**kwargs)
        if len(self.expts) > 1:
            for expt in self.expts[1:]:
                e_trial = expt.get_trial_image(**kwargs)
                trial_responses = np.concatenate((trial_responses, e_trial), axis=1)

        return trial_responses

    def get_frames(self, frame_list, frame_ends=None, tif_files=None):
        y, x = self.metadata.pixel_frame_size()
        img_data = np.empty((y, x, len(frame_list)), dtype=np.uint16)

        current_frame = np.inf

        for idx, frame in enumerate(frame_list):
            file_num, file_frame, frame_ends, tif_files = self._frame_file(frame,
                                                                           frame_ends,
                                                                           tif_files)

            if current_frame is not file_num:
                file_data = pims.open(str(tif_files[file_num]))
                current_frame = file_num

            img_data[:, :, idx] = file_data[file_frame]

        return img_data, frame_ends, tif_files

    def _frame_file(self, frame_num, frame_ends=None, tif_files=None):
        if tif_files is None:
            tif_files = self._tif_files()
        if frame_ends is None:
            frame_ends = np.empty(len(tif_files), dtype=int)
            for idx, file in enumerate(tif_files):
                file_data = pims.open(str(file))
                frame_ends[idx] = len(file_data)
            frame_ends = np.cumsum(frame_ends)

        file_num = np.argwhere(frame_ends - frame_num > 0)

        if file_num.size == 0:
            return np.nan, np.nan, frame_ends, tif_files

        file_num = int(file_num[0])

        file_frame = frame_num if file_num == 0 else frame_num - frame_ends[file_num-1]

        return file_num, file_frame, frame_ends, tif_files

    def _tif_files(self):

        tif_list = []
        for expt in self.expts:
            tif_list = tif_list + expt._tif_files()
        return tif_list

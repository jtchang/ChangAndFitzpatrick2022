import logging
from os import linesep
from typing import Any, Tuple, List, Callable
import numpy as np
from numpy.core.multiarray import concatenate
from fleappy.analysis.orientation import OrientationAnalysis
from fleappy.experiment.tpwrapper import TPWrapper
from fleappy.experiment.tpexperiment import TPExperiment
from fleappy.roimanager import Roi


class PiezoExperiment(TPWrapper):

    __slots__ = ['slices', 'zstep']

    def __init__(self, expt_list: List[TPWrapper], zstep: float = 0,  **kwargs):
        """Construct Piezoexperiments

        Args:
            expt_list (List[TPWrapper]): List of TPExperiments (single planes)
            zstep (float, optional): distance in z between each plane. Defaults to 0.
        """
        if zstep is None:
            zstep = 0

        if 'animal_id' not in kwargs:
            kwargs['animal_id'] = expt_list[0].animal_id
        super().__init__(**kwargs)
        self.zstep = zstep
        self.slices = sorted(expt_list, key=lambda x: x.metadata.slice_id())

    def __str__(self, short: bool = None) -> str:
        str_ret = f'{self.animal_id} - {self.__class__.__name__}: {linesep}'
        for expt in self.slices:
            str_ret = str_ret + '\t' + expt.__str__(short=True)
        return str_ret

    def load_roi(self):
        for expt in self.slices:
            expt.load_roi()

    def load_ts_data(self, override: bool = False):
        for expt in self.slices:
            expt.load_ts_data(self, override=override)

    def baseline_roi(self, field: str, target_field: str, **kwargs):
        """Baseline time series.

        Args:
            field (str): Field to use for baselining.
            target_field (str): Field to use for basline.
        Kwargs:
            Uses TPExperiment baseline_roi kwargs
        """
        for expts in self.slices:
            expts.baseline_roi(field, target_field, **kwargs)

    def compute_dff(self, signal: str, baseline: str, dff_name: str, **kwargs):
        """[summary]

        Args:
            signal (str): [description]
            baseline (str): [description]
            dff_name (str): [description]
        """
        for expts in self.slices:
            expts.compute_dff(signal, baseline, dff_name, **kwargs)

    def map_to_roi(self, func: Callable, **kwargs) -> np.ndarray:
        """Applies a function to each roi in the associated experiment.

        Args:
            func(method): method to apply

        Returns:
            (np.ndarray): results of applying function to method
        """
        result = np.array([func(r) for r in self.slices[0].roi])

        if len(self.slices) > 1:
            for expt in self.slices[1:]:
                e_result = np.array([func(r) for r in expt.roi])
                result = np.concatenate((result, e_result), axis=0)
        return result

    def update_path(self, path: str):
        super().update_path(path)
        for slice_expt in self.slices:
            slice_expt.update_path(path)

    # Accessor Methods
    def get_path(self):
        return self.slices[0].get_path()

    def get_trial_responses(self, roi_id: int, field: str, prepad: float = 0, postpad: float = 0) -> tuple:
        """Get trial response traces.

        Args:
            roi_id (int): [description]
            field (str): [description]
            prepad (float, optional): [description]. Defaults to 0.
            postpad (float, optional): [description]. Defaults to 0.

        Returns:
            tuple:  Trial Responses (stim x trial x time) , Trial Masks (time, pre/stim/post)
        """
        slice_id, roi_num = self._find_roi_slice(roi_id)

        return self.slices[slice_id].get_trial_responses(roi_num, field, prepad, postpad)

    def get_tseries(self, roi_id: int, field: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get full roi time series.

        Args:
            roi_id (int): Roi Number
            field (str): Field to get time series from

        Returns:
            np.ndarray[float]: Time series
        """
        slice_id, roi_num = self._find_roi_slice(roi_id)

        return self.slices[slice_id].get_tseries(roi_num, field)

    def get_all_tseries(self, field: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get time series data for all roi.

        Args:
            field (str): Time series to use.

        Returns:
            Tuple[np.ndarray[float], np.ndarray[float]]: Tuple of times and fluorescence trace for all roi.
        """

        for idx, expt in enumerate(self.slices):
            if idx == 0:
                times, responses = expt.get_all_tseries(field)
            else:
                _, resp = expt.get_all_tseries(field)
                responses = np.concatenate((responses, resp), axis=0)

        return times, responses

    def get_all_trial_responses(self, field: str, prepad: float = 0, postpad: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Trial Responses for all

        Args:
            field (str): [description]
            prepad (float, optional): [description]. Defaults to 0.
            postpad (float, optional): [description]. Defaults to 0.

        Returns:
            Tuple[np.ndarray[float], np.ndarray[bool]]: Trial Responses,  Trial Masks (prestim, stim, poststim)
        """
        for idx, expt in enumerate(self.slices):
            if idx == 0:
                trial_responses, trial_masks = expt.get_all_trial_responses(field,
                                                                            prepad=prepad,
                                                                            postpad=postpad)
            else:
                temp_resps, _ = expt.get_all_trial_responses(field, prepad=prepad, postpad=postpad)

                if trial_responses.shape[3] != temp_resps.shape[3]:
                    time_elements = np.min([trial_responses.shape[3], temp_resps.shape[3]])
                    trial_responses = trial_responses[:, :, :, 0:time_elements]
                    trial_masks = trial_masks[0:time_elements, :]
                    temp_resps = temp_resps[:, :, :, 0:time_elements]

                trial_responses = np.concatenate((trial_responses, temp_resps), axis=0)
        return trial_responses, trial_masks

    def get_roi(self) -> List[Roi]:
        """Get all ROI.

        Returns:
            List[Roi]: List of ROIs from experiments.
        """
        roi = []

        for expt in self.slices:
            roi = roi + expt.get_roi()
        return roi

    def num_roi(self, **kwargs) -> int:
        """Return the total number of ROI for all slices.

        Returns:
            int: Total Number of ROI
        """

        num_roi = 0

        for expt in self.slices:
            num_roi += expt.num_roi(**kwargs)

        return int(num_roi)

    def frame_rate(self) -> float:
        """Get Imaging frame rate.

        Returns:
            float: frames per second
        """
        return self.slices[0].frame_rate()

    def num_stims(self):
        return self.slices[0].num_stims()

    def num_trials(self):
        return self.slices[0].num_trials()

    def do_blank(self) ->bool:
        return self.slices[0].do_blank()

    def pixel_frame_size(self)-> Tuple[int, int]:
        return self.slices[0].pixel_frame_size()

    def scaling_factor(self) -> float:
        return self.slices[0].scaling_factor()

    def stim_duration(self) -> float:
        return self.slices[0].stim_duration()

    def stim_type(self) -> str:
        return self.slices[0].stim_type()

    def roi_positions(self, include_z: bool = False) -> np.ndarray:
        """Get positions of the roi.

        Args:
            include_z (bool, optional): Include Z position, or only use the x,y projection. Defaults to False.

        Returns:
            np.ndarray[float]: Positions of roi (roi x 2/3)
        """
        if include_z:
            positions = np.zeros((self.num_roi(), 3))
            cell_idx = 0
            for expt in self.slices:
                positions[cell_idx:expt.num_roi()+cell_idx, 0:2] = expt.roi_positions()
                positions[cell_idx:expt.num_roi()+cell_idx, 2] = self.zstep*(expt.metadata.slice_id()-1)
                cell_idx = cell_idx+expt.num_roi()
        else:
            positions = np.stack(self.map_to_roi(lambda x: x.centroid()))*self.scaling_factor()

        return positions

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
            param = self.slices[0].get_expt_parameter(field)
        return param

    def get_frame_times(self):

        return self.slices[0].get_frame_times()

    def _find_roi_slice(self, roi_id: int) ->Tuple[int, int]:
        roi_nums = [x.num_roi() for x in self.slices]
        cum_roi = np.cumsum(roi_nums)
        roi_diff = cum_roi - roi_id
        slice_id = list(roi_diff).index(min(i for i in roi_diff if i > 0))
        start_idx = np.concatenate(([0], cum_roi))

        roi_num = roi_id - start_idx[slice_id]

        return slice_id, roi_num

    def __getitem__(self, key: int) -> TPExperiment:
        return self.slices[key]

    def __setitem__(self, key: int, value: TPExperiment):
        if not isinstance(value, TPExperiment):
            raise TypeError('Slice must be of type TPExperiment')
        self.slices[key] = value

    def __delitem__(self, key: int):
        self.slices.pop(key)

    def __getattribute__(self, name):
        if name == 'roi':
            return [roi for expt in self.slices for roi in expt.roi]
        return super().__getattribute__(name)

    def _tif_files(self, slice_id: int = 0):
        return self.slices[slice_id]._tif_files()

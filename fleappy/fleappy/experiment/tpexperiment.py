from math import inf
from dotenv import load_dotenv, find_dotenv
from itertools import chain
import logging
import natsort as ns
import numpy as np
import os
from pathlib import Path
from numpy.lib.function_base import percentile
from scipy.sparse import csr_matrix
import skimage.io as io
import pims
from typing import Tuple

from fleappy.metadata import TPMetadata
from fleappy.experiment.tpwrapper import TPWrapper
from fleappy.experiment.baselinefunctions import default_dff, percentile_filter
from fleappy.roimanager import Roi
from fleappy.roimanager import nproi, imagejroi

logging.getLogger(__name__)
load_dotenv(find_dotenv())


class TPExperiment(TPWrapper):
    """Two-Photon experient class.

    Class for handling two-photon experiments. Extends the BaseExperiment class.

    Attributes:
        roi (list): List of ROI objects.
    """

    __slots__ = ['roi']

    def __init__(self, path: str, expt_id: str, animal_id: str, **kwargs):
        self.roi = []
        super().__init__(animal_id=animal_id, **kwargs)

        if 'imaging' in kwargs and 'slice_id' in kwargs['imaging']:
            slice_id = kwargs['imaging']['slice_id']
        else:
            slice_id = 1

        tif_path = Path(path, expt_id, f'Registered/slice{slice_id}/')

        if Path(tif_path, 'header.mat').exists():
            header_path = Path(tif_path, 'header.mat')
        elif Path(tif_path, 'header.json'):
            header_path = Path(tif_path, 'header.json')
        else:
            EOFError('Header File not found!')

        self.metadata = TPMetadata(path=path,
                                   expt_id=expt_id,
                                   header=header_path,
                                   **kwargs)

    def __str__(self, short=False):
        str_ret = f'{self.animal_id}: {self.__class__.__name__}: '

        if short:
            str_ret = str_ret + f'Slice {self.metadata.slice_id()} {os.linesep}'
        else:
            str_ret = str_ret + f'{os.linesep}'
            for key in chain.from_iterable(getattr(cls, '__slots__', []) for cls in TPExperiment.__mro__):
                if key in ['roi']:
                    str_ret = str_ret + \
                        f'{key}:{len(getattr(self, key))}{os.linesep}'
                else:
                    str_ret = str_ret + f'{key}: {getattr(self, key)}{os.linesep}'
        return str_ret

    def load_roi(self, **kwargs):
        """Load roi from tif filed.

        Looks for rois in tif file in default path. If can not be found, looks for .zip file of ImageJ roi and converts
        them to a tif file. Loads each roi into experiment roi array.

        Raises:
            OSError: ROI files (.zip and .tif) are not available
        """

        # load roi from tif or zip
        slice_id = 'slice'+str(self.metadata.slice_id())
        roi_path = self._roi_path(slice_id=slice_id)
        logging.debug(roi_path)
        if roi_path.exists():
            rois = nproi.load_from_file(roi_path)
        else:
            zip_path = self._zip_path(slice_id=slice_id)
            logging.info(f'{zip_path}')
            if zip_path.exists():
                imagejroi.zip_to_tif(str(zip_path.as_posix()), str(roi_path.as_posix()))
                rois = nproi.load_from_file(roi_path)
            else:
                raise OSError('ROI files could not be found at %s' % zip_path)

        # load roi names
        name_path = self._name_path(slice_id=slice_id)
        if name_path.exists():
            roi_names = np.loadtxt(name_path, dtype=str, delimiter=';')
        else:
            roi_names = []
        logging.debug(rois.shape)

        # associate names to the roi and add to roi
        for idx, mask in enumerate(rois):
            roi_name = roi_names[idx] if len(roi_names) == rois.shape[0] else str(idx)
            roi = Roi(roi_id=idx,
                      name=roi_name,
                      roi_type='primary',
                      mask=csr_matrix(mask))
            if roi not in self.roi:
                self.roi.append(roi)
            else:
                logging.debug('ROI# %i already exists, skipping...', roi.id)

    def load_ts_data(self, override: bool = False, **kwargs):
        """Loads time series data based on the properties associated with the experiment.

           Load time series for roi preloaded and tif files specified in the file directory.
        """
        shift_zero = kwargs.get('shift_zero', False)
        if not override and len(self.roi) > 0 and 'rawF' in self.roi[0].ts_data.keys():
            logging.info('[%s]: Time series data already loaded...', self.animal_id)
        else:
            logging.info('[%s]: Loading Time Series Data...', self.animal_id)
            tif_files = self._tif_files()
            if len(self.roi) == 0:
                self.load_roi()

            rois = np.empty(
                (len(self.roi), self.roi[0].mask.shape[0], self.roi[0].mask.shape[1]), dtype=np.bool)
            for idx, roi in enumerate(self.roi):
                rois[idx, :, :] = roi.mask.todense()

            ts_data = np.empty((len(self.roi), 0))
            for ts_file in tif_files:
                logging.debug('Loading file: {0}'.format(ts_file.name))
                ts_temp = nproi.tseries_data(rois, io.imread(ts_file, plugin='pil'))
                ts_data = np.concatenate((ts_data, ts_temp), axis=1)

            for idx, data in enumerate(ts_data):
                self.roi[idx].ts_data['rawF'] = data

            # reconcile frames times and total frame number
            if len(self.roi[0].ts_data['rawF']) != len(self.metadata.imaging['times']):
                total_frames = int(np.min([len(self.roi[0].ts_data['rawF']), len(self.metadata.imaging['times'])]))

                self.metadata.imaging['times'] = self.metadata.imaging['times'][:total_frames]

                for roi in self.roi:
                    roi.ts_data['rawF'] = roi.ts_data['rawF'][:total_frames]

                self.metadata.drop_unrecorded_trials(self.metadata.imaging['times'][total_frames-1])

            if shift_zero:
                logging.info('[%s]: Shifting negative values', self.animal_id)
                for roi in self.roi:
                    min_val = np.nanmin(roi.ts_data['rawF'])
                    if min_val <= 0:
                        roi.ts_data['rawF'] = roi.ts_data['rawF'] - min_val + 1

    def baseline_roi(self, field: str, target_field: str, baseline_func=percentile_filter, **kwargs):
        """Baseline roi time series

        Args:
            field (str): Desired time series to baseline.
            target_field (str): Time series name to save computed baseline
            baseline_func (function, optional): Defaults to baselinefunctions.percentile_filter.
                Method used to compute the baseline
        """
        logging.info('[%s]: Baselining data...', self.animal_id)

        if 'frame_rate' not in kwargs:
            kwargs['frame_rate'] = self.metadata.frame_rate()

        for idx in range(self.num_roi()):
            self.roi[idx].ts_data[target_field] = baseline_func(self.roi[idx].ts_data[field], **kwargs)

            self.roi[idx].ts_data['baseline_params'] = kwargs

    def compute_dff(self, field: str, baseline: str, target_field: str, clip_zero=True, dff_func=default_dff):
        """Compute Delta F/ F for timeseries

        Args:
            field (str): [description]
            baseline (str): [description]
            target_field (str): [description]
            clip_zero (bool, optional): Defaults to True. Clip negative values.
            dff_func(method): Method for calculating dff, defaults to baselinefunctions.default_dff
        """
        logging.info('[%s]: Calculating dff...', self.animal_id)
        for idx in range(self.num_roi()):
            self.roi[idx].ts_data[target_field] = dff_func(self.roi[idx].ts_data[field],
                                                           self.roi[idx].ts_data[baseline])

            if clip_zero:
                self.roi[idx].ts_data[target_field][self.roi[idx].ts_data[target_field] < 0] = 0

    def map_to_roi(self, func, **kwargs):
        """Applies a function to each roi in the associated experiment.

        Args:
            func (method): method to employ

        Returns:
            (list): results of applying function to method
        """

        return np.array([func(r) for r in self.roi])

    # Attribute Accessors

    def get_trial_responses(self, roi_id: int, field: str, prepad: float = 0, postpad: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Returns single trial responses for a specified ROI.

        Args:
            roi_id (int): Desired roi # (0-N)
            field (str): Desired time series to chop into trial responses
            prepad (float, optional): Defaults to 0. Time to pad response before trial start
            postpad (float, optional): Defaults to 0. Time to pad response after trial end

        Returns:
            Tuple[np.ndarray, np.ndarray]: Trial Responses (# stims x # trials x time), Trial masks (time x stim portion [pre, stim, post])
        """
        stim_duration = self.metadata.stim_duration()
        frame_rate = self.metadata.frame_rate()

        prepad_frames = int(np.round(prepad * frame_rate))
        postpad_frames = int(np.round(postpad * frame_rate))
        stim_frames = int(np.round(stim_duration * frame_rate))

        window_length = prepad_frames + stim_frames + postpad_frames

        trial_responses = np.empty((self.metadata.num_stims(),
                                    self.metadata.num_trials(),
                                    window_length))
        logging.debug(f'{trial_responses.shape} - {window_length}')
        if isinstance(self.roi[roi_id], Roi):
            for idx, (stim, time) in enumerate(self.metadata.stim['triggers']):
                frame_idx, _ = self.metadata.find_frame_idx(time)
                frame_idx = int(frame_idx - prepad_frames)
                trial_responses[stim-1, int(np.floor(idx/self.metadata.num_stims())),
                                :] = self.roi[roi_id].ts_data[field][frame_idx:frame_idx+window_length]

        trial_masks = np.zeros((window_length, 3), dtype=bool)
        trial_masks[0:prepad_frames, 0] = True
        trial_masks[-postpad_frames:, 2] = True
        trial_masks[prepad_frames:prepad_frames+stim_frames, 1] = True
        return trial_responses, trial_masks

    def get_frame_times(self):
        return self.metadata.imaging['times']

    def get_frame_indices(self, **kwargs):
        return self.metadata.load_frame_indices(**kwargs)

    def get_tseries(self, roi_id: int, field: str):
        """Returns the time series labeled with field for a roi.

        Args:
            roi_id (int): Desired roi # (0-N).
            field (str): Desired time series field.

        Returns:
            numpy.ndarray, numpy.ndarray: frame times, time series data
        """

        return self.get_frame_times(), self.roi[roi_id].ts_data[field]

    def get_all_tseries(self, field: str):
        """Returns time series data for all roi.

        Args:
            field (str): Desires time series field.

        Returns:
            numpy.ndarray, numpy.ndarray: frame times, time series data (# cells x time)
        """

        if self.num_roi() == 0:
            return None
        times, roi_resp = self.get_tseries(0, field)

        responses = np.empty((self.num_roi(), roi_resp.shape[0]))
        responses[0, :] = roi_resp
        for roi_id in range(1, self.num_roi()):
            _, responses[roi_id, :] = self.get_tseries(roi_id, field)

        return times, responses

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

        stim_duration = self.metadata.stim_duration()
        frame_rate = self.metadata.frame_rate()
        window_length = int(np.round(prepad*frame_rate) + np.round(postpad *
                                                                   frame_rate) + np.round(stim_duration*frame_rate))

        trial_responses = np.empty((self.num_roi(),
                                    self.metadata.num_stims(), self.metadata.num_trials(),
                                    window_length))
        for idx in range(self.num_roi()):
            trial_responses[idx, :, :, :], trial_masks = self.get_trial_responses(
                idx, field, prepad=prepad, postpad=postpad)
        return trial_responses, trial_masks

    def num_roi(self, **kwargs):
        """Return the total number of ROI.

        Returns:
            int: Number of ROI associated with experiment.
        """

        return len(self.roi)

    def get_roi(self):
        return self.roi

    # tif functions

    def get_fov_image(self):
        return np.mean(pims.open(self._tif_files()[0]), axis=2)



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

    def _tif_path(self, slice_id=None):
        if slice_id is None:
            slice_id = 'slice' + str(self.metadata.slice_id())
        return Path(self.metadata.expt['path'], self.metadata.expt['expt_id'], f'Registered/{slice_id}/')

    def _tif_files(self, slice_id=None):
        tif_path = self._tif_path(slice_id=slice_id)
        return ns.natsorted(list(tif_path.glob('stack_*.tif')), alg=ns.PATH)

    def _roi_path(self, slice_id=None):
        if slice_id is None:
            slice_id = 'slice' + str(self.metadata.slice_id())
        expt_id = self.metadata.expt['expt_id']
        pth = self.metadata.expt['path']
        default_path = Path(pth,
                            expt_id,
                            f'Registered/{slice_id}_ROIs.tif')

        return default_path

    def _zip_path(self, slice_id=None):

        if slice_id is None:
            slice_id = 'slice' + str(self.metadata.slice_id())

        expt_id = self.metadata.expt['expt_id']
        return Path(
            self.metadata.expt['path'],
            expt_id,
            f'Registered/{slice_id}/{expt_id}_ROIs.zip')

    def _name_path(self, slice_id=None):
        if slice_id is None:
            slice_id = 'slice' + str(self.metatadata.slice_id())
        return Path(self.metadata.expt['path'],
                    self.metadata.expt['expt_id'],
                    f'Registered/{slice_id}_ROIs.tif.names')

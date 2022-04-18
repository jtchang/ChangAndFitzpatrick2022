"""Class Definition for Two-Photon Imaging metadata
"""

import logging
import math
import numpy as np
import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from fleappy.metadata.basemetadata import BaseMetadata
from fleappy.tiffread.scanimage import parse_header_mat, parse_header_json

logging.getLogger(__name__)


class TPMetadata(BaseMetadata):
    """Two-Photon Imaging metadata.

    Class for handling Two-Photon imaging sessions. Extends the BaseMetadata class.

    Attributes:
        imaging (dict): Dictionary for two-photon imaging information including times.
    """

    __slots__ = ['imaging']

    def __init__(self, path=None, expt_id=None, **kwargs):
        BaseMetadata.__init__(self, path=path, expt_id=expt_id, **kwargs)
        self.imaging = {'times': np.empty(0,), 'zoom': np.nan}

        logging.info(kwargs['header'])
        if 'header' in kwargs:
            if kwargs['header'].exists():
                logging.info('Pulling header from %s' % kwargs['header'])

                if kwargs['header'].suffix == '.mat':
                    self.imaging['header'] = parse_header_mat(str(kwargs['header']))
                elif kwargs['header'].suffix == '.json':
                    self.imaging['header'] = parse_header_json(str(kwargs['header']))
                else:
                    logging.warning('No other header formats are supported at the moment')

        if 'imaging' in kwargs:
            for key, value in kwargs['imaging'].items():
                self.imaging[key] = value
        if 'slice_id' not in self.imaging:
            self.imaging['slice_id'] = 1
        if 'num_slices' not in self.imaging:
            self.imaging['num_slices'] = self.piezo_slices()

        if path is not None:
            self.load_two_photon()
            self.load_stims()

    def __str__(self):
        str_ret = f'{self.__class__.__name__}: {os.linesep}'
        str_ret = str_ret + f'{BaseMetadata.__str__(self)}{os.linesep}'
        str_ret = str_ret + f'imaging: {self.imaging}{os.linesep}'
        return str_ret + '>'

    def load_two_photon(self, slice_id: tuple = None, override_file: str = None, override_file_name: str = None):
        """Load frame triggers.
            override_file (str, optional): Defaults to None. Filepath as string to a file of 2p frame triggers.
        """
        if slice_id is None:
            slice_id = (self.slice_id(), self.piezo_slices())
        if override_file == None and override_file_name == None:
            filepath = Path(self.expt['path'], self.expt['expt_id'], os.getenv("DEFAULT_TWOPHOTON_FRAME_TIMES"))
        else:
            filepath = Path(override_file)

        tp_time_file = open(filepath, 'r')
        file_contents = [x for x in tp_time_file.read().rstrip().split(' ') if x]

        if slice_id[1] > 1:
            file_contents = file_contents[slice_id[0]-1::slice_id[1]]

        if len(file_contents) > 0:
            self.imaging['times'] = np.array(file_contents).astype(np.float)
        else:
            raise EOFError("Empty frame trigger file %s", filepath)

    def update_path(self, path: str):
        super().update_path(path)

    # Accessor Methods

    def find_frame_idx(self, timestamp: float):
        """Find the closest frame trigger for associated time.

        Args:
            timestamp (float): Target time to look for.

        Returns:
            (int, float): Closest two-photon frame idx, Closest two-photon frame time
        """

        idx = np.searchsorted(self.imaging['times'], timestamp)
        if idx > 0 and idx == len(
                self.imaging['times']) or math.fabs(
                timestamp - self.imaging['times'][idx - 1]) < math.fabs(
                timestamp - self.imaging['times'][idx]):
            return (idx-1, self.imaging['times'][idx-1])
        else:
            return (idx, self.imaging['times'][idx])

    def frame_rate(self)->float:
        """Get two-photon imaging frame rate.
        Returns:
            float: frames per second
        """

        return 1/np.mean(np.diff(self.imaging['times']))

    def zoom(self) -> float:
        """Get the imaging zoom factor

        Returns:
            float: Imaging Zoom factor
        """
        if 'header' in self.imaging and 'SI' in self.imaging['header']:
            return self.imaging['header']['SI']['hRoiManager']['scanZoomFactor']
        elif 'zoom' in self.imaging:
            return self.imaging['zoom']

        AttributeError('Unsupported header version')

    def piezo_slices(self) -> int:
        if 'header' in self.imaging:
            if 'SI' in self.imaging['header']:
                if self.imaging['header']['SI']['hFastZ']['enable']:
                    return int(self.imaging['header']['SI']['hFastZ']['numFramesPerVolume'])
                else:
                    return 1
            else:
                logging.warning('Unsupported header version')
                return 1
        elif 'num_slices' in self.imaging:
            return self.imaging['num_slices']

        return 1

    def piezo_step_size(self) -> float:
        if self.piezo_slices() > 1:
            if 'header' in self.imaging:
                if 'SI' in self.imaging['header']:
                    return self.imaging['header']['SI']['hStackManager']['stackZStepSize']

        return 0

    def slice_id(self)->int:
        """Returns the piezo slice id

        Returns:
            int: Piezo slice id [1-number of piezo slices]
        """
        return self.imaging['slice_id']

    def pixel_frame_size(self) -> np.array:

        if 'header' in self.imaging and 'SI' in self.imaging['header']:
            frame_size = np.array([int(self.imaging['header']['SI']['hRoiManager']['pixelsPerLine']),
                                   int(self.imaging['header']['SI']['hRoiManager']['linesPerFrame'])])
        else:
            logging.info(f'Using Default Frame Size for {self.expt["path"]}')
            frame_size = np.array((512, 512))
        return frame_size

    def scaling_factor(self)->float:
        """Returns the correction factor for two photon zoom
        Returns:
            float: scaling factor in microns/pixel
        """
        zoom = self.zoom()
        if zoom == 1:
            scaling_factor = 2.73
        elif zoom == 1.2:
            scaling_factor = 2.025
        elif zoom == 1.3:
            scaling_factor = 1.83
        else:
            scaling_factor = 2.73 * 1 / zoom
        frame_size = self.pixel_frame_size()[0]

        return scaling_factor * (512.0/frame_size)

    def load_frame_indices(self, **kwargs):

        prepad = kwargs.get('prepad', 0)
        postpad = kwargs.get('postpad', 0)

        frame_rate = self.frame_rate()
        stim_duration = self.stim_duration()

        prepad_frames = int(np.round(prepad * frame_rate))
        postpad_frames = int(np.round(postpad * frame_rate))
        stim_frames = int(np.round(stim_duration * frame_rate))

        window_length = prepad_frames + stim_frames + postpad_frames

        frame_numbers = np.empty((self.num_stims(),
                                  self.num_trials(),
                                  window_length), dtype=int)

        for idx, (stim, time) in enumerate(self.stim['triggers']):
            frame_idx, _ = self.find_frame_idx(time)
            frame_idx = int(frame_idx-prepad_frames)

            frame_numbers[stim-1, int(np.floor(idx/self.num_stims())), :] = np.arange(frame_idx,
                                                                                      frame_idx+window_length,
                                                                                      1, dtype=int)

        trial_masks = np.zeros((window_length, 3), dtype=bool)
        trial_masks[0:prepad_frames, 0] = True
        trial_masks[-postpad_frames:, 2] = True
        trial_masks[prepad_frames:prepad_frames+stim_frames, -1] = True

        return frame_numbers, trial_masks

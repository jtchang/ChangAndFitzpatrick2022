import logging
from math import fabs
import numpy as np
import os
from pathlib import Path
from fleappy.metadata.basemetadata import BaseMetadata

DEFAULT_FRAME_RATE = 15.1
DEFAULT_RESOLUTION = 1000/172


class EpiMetadata(BaseMetadata):

    def __init__(self, path=None, expt_id=None, scaling_factor=(1, 1, 1), **kwargs):
        #global DEFAULT_RESOLUTION, DEFAULT_FRAME_RATE
        BaseMetadata.__init__(self, path=path, expt_id=expt_id, **kwargs)

        self.imaging = {'times': np.empty(0,),
                        'scaling_factor': scaling_factor,
                        'resolution': (EpiMetadata.default_resolution(), EpiMetadata.default_resolution())}

        self.load_frame_times()

    def __str__(self):
        str_ret = f'{self.__class__.__name__}: {os.linesep}'
        str_ret = str_ret + f'{BaseMetadata.__str__(self)}{os.linesep}'
        str_ret = str_ret + f'imaging: {self.imaging}{os.linesep}'
        return str_ret + '>'

    def load_frame_times(self, override_file: str = None):
        """Loads frame information from a file.
            override_file (str, optional): Defaults to None. Filepath as string to a file of Epi frame triggers.
        """

        if override_file == None:
            filepath = Path(self.expt['path'], self.expt['expt_id'], 'twophotontimes.txt')
        else:
            filepath = Path(override_file)

        if filepath.exists():
            tp_time_file = open(filepath, 'r')
            file_contents = [x for x in tp_time_file.read().rstrip().split(' ') if x]
            if len(file_contents) > 0:
                self.imaging['times'] = np.array(file_contents).astype(np.float)
            else:
                logging.warning(f'Empty frame trigger file {filepath.absolute()}, using default frame rate')
                self.imaging['times'] = []

        else:
            logging.warning(f'No frame triggers found, using default frame rate {filepath.absolute()}')
            self.imaging['times'] = []

    def frame_rate(self):
        return 1 / np.mean(
            np.diff(self.imaging['times'])) if not len(
            self.imaging['times']) == 0 else EpiMetadata.default_frame_rate()

    def find_frame_idx(self, timestamp):
        idx = np.searchsorted(self.imaging['times'], timestamp)
        if idx > 0 and idx == len(
                self.imaging['times']) or fabs(
                timestamp - self.imaging['times'][idx - 1]) < fabs(
                timestamp - self.imaging['times'][idx]):
            return (idx-1, self.imaging['times'][idx-1])
        else:
            return (idx, self.imaging['times'][idx])

    @staticmethod
    def default_resolution():
        return 1000/172  # microns per pixel for 4x4 binning

    @staticmethod
    def default_frame_rate():
        return 15.1  # frames/sec

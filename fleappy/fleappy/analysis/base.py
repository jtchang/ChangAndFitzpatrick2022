from pathlib import Path
import pickle
import logging

from collections import defaultdict
from typing import Union

import imageio
import numpy as np

import fleappy.experiment.baseexperiment as be


class BaseAnalysis(object):
    """A metaclass for analysis of experiments.

    This is a base class for experiment analysis that handles associating analyses with an experiment, an id, and the
    associated data field.

    TODO:
    * Move Cache to use Shelve so that memory overhead is limited

    Attributes:
        expt (fleappy.experiment.BaseExperiment): Experiment analysis is associated with
    """
    __slots__ = ['expt', 'id', 'field', 'cache']

    def __init__(self, expt: be.BaseExperiment, field: str, **kwargs):
        if not isinstance(expt, be.BaseExperiment):
            raise TypeError('%s is not a Experiment Object' % (expt))
        self.expt = expt
        self.id = kwargs['analysis_id'] if 'analysis_id' in kwargs else field
        self.field = field
        self.cache = {}

    def __str__(self):
        return f'{type(self)}: {self.id}'

    @staticmethod
    def _return_none():
        """Returns None for Default Dict

        This method returns None, so that defaultdict used with cache can be pickled.
        Returns:
            None
        """
        return None

    def run(self, **kwargs):
        """Run all analyses.
        """
        logging.debug('Running Analysis...')

    def map_to_roi(self, func):
        """Applies a function to each roi in the associated experiment.

        Args:
            func (method): method to employ

        Returns:
            (list): results of applying function to method
        """

        return self.expt.map_to_roi(func)

    def clear_cache(self, key: str = None):
        """Clear Cache.

        Delete key from cache or delete all values of key is not specified.

        Args:
            key (str, optional): key. Defaults to None.
        """
        if key is None:
            self.cache = {}
            logging.info('[%s] Cleared all of cache', self.expt.animal_id)
        else:
            logging.info('[%s] Clearing %s from cache', self.expt.animal_id, key)
            if key in self.cache:
                del self.cache[key]

    def save_cache_to_file(self, file_target: Union[str, Path]):
        """Save Cache to a file.

        Args:
            file_target (Union[str, Path]): Location for the cache to be saved.
        """
        pickle.dump(dict(self.cache), open(file_target, 'wb'), protocol=4)

    def load_cache_from_file(self, file_target: Union[str, Path]):
        """Load Cache from file.


        Args:
            file_target (Union[str, Path]): Cache Location

        Raises:
            AttributeError: When cache is not empty raises an error.

        """
        if not self.cache:
            self.cache = defaultdict(lambda: None, pickle.load(open(file_target, 'rb')))
        else:
            raise AttributeError('Currently overwriting cache is not supported. Please clear cache first')

    def _save_movie(self, filename, event, fps, **kwargs):

        if event.dtype == np.uint8 and not kwargs.get('scale', False):
            scaled_event = event
        else:
            scaled_event = 255.0 * event / np.nanpercentile(event, kwargs.get('percentile', 99))
            scaled_event[~np.isfinite(scaled_event)] = 0

            scaled_event[scaled_event > 255] = 255
            scaled_event[scaled_event < 0] = 0
            scaled_event = scaled_event.astype(np.uint8)
        if kwargs.get('annotate_frames', None) is not None:
            scaled_event[kwargs.get('annotate_frames', None), 10:20, 10:50] = 255

        writer = imageio.get_writer(filename, fps=fps)

        for frame in scaled_event:

            writer.append_data(frame)

        writer.close()
        return None

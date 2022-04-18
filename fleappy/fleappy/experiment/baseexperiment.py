import pickle
import logging
import fleappy.metadata.basemetadata as basemetadata
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class BaseExperiment(object):
    """Base Experiment Class.

    Attributes:
        metadata (list): List of metadata objects.
        analysis (dict): Dictionary for analysis.
    """

    __slots__ = ['metadata', 'analysis', 'animal_id']

    def __init__(self, animal_id=None, **kwargs):
        self.analysis = {}
        self.metadata = None
        self.animal_id = animal_id

    def save_to_file(self, filename):
        logging.info('Saving to %s', filename)
        filehandler = open(filename, 'wb')
        pickle.dump(self, filehandler, protocol=4)

    def get_expt_parameter(self, field, **kwargs):
        if isinstance(self.metadata, basemetadata.BaseMetadata):
            return self.metadata.get_expt_params(field, **kwargs)
        return None

    def get_path(self):
        if isinstance(self.metadata, basemetadata.BaseMetadata):
            return self.metadata.get_path()

    def update_path(self, path: str):
        if isinstance(self.metadata, basemetadata.BaseMetadata):
            self.metadata.update_path(path)

    def stim_type(self):
        if isinstance(self.metadata, basemetadata.BaseMetadata):
            return self.metadata.stim_type()

    def frame_rate(self):
        if isinstance(self.metadata, basemetadata.BaseMetadata):
            return self.metadata.frame_rate()

    def do_blank(self):
        if isinstance(self.metadata, basemetadata.BaseMetadata):
            return self.metadata.do_blank()

    def pixel_frame_size(self, **kwargs):
        if isinstance(self.metadata, basemetadata.BaseMetadata):
            return self.metadata.pixel_frame_size()

    def scaling_factor(self):
        if isinstance(self.metadata, basemetadata.BaseMetadata):
            return self.metadata.scaling_factor()

    def stim_duration(self):
        if isinstance(self.metadata, basemetadata.BaseMetadata):
            return self.metadata.stim_duration()

    def num_stims(self):
        return self.metadata.num_stims()

    def num_trials(self):
        return self.metadata.num_trials()

    @staticmethod
    def load_file(filename):
        filehandler = open(filename, 'rb')
        return pickle.load(filehandler)

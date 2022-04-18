from distutils.util import strtobool
import os
import warnings
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import json
import numpy as np
import logging
load_dotenv(find_dotenv())


class BaseMetadata(object):
    """Base metadata class.

    Base metadata class handling stimulus information and references to experiment data locations.

    Attributes:
        stim (defaultdict): Defaultdict for stimulus information.
        expt (dict): Dictionary of file path and time series label.
    """
    __slots__ = ['stim', 'expt']

    def __init__(self, path=None, expt_id=None, **kwargs):
        self.stim = defaultdict(None)
        self.expt = {}
        self.expt['path'] = path
        self.expt['expt_id'] = expt_id

        if 'expt' in kwargs:
            for key, val in kwargs['expt'].items():
                self.expt[key] = val

    def __str__(self):
        str_ret = f'expt: {self.expt}{os.linesep}'
        str_ret = str_ret + f'stim: {self.stim}'
        return str_ret

    def load_stims(self, override_py_file: str = None, override_trigger_file: str = None):
        """Load stimulus definitions and triggers.

        Loads the stimulus definitions and triggers from the default paths specified in the user configuration.
        Stimulus definitions are usually python files and are parsed based on the stim_def.json file. Stim triggers
        should be a text file of  following the format "<code> <onset> <code> <onset> ..."

        Args:
            override_py_file (str, optional): Defaults to None. Overrides the path to stimulus definition.
            override_trigger_file (str, optional): Defaults to None. Overrides the path to the stim trigger file.
        """

        if override_py_file == None:
            pth = Path(self.expt['path'], self.expt['expt_id'])
            stimfiles = pth.glob('*.py')
        else:
            stimfiles = Path(override_py_file)

        self.stim['files'] = [str(f) for f in stimfiles]

        self._parse_stims()

        if override_trigger_file == None:
            pth = Path(self.expt['path'], self.expt['expt_id'])
            stim_trigger_files = list(pth.glob('stimontimes.txt'))
        else:
            stim_trigger_files = list(Path(override_trigger_file))
        if len(stim_trigger_files) > 0:
            stim_file = open(stim_trigger_files[0], 'r')
            stim_file_contents = stim_file.readlines()[0].rstrip()
            if stim_file_contents == '':
                stim_file_contents = np.array([])
            else:
                stim_file_contents = np.array(stim_file_contents.split()).astype(np.float)
            self.stim['triggers'] = np.empty((int(len(stim_file_contents)/2),), dtype=[('id', 'i4'), ('time', 'f8')])
            for idx in range(self.stim['triggers'].shape[0]):
                self.stim['triggers'][idx]['id'] = stim_file_contents[2*idx]
                self.stim['triggers'][idx]['time'] = stim_file_contents[2*idx+1]

        else:
            self.stim['triggers'] = np.empty((0,), dtype=[('id', 'i4'), ('time', 'f8')])

        if self.stim['triggers']['id'].shape[0] > 0:
            if np.max(self.stim['triggers']['id']) > 1:
                self.stim['triggers'] = np.array([x for x in self.stim['triggers'] if x['id'] > 0])

            extra_trials = len(self.stim['triggers']) % int(len(np.unique(self.stim['triggers']['id'])))
            if extra_trials > 0:
                logging.warning('%s: dropping %i extra trials',
                                self.expt['path'],
                                extra_trials)
                self.stim['triggers'] = self.stim['triggers'][0:-extra_trials]
            self.stim['numTrials'] = int(np.floor(len(self.stim['triggers'])/self.num_stims()))

    def update_path(self, path):

        logging.info('Changing path from %s to %s', self.expt["path"], path)
        self.expt['path'] = path
        self.load_stims()
    # Accessor Methods

    def get_path(self):
        return self.expt['path']

    def get_expt_params(self, field: str, **kwargs):
        subtype = kwargs.get('subtype', None)

        if subtype is None and field in self.expt.keys():
            return self.expt[field]
        elif subtype is 'stim':
            return self.stim[field]
        return None

    def drop_unrecorded_trials(self, last_trigger_time):
        if self.stim['triggers']['id'].shape[0] > 0:
            trigger_times = self.stim['triggers']['time']
            last_valid_trial = int(np.argwhere((last_trigger_time - trigger_times) > 0)[-1]+1)
            self.stim['triggers'] = self.stim['triggers'][:last_valid_trial]

            extra_trials = len(self.stim['triggers']) % int(len(np.unique(self.stim['triggers']['id'])))
            if extra_trials > 0:
                logging.warning('%s: dropping %i extra trials',
                                self.expt['path'],
                                extra_trials)
                self.stim['triggers'] = self.stim['triggers'][0:-extra_trials]
            self.stim['numTrials'] = int(np.floor(len(self.stim['triggers'])/self.num_stims()))

    def do_blank(self)->bool:
        return strtobool(self.stim['doBlank']) if 'doBlank' in self.stim else False

    def num_stims(self)->int:
        """Return the number of unique stimuli.

        Returns:
            int: Number of stimuli.
        """

        return len(np.unique(self.stim['triggers']['id']))

    def num_trials(self)->int:
        """Return the number of trials.

        Returns the number of trials runs. Assumes that an even number of trials have been run for each stimulus.

        Returns:
            int: Number of trials
        """

        return int(np.floor(len(self.stim['triggers'])/self.num_stims()))

    def stim_duration(self)->float:
        """Return stimulus duration

        Returns the duration of the stimulation. Field must be specified as stimDuration.

        Returns:
            float: Stimulus duration
        """

        if 'stimDuration' in self.stim.keys():
            return float(self.stim['stimDuration'])
        else:
            return np.nan

    def stim_type(self)->str:
        """Return the stimulus type.

        Return the type of stimulus based on the file containing stimulus information.

        Returns:
            str: Stimulus type
        """

        return self.stim['type'] if 'type' in self.stim.keys() else None

    def _parse_stims(self):
        """Parse stimulus file.

        Parses stimulus file based on the information found in stim_def.json.
        """

        stimdefs = self._load_stim_defs()['psychoPy']['stims']
        for f in self.stim['files']:
            stim_type = Path(f).name.split('.')[0]
            if stim_type == 'serialTriggerDaqOut':
                continue
            else:
                self.stim['type'] = stim_type
                if stim_type not in stimdefs.keys():
                    continue
                else:
                    file_contents = open(Path(f)).read().replace(' ', '').replace(';', '').split('\n')
                    if 'fields' in stimdefs[stim_type].keys():
                        for key in stimdefs[stim_type]['fields']:
                            matched_lines = [x for x in file_contents if key+'=' in x]
                            if len(matched_lines) > 0:
                                self.stim[key] = matched_lines[0].split('=')[1].split('#')[0]
                            else:
                                self.stim[key] = ''

    def _load_stim_defs(self):
        return json.load(open(os.getenv('STIM_DEFINITIONS'), 'r'))['stimMetadata']

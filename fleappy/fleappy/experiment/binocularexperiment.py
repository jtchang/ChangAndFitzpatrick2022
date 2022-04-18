from os import linesep
from fleappy.experiment.baseexperiment import BaseExperiment
from fleappy.analysis.binocular import BinocularAnalysis
from fleappy.experiment.baselinefunctions import percentile_filter


class BinocularExperiment(BaseExperiment):

    __slots__ = ['binoc', 'contra', 'ipsi']

    def __init__(self, binoc=None, contra=None, ipsi=None, **kwargs):

        super().__init__(**kwargs)

        if isinstance(binoc, BaseExperiment) or binoc is None:
            self.binoc = binoc
        else:
            TypeError('Binoc must be an experiment or None!')
        if isinstance(binoc, BaseExperiment) or ipsi is None:
            self.ipsi = ipsi
        else:
            TypeError('Ipsi must be an experiment or None!')
        if isinstance(binoc, BaseExperiment) or contra is None:
            self.contra = contra
        else:
            TypeError('Contra must be an experiment or None!')

        if kwargs.get('animal_id', None) is None:
            self.animal_id = self.binoc.animal_id

    def __str__(self):
        str_ret = f'{self.animal_id} - {self.__class__.__name__}: {linesep}'

        for eye in ['binoc', 'ipsi', 'contra']:
            if getattr(self, eye) is not None:
                eye_string = getattr(self, eye).__str__(short=True)
            else:
                eye_string = f'None {linesep}'
            str_ret = str_ret + f'{eye} - {eye_string}'

        return str_ret

    def baseline_roi(self, field: str, target_field: str, **kwargs):
        for eye in self._eyes():
            getattr(self, eye).baseline_roi(field, target_field, **kwargs)

    def compute_dff(field, baseline, target_field, **kwargs):
        for eye in self._eyes():
            getattr(self, eye).compute_dff(field, baseline, target_field, **kwargs)

    def add_analysis(self, field: str, **kwargs):
        for eye in self._eyes():
            getattr(self, eye).add_analysis(field, **kwargs)

        analysis = BinocularAnalysis(self, field, **kwargs)
        if analysis is not None:
            self.analysis[analysis.id] = analysis

    def update_path(self, path: str):
        super().update_path(path)
        for eye in self._eyes():
            getattr(self, eye).update_path(path)

    # Accessor Methods

    def get_path(self):
        eye = self._eyes()[0]
        return getattr(self, eye).get_path()

    def get_expt_parameter(self, field, eye='binoc'):
        return getattr(self, eye).get_expt_parameter(field)

    def num_roi(self):
        return self.binoc.num_roi()

    def roi_positions(self, **kwargs):
        eye = self._eyes()[0]
        return getattr(self, eye).roi_positions(**kwargs)

    def distance_matrix(self, **kwargs):
        eye = self._eyes()[0]
        return getattr(self, eye).distance_matrix(**kwargs)

    def _eyes(self):
        eyes = []

        for eye in ['binoc', 'ipsi', 'contra']:
            if getattr(self, eye) is not None:
                eyes = eyes + [eye]
        return eyes

from fleappy.analysis.basegroup import BaseAnalysisGroup
import numpy as np


class EpiBinocular(BaseAnalysisGroup):

    def __init__(self, analysis_list, eye_order):
        super().__init__(analysis_list)

        if 'c' in eye_order and 'i' in eye_order:
            self.eye_order = eye_order
        else:
            raise TypeError('Needs at least a contra and ipsi analysis')

    def compute_ocular_dominance(self, cache=True, dff=False, prestim_bl=False):

        peak_responses = {}

        for analysis_idx, analysis in enumerate(self.analysis_list):
            if self.eye_order[analysis_idx] == 'binoc':
                continue
            avg_responses = analysis.stimulus_avg_responses(dff=dff, prestim_bl=True, cache=cache)
            peak_responses[self.eye_order[analysis_idx]] = np.max(avg_responses, axis=0)

        od_matrix = (peak_responses['c']-peak_responses['i']) / (peak_responses['c']+peak_responses['i'])
        if cache:
            self.cache['od_matrix'] = od_matrix
        return od_matrix

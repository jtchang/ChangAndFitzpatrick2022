import logging
import numpy as np
import pandas as pd
from typing import Union
from fleappy.analysis.base import BaseAnalysis


class TwoPhotonAnalysis(BaseAnalysis):
    """A metaclass for analysis of experiments.

    This is a base class for experiment analysis that handles associating analyses with an experiment, an id, and the
    associated data field.

    TODO:
    * Move Cache to use Shelve so that memory overhead is limited

    Attributes:
        expt (fleappy.experiment.BaseExperiment): Experiment analysis is associated with
        metrics(pd.DataFrame): Dataframe of computed metrics (each row corresponds to an roi)
    """
    __slots__ = ['metrics']

    def __init__(self, expt, field: str, **kwargs):

        super().__init__(expt, field, **kwargs)
        self.metrics = pd.DataFrame()
        self.metrics['id'] = [r.name for r in self.expt.get_roi()]

    def run(self, **kwargs) -> None:
        """Run all associated analyses
        """
        super().run(**kwargs)

    def distance_matrix(self, **kwargs) -> np.ndarray:
        """Returns Matrix of cellular pairwise distance.

        Uses the TwoPhotonExperiment method.

        Returns:
            numpy.ndarray[float]: NxN matrix of pairwise distance
        """
        return self.expt.distance_matrix(**kwargs)

    def clear_cache(self, key: str = None):
        """Clear Cache.

        Delete key from cache or delete all values of key is not specified.

        Args:
            key (str, optional): key. Defaults to None.
        """
        super().clear_cache(key)

        if key in self.metrics.columns:
            logging.info('Cleaning %s from metrics', key)
            del self.metrics[key]

    def format_scatter_plot(self, ax, **kwargs):

        x, y = self.expt.pixel_frame_size()
        scaling_factor = self.expt.scaling_factor()
        kwargs['fov'] = kwargs.get('fov', ((0, x), (y, 0)))

        _ = ax.axis('square')
        _ = ax.set_ylim((kwargs['fov'][1][0] * scaling_factor, kwargs['fov'][1][1] * scaling_factor))
        _ = ax.set_xlim((kwargs['fov'][0][0] * scaling_factor, kwargs['fov'][0][1] * scaling_factor))

        if kwargs.get('scale_bar', False):
            _ = ax.plot([kwargs['fov'][0][1]*scaling_factor - 50, kwargs['fov'][0][1]*scaling_factor - 50 - kwargs['scale_bar']],
                        [kwargs['fov'][1][0]*scaling_factor - 50, kwargs['fov'][1][0]*scaling_factor - 50],
                        color='k', zorder=3)

        _ = ax.set_xticks([])
        _ = ax.set_yticks([])

        if kwargs.get('background', None) is not None:
            ax.set_facecolor(kwargs['background'])

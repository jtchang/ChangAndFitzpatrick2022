from itertools import combinations
import logging
import numpy as np
import warnings
from typing import Any
from fleappy.analysis.base import BaseAnalysis
from fleappy.analysis.blockwise import BlockwiseAnalysis
from fleappy.analysis.orientation import OrientationAnalysis
# from fleappy.experiment.binocularexperiment import BinocularExperiment
from fleappy.experiment.tpwrapper import TPWrapper


class BinocularAnalysis(BaseAnalysis):

    __slots__ = ['odi', 'mismatch']

    def __init__(self, expt, field: str, **kwargs):
        # if not isinstance(expt, BinocularExperiment):
            # TypeError('Experiment must be a BinocularExperiment!')
        super().__init__(expt, field, **kwargs)
        self.odi = np.full((expt.num_roi(), ), np.nan)
        self.mismatch = np.full((expt.num_roi(), 3), np.nan)

    def run(self, **kwargs):
        kwargs['cocktail'] = kwargs['cocktail'] if 'cocktail' in kwargs else True
        super().run(**kwargs)
        self.compute_odi(**kwargs)
        self.compute_mismatch(**kwargs)

    def compute_odi(self, cocktail: bool = True, **kwargs) -> None:
        """Compute ocular dominance of responses.

        Ocular dominance index calculated as (C - I) / (C + I).

        Args:
            cocktail (bool, optional): Use cocktail response. Otherwise use peak responses Defaults to True.

        Raises:
            ValueError: Missing a monocular
            ValueError: [description]
            ValueError: Missing a blockwise analysis
        """
        if self.expt.contra is None or self.expt.ipsi is None:
            logging.warning('One of the monocular experiments is NONE!')
            return None

        for eye in ['contra', 'ipsi']:
            if not isinstance(getattr(self.expt, eye), TPWrapper):
                logging.warning('%s is not a valid experiment is of type %s',
                                eye,
                                type(getattr(self.expt, eye)))
                return None
            elif self.id not in getattr(self.expt, eye).analysis:
                logging.warning('%s is not an analysis for %s', self.id, eye)
            elif not isinstance(getattr(self.expt, eye).analysis[self.id], BlockwiseAnalysis):
                logging.warning('Need a blockwise analysis for %s', eye)
                return None

        if self.id in self.expt.contra.analysis and self.id in self.expt.ipsi.analysis:
            field_name = self.id
        else:
            logging.error('Unknown field name %s/%s', self.id, self.field)

        contra_responses = np.median(
            self.expt.contra.analysis[field_name].single_trial_responses(), axis=2)
        ipsi_responses = np.median(
            self.expt.ipsi.analysis[field_name].single_trial_responses(), axis=2)

        if cocktail:
            contra_responses = np.sum(contra_responses, axis=1)
            ipsi_responses = np.sum(ipsi_responses, axis=1)
        else:
            contra_responses = np.max(contra_responses, axis=1)
            ipsi_responses = np.sum(ipsi_responses, axis=1)
        contra_responses[contra_responses < 0] = 0
        ipsi_responses[ipsi_responses < 0] = 0

        self.odi = (contra_responses - ipsi_responses) / \
            (contra_responses + ipsi_responses)

    def compute_mismatch(self, **kwargs) -> None:
        """Calculate angle difference between all eyes.

        Assign orientation preference mismatches to  (Binoc-Ipsi, Binoc-Contra, Ipsi-contra) to mismatch.

        Raises:
            Warning: Warns if any of the eyes are
        """
        kwargs['override'] = kwargs.get('override', False)

        if not kwargs['override'] and not np.all(np.isnan(self.mismatch)):
            logging.info('%s: Skipping recomputing mismatches', self.expt.animal_id)
            return None
        for eye in ['binoc', 'contra', 'ipsi']:
            if not isinstance(getattr(self.expt, eye), TPWrapper):
                warnings.warn('%s is not a valid experiment is of type %s' %
                              (eye, type(getattr(self.expt, eye))))
            elif not isinstance(getattr(self.expt, eye).analysis[self.id], OrientationAnalysis):
                warnings.warn('%s is not a valid experiment is of type %s',
                              (eye, type(getattr(self.expt, eye).analysis[self.id])))

        for comp_idx, (eye_a, eye_b) in enumerate(combinations(['binoc', 'ipsi', 'contra'], 2)):
            if getattr(self.expt, eye_a) is None or getattr(self.expt, eye_b) is None:
                continue

            theta_a = getattr(
                self.expt, eye_a).analysis[self.id].get_orientation_preferences(**kwargs)
            theta_b = getattr(
                self.expt, eye_b).analysis[self.id].get_orientation_preferences(**kwargs)

            self.mismatch[:, comp_idx] = OrientationAnalysis.angle_diff(
                theta_a, theta_b, orcorrection=2)

    def scatter_plot_metric(self, metric, ax,  **kwargs):

        scaling_factor = self.expt.binoc.scaling_factor()
        x, y = self.expt.binoc.pixel_frame_size() * scaling_factor
        kwargs['s'] = kwargs['s'] if 's' in kwargs else 20
        if 'fov' in kwargs:
            fov = kwargs['fov']
            del kwargs['fov']
        else:
            fov = ((0, x), (y, 0))
        positions = self.expt.binoc.roi_positions(include_z=False)
        metric_vals = getattr(self, metric)

        if 'vmin' not in kwargs:
            kwargs['vmin'] = np.nanmin(metric_vals)
        if 'vmax' not in kwargs:
            kwargs['vmax'] = np.nanmax(metric_vals)

        ax.scatter(positions[:, 0],
                   positions[:, 1],
                   c=metric_vals,
                   **kwargs)

        _ = ax.set_xticks([])
        _ = ax.set_yticks([])
        _ = ax.axis('square')
        _ = ax.set_ylim(fov[1])
        _ = ax.set_xlim(fov[0])

    def binocular_responsive(self, eyes=('binoc', 'ipsi', 'contra')):
        responsive = np.ones((self.expt.num_roi(), ), dtype=bool)
        for eye in eyes:
            if getattr(self.expt, eye) is not None:
                responsive = np.logical_and(responsive, getattr(self.expt, eye).analysis[self.id].responsive())
            else:
                return np.zeros((self.expt.num_roi(), ), dtype=bool)
        return responsive

    def well_fit(self, rsq_threshold=0.6, si_threshold=None, eyes=None, orientation=True, pref_type='dr_fit'):
        well_fit_cells = np.ones((self.expt.num_roi(), ), dtype=bool)
        eyes = self.expt._eyes() if eyes is None else eyes

        for eye in eyes:
            if getattr(self.expt, eye) is not None:
                eye_fits = getattr(self.expt, eye).analysis[self.id].well_fit(
                    rsq_threshold=rsq_threshold, si_threshold=si_threshold, orientation=orientation, pref_type=pref_type)
                well_fit_cells = np.logical_and(well_fit_cells, eye_fits)
        return well_fit_cells

    def __getattribute__(self, name: str) -> Any:

        if name == 'bcdiff':
            return self.mismatch[:, 1]
        if name == 'bidiff':
            return self.mismatch[:, 0]
        if name == 'icdiff':
            return self.mismatch[:, 2]
        if name == 'cidiff':
            return -1 * self.mismatch[:, 2]

        if name == '|binoc-contra|':
            return np.abs(self.mismatch[:, 1])
        if name == '|binoc-ipsi|':
            return np.abs(self.mismatch[:, 0])
        if name == '|contra-ipsi|' or name == '|ipsi-contra|':
            return np.abs(self.mismatch[:, 2])

        if name == 'monocularity':
            return np.abs(self.odi)
        return object.__getattribute__(self, name)

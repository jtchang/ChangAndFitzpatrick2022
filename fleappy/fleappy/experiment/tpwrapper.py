import logging
import numpy as np

from fleappy.experiment.baseexperiment import BaseExperiment
from fleappy.analysis.orientation import OrientationAnalysis
from fleappy.analysis.spontaneous import SpontaneousAnalysis


class TPWrapper(BaseExperiment):
    """A Wrapper class for all two photon experiments (single plane or piezo imaging)

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_analysis(self, field: str, **kwargs):
        """Add analysis object to experiment.

        Args:
            field (str): time series field to use for analysis.
        """
        analysis = None
        if self.stim_type() == 'driftingGrating':
            analysis = OrientationAnalysis(
                self, field, **kwargs)
            logging.info('Creating Analysis %s', analysis.id)
        elif self.stim_type() == 'blackScreen':
            analysis = SpontaneousAnalysis(self, field, **kwargs)
            logging.info('Creating Analysis %s', analysis.id)
        if analysis is not None:
            analysis.run()
            self.analysis[analysis.id] = analysis

    def distance_matrix(self, include_z: bool = False, shuffle=False) -> np.array:
        """Pairwise distance matrix for all cells

        Returns:
            np.array: pairwise distance matrix (num cells x num cells)
        """
        positions = self.roi_positions(include_z=include_z)

        if shuffle:
            indices = np.arange(positions.shape[0])
            np.random.shuffle(indices)
            positions = positions[indices, :]
        dist_matrix = np.empty((positions.shape[0], positions.shape[0]))
        for idx_a, pos_a in enumerate(positions):
            for idx_b, pos_b in enumerate(positions):
                dist_matrix[idx_a, idx_b] = np.linalg.norm(pos_a-pos_b)
        return dist_matrix

    def roi_positions(self, include_z: bool = False) -> np.ndarray:
        positions = np.stack(self.map_to_roi(lambda x: x.centroid()))*self.scaling_factor()
        if include_z:
            positions = np.pad(positions, ((0, 0), 0, 1), mode='constant', constant_value=0)
        return positions

    def num_roi(self, **kwargs):
        NotImplementedError('num_roi is unknown!')

    # tif functions

    def get_trial_image(self, **kwargs):

        prepad = kwargs.get('prepad', 0)
        postpad = kwargs.get('postpad', 0)
        frame_numbers, _ = self.get_frame_indices(prepad=prepad, postpad=postpad)
        y, x = self.metadata.pixel_frame_size()
        trial_images = np.empty((frame_numbers.shape[0],
                                 frame_numbers.shape[1],
                                 frame_numbers.shape[2],
                                 y,
                                 x), dtype=np.uint16)

        frame_ends = None
        tif_files = None

        frame_numbers = np.transpose(frame_numbers, [1, 0, 2])

        for trial_idx, trial_data in enumerate(frame_numbers):
            logging.info('[%s]: Processing Trial %s', self.animal_id, trial_idx)
            for stim_idx, frames in enumerate(trial_data):
                img_data, frame_ends, tif_files = self.get_frames(frames,
                                                                  frame_ends=frame_ends,
                                                                  tif_files=tif_files)
                trial_images[stim_idx, trial_idx, :, :] = np.transpose(img_data, axes=[2, 0, 1])
        return trial_images

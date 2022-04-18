import logging
import numpy as np
from matplotlib.axes import Axes
import fleappy.experiment.tpexperiment as tpexperiment
import fleappy.analysis.dendorientation as dor_analysis
from fleappy.roimanager import imagejroi, roi
from skimage.draw import polygon


class DendriteExperiment(tpexperiment.TPExperiment):

    def load_roi(self, **kwargs):
        """Load associated roi.

        Args:
            slice_id (int, optional): Which imaging slice to use. Defaults to 1.
        """

        n_points = kwargs.get('n_points', 10)
        width = kwargs.get('width', 5)

        slice_id = 'slice'+str(self.metadata.slice_id())
        zip_path = self._zip_path(slice_id=slice_id)

        self.roi = []

        cells = 0
        branches = 0
        neuropil = 0
        for idx, ij_roi in enumerate(imagejroi.read_roi_zip(zip_path).values()):
            roi_type = 'unknown'
            if ij_roi['type'] in ['oval', 'polygon', 'freehand', 'rectangle']:
                if 'neuropil' in ij_roi['name']:
                    ij_roi['name'] = f'Neuropil_{neuropil}'
                    neuropil = neuropil+1
                    roi_type = 'neuropil'
                else:
                    ij_roi['name'] = f'Cell_{cells}'
                    cells = cells+1
                    roi_type = 'cell'
            elif ij_roi['type'] in ['line', 'polyline', 'freeline']:
                ij_roi['name'] = f'Branch_{branches}'
                branches = branches+1
                roi_type = 'dendrite'

            self.roi.append(roi.Roi(idx,
                                    roi_type=roi_type,
                                    mask=imagejroi.to_array(ij_roi),
                                    name=ij_roi['name'],
                                    ij_roi=ij_roi)
                            )

        for roi_val in self.roi:
            if roi_val.type == 'dendrite':
                total_roi = len(self.roi)
                masks = self._generate_branch_segments(roi_val.points(), width=width, n_points=n_points)
                for idx, mask in enumerate(masks):
                    self.roi.append(roi.Roi(total_roi+idx,
                                            roi_type='dendrite_segment',
                                            name=f'{roi_val.name}_s{idx}',
                                            mask=mask))

    def num_roi(self, **kwargs):
        if 'type' not in kwargs:
            return len(self.roi)
        return len([roi for roi in self.roi if roi.type == kwargs['type']])

    def add_analysis(self, field: str, **kwargs):

        analysis = None
        if self.stim_type() == 'driftingGrating':
            analysis = dor_analysis.DendOrientationAnalysis(
                self, field, **kwargs)
            logging.info('Creating Analysis %s', analysis.id)

        if analysis is not None:
            analysis.run()
            self.analysis[analysis.id] = analysis

    def color_roi(self, ax: Axes = None):
        np.random.seed(1)
        x, y = self.pixel_frame_size()
        full_frame = np.zeros((x, y, 4))
        for roi_idx, roi in enumerate(self.roi):
            if roi.type == 'dendrite':
                continue
            full_frame[roi.mask.todense(), :] = [np.random.random(), np.random.random(), np.random.random(), 1]

        ax.imshow(full_frame)

    def num_branches(self):
        return len([x for x in self.roi if x.type == 'dendrite'])

    def branch_roi(self, branch_num: str):
        return np.array([idx for idx, roi in enumerate(self.roi) if roi.branch() == str(branch_num)], dtype=int)

    def process(self):
        logging.info('[%s]: Loading ROI...', self.animal_id)
        self.load_roi()
        self.load_ts_data(shift_zero=True)
        logging.info('[%s]: Baseling time series data...', self.animal_id)
        self.baseline_roi('rawF', 'baseline')
        logging.info('[%s]: Computing DF/F...', self.animal_id)
        self.compute_dff('rawF', 'baseline', 'dff', clip_zero=False)

    @staticmethod
    def roi_types():
        return ['dendrite', 'dendrite_segment', 'primary', 'cell']

    @staticmethod
    def _generate_branch_segments(xy_points, width=5, n_points=10):

        bb_upper = np.full((xy_points.shape[0]-1, 2), np.nan, dtype=int)
        bb_lower = np.full((xy_points.shape[0]-1, 2), np.nan, dtype=int)

        for idx, (start_pt, end_pt) in enumerate(zip(xy_points[:-1, :], xy_points[1:, :])):
            normal_vec = start_pt-end_pt

            cw_vec = np.matmul(normal_vec, np.array([[0, 1], [-1, 0]]))
            cw_angle = np.angle(cw_vec[0] + 1j*cw_vec[1])
            cw_scaled_vec = xy_points[idx, :] + width * np.array([np.cos(cw_angle), np.sin(cw_angle)])
            ccw_scaled_vec = xy_points[idx, :] + width * np.array([np.cos(np.pi+cw_angle), np.sin(np.pi+cw_angle)])

            if idx == 0 or np.linalg.norm(bb_upper[idx-1, :]-cw_scaled_vec) < np.linalg.norm(bb_lower[idx-1, :]-cw_scaled_vec):
                bb_upper[idx, :] = np.round(cw_scaled_vec)
                bb_lower[idx, :] = np.round(ccw_scaled_vec)
            else:
                bb_upper[idx, :] = np.round(ccw_scaled_vec)
                bb_lower[idx, :] = np.round(cw_scaled_vec)

        mask = np.zeros((bb_upper.shape[0]//n_points, 512, 512), dtype=bool)

        for idx in range(1, bb_upper.shape[0]-n_points, n_points):
            rr, cc = polygon([bb_lower[idx, 0], bb_upper[idx, 0], bb_upper[idx+n_points, 0], bb_lower[idx+n_points, 0]],
                             [bb_lower[idx, 1], bb_upper[idx, 1], bb_upper[idx+n_points, 1], bb_lower[idx+n_points, 1]])
            mask[idx//n_points, rr, cc] = True

        return mask

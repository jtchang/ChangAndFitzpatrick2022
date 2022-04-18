import numpy as np

from scipy.ndimage.measurements import center_of_mass
from fleappy.analysis.base import BaseAnalysis
from fleappy.filter.fermi import default_filter_params


class EpiAnalysis(BaseAnalysis):

    __slots__ = ['filter_params']

    def __init__(self, expt, field: str, **kwargs):
        super().__init__(expt, field, **kwargs)

        # filter parameters to be used for analysis
        filter_defaults = default_filter_params()
        filter_params = kwargs.pop('filter_params', filter_defaults)
        for key, val in filter_defaults.items():
            if key not in filter_params:
                filter_params[key] = val

        self.filter_params = filter_params
        _ = self.roi_centroid()

    def roi(self, cache=True):
        if cache and 'roi' in self.cache:
            return self.cache['roi']
        roi = self.expt.roi(self.field)

        if cache:
            self.cache['roi'] = roi

        return roi

    def roi_centroid(self, cache=True):
        if cache and 'roi_centroid' in self.cache:
            return self.cache['roi_centroid']
        centroid = tuple(np.round(center_of_mass(self.roi())).astype(int))
        if cache:
            self.cache['roi_centroid'] = centroid

        return centroid

    def resolution(self):
        """Get the resolution of the analysis data.

        Returns:
            float: resolution in microns /pixel
        """
        if 'resolution' not in self.cache:
            self.cache['resolution'] = self.expt.resolution(self.field)
        return self.cache['resolution']

    def pixel_frame_size(self):

        return self.expt.pixel_frame_size(self.field)

    def grid_roi(self, spacing: float = 0.15):
        """Get a equally spaced grid that overlaps with the roi

        Args:
            spacing ([float]): Distance in millimeters between points

        Returns:
            [type]: [description]
        """
        frame_size = self.pixel_frame_size()
        resolution = self.resolution()[0] * 1e-3  # mm/pixel
        roi = self.roi()
        pixel_spacing = np.round(spacing/resolution).astype(int)
        pixel_spacing = 1 if pixel_spacing == 0 else pixel_spacing

        xv, yv = np.round(np.meshgrid(np.arange(0, frame_size[0], pixel_spacing), np.arange(
            0, frame_size[1], pixel_spacing))).astype(int)

        spaced_grid = np.zeros(roi.shape, dtype=bool)
        spaced_grid[xv, yv] = True

        seed_pos_list = np.argwhere(np.logical_and(roi, spaced_grid))

        return seed_pos_list

    def _roi_bounding(self):
        pts = np.where(self.roi())
        return np.min(pts[0]), np.max(pts[0])+1, np.min(pts[1]), np.max(pts[1])+1

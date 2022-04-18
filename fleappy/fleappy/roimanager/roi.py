import os
import numpy as np
import fleappy.roimanager.nproi
from scipy.sparse import csr_matrix
from . import imagejroi


class Roi(object):
    """ROI Class for handling cellular/subcellular ROI

    Attributes:
        id (str): Identifying string
        name (str): Name of roi
        type (str): Type of roi
        ts_data (dict): Dictionary of time series data.
        mask (scipy.sparse.csr.csr_matrix): Scipy sparse matrix with the associated ROI mask.

    """

    __slots__ = ['id', 'type', 'mask', 'ts_data', 'name', 'ij_roi']

    def __init__(self, roi_id: str = None, roi_type: str = None, mask: csr_matrix = None, name=None, ij_roi=None):
        self.id = roi_id
        self.name = name
        if roi_type not in self.allowed_types():
            raise TypeError('%s:%s is not an allowed roi type!' % (name, roi_type))
        self.type = roi_type
        self.mask = csr_matrix(mask) if isinstance(mask, np.ndarray) else mask
        self.ts_data = {}
        self.ij_roi = ij_roi

    def __str__(self):
        ret_str = f'fleappy ROI object: {os.linesep}'
        ret_str = ret_str + f'id: {self.id}{os.linesep}'
        ret_str = ret_str + f'name: {self.name}{os.linesep}'
        ret_str = ret_str + f'type: {self.type}{os.linesep}'
        ret_str = ret_str + f'mask: Mask {self.mask.shape}{os.linesep}'
        ret_str = ret_str + f'ts_data: {list(self.ts_data.keys())}{os.linesep}'
        return ret_str

    def __eq__(self, other):
        if (self.mask.todense() != other.mask.todense()).any():
            return False
        return True

    def centroid(self)->tuple:
        """Returns the centroid of the roi

        Returns:
            tuple: (y,x) of centroid
        """

        return fleappy.roimanager.nproi.centroid(self.mask.toarray())

    def outline(self):
        pass

    def branch(self):
        """Get the branch associated with the ROI.

        Returns:
            [type]: [description]
        """

        if self.type == 'dendrite_segment':
            names = self.name.split('_')
            if len(names) > 2:
                return names[1]

        return None

    def points(self):
        if self.ij_roi is not None and self.ij_roi['type'] in ['freeline', 'line', 'polyline']:
            return imagejroi.line_points(self.ij_roi)
        else:
            return np.argwhere(self.mask.todense())

    @staticmethod
    def allowed_types():
        """ Types of allowed ROI.
        """

        return ['cell', 'dendrite', 'neuropil', 'dendrite_segment', 'primary']

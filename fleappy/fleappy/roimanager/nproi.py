import numpy as np
from scipy.ndimage.measurements import center_of_mass
import skimage.io as io
from pathlib import Path
from typing import Union
import logging


def centroid(roi)->tuple:
    """ Compute the centroid of an object.

    Given a 2-d array object, returns the centroid (x,y), using scipy.ndimage.measurements.center_of_mass.

    Args:
        roi (ndarray): 2-d array object

    Returns:
        tuple: (y,x) of centroid
    """

    return center_of_mass(roi)


def tseries_data(rois: np.ndarray, timedata: np.ndarray)->np.ndarray:
    """Compute time series data for rois over timedata.

    Args:
        rois (np.ndarray): Array of ROIs.
        timedata (np.ndarray): Array of imaging data.

    Returns:
        np.ndarray: Array of time series data (cell, time).
    """

    ts_data = np.empty((rois.shape[0], timedata.shape[0]))
    logging.debug(rois.shape)
    for roi_idx, roi in enumerate(rois):
        for t_index, frame in enumerate(timedata):
            ts_data[roi_idx, t_index] = np.mean(frame[roi])

    return ts_data


def load_from_file(filename: Union[str, Path])->np.ndarray:
    """"Loads roi from tif file.

    Args:
        filename (str or Path): tif file to load roi from. 

    Returns:
        numpy.ndarray: Array of roi masks (cells, y, x)
    """

    if isinstance(filename, str):
        filepath = Path(filename)
    elif isinstance(filename, Path):
        filepath = filename
    return io.imread(filepath, plugin='pil').astype(np.bool)

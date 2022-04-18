""" Collection of functions to handle conversions of ImageJ ROIs.

Depends heavily on the `read_roi <https://pypi.org/project/read-roi/>` library.

"""


import logging
from pathlib import Path
import pickle

from imageio import mimwrite
# from read_roi import read_roi_zip
from roifile import roiread
from matplotlib.path import Path as MplPath
import numpy as np
from skimage.draw import ellipse, line

DEFAULT_FRAME_SIZE = (512, 512)
"""tuple: Frame size in pixels to be used for generating roi masks."""

ROI_TYPES = {'POLYGON': 0,
             'RECT': 1,
             'OVAL': 2,
             'LINE': 3,
             'FREELINE': 4,
             'POLYLINE': 5,
             'NOROI': 6,
             'FREEHAND': 7,
             'TRACED': 8,
             'ANGLE': 9,
             'POINT': 10
             }


def line_points(roi):
    """Get all Points between vertices for the roi.

    Args:
        roi ([type]): [description]

    Returns:
        [type]: Returns (y,x) points connecting the vertices
    """

    vertices = roi.coordinates()

    pts = np.empty((0, 2), dtype=int)

    for (x_1, y_1), (x_2, y_2) in zip(vertices[:-1, :], vertices[1:, :]):
        logging.debug('line from (%i, %i) to (%i, %i)', x_1, y_1, x_2, y_2)

        y, x = line(y_1, x_1, y_2, x_2)

        pts = np.concatenate((pts,
                              np.vstack((y, x)).T),
                             axis=0)

    return pts


def to_array(roi, framesize: tuple = DEFAULT_FRAME_SIZE)->np.ndarray:
    """Convert ImageJ roi to numpy array mask.

    Args:
        roi (ImageJ ROI): ImageJ Roi
        framesize (tuple, optional): Defaults to DEFAULT_FRAME_SIZE. Frame size to use (y,x) should be type int.

    Returns:
        numpy.ndarray: [description]
    """

    mask = np.zeros(framesize, dtype=bool)
    if roi.roitype == ROI_TYPES['FREEHAND'] or roi.roitype == ROI_TYPES['POLYGON']:
        x_vertices = roi.coordinates()[:, 0]
        y_vertices = roi.coordinates()[:, 1]

        mask = np.zeros(framesize, dtype=bool)
        y_range = np.arange(roi.top, roi.bottom+1)
        x_range = np.arange(roi.left, roi.right+1)
        x, y = np.meshgrid(x_range, y_range)
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        pth = MplPath(list(zip(x_vertices, y_vertices)))
        grid = pth.contains_points(points)
        mask[roi.top:roi.bottom+1, roi.left:roi.right+1] = np.reshape(grid, (roi.bottom-roi.top+1,
                                                                             roi.left-roi.right+1))
    elif roi.roitype == ROI_TYPES['RECT']:
        mask[roi.top:roi.bottom+1, roi.left:roi.right+1] = True
    elif roi.roitype == ROI_TYPES['OVAL']:
        center_y = np.mean([roi.top, roi.bottom])
        center_x = np.mean([roi.left, roi.right])
        r_y = (roi.top-roi.bottom)/2
        r_x = (roi.right-roi.left)/2
        y, x = ellipse(center_y, center_x, r_y, r_x)
        mask[y, x] = True
    elif roi.roitype == ROI_TYPES['LINE']:
        y, x = line(int(roi.y1), int(roi.x1), int(roi.y2), int(roi.x2))
        mask[y, x] = True
    elif roi.roitype == ROI_TYPES['FREELINE'] or roi.roitype == ROI_TYPES['POLYLINE']:
        pts = line_points(roi)
        mask[pts[:, 0], pts[:, 1]] = True
    else:
        NotImplementedError('Roi of type %s are not handled yet!', roi.roitype)

    return mask


def to_stack(rois: list, framesize: tuple = DEFAULT_FRAME_SIZE):
    """Convert dictionary or ImageJ roi to numpy stack of masks.

    Args:
        rois (dict): ImageJ rois from zip file.
        framesize (tuple, optional): Defaults to DEFAULT_FRAME_SIZE. Frame size to use (y,x) should be type int.

    Returns:
        list, numpy.ndarray: list of roi names, numpy array of masks (# cell , y , x)
    """

    tiffstack = np.zeros((len(rois), framesize[0], framesize[1]), dtype=np.bool)
    names = []
    for idx, roi in enumerate(rois):
        logging.debug('Converting %i roi', idx)
        tiffstack[idx, :, :] = to_array(roi, framesize=framesize)

    names = [roi.name for roi in rois]
    return names, tiffstack


def zip_to_tif(filesource: str, filetarget: str, framesize: tuple = DEFAULT_FRAME_SIZE):
    """Open a .zip of imagej rois and write them to a tif file

    Opens imagej rois in \*.zip and writes them to a tif file. Currently does not save the ROI names.

    Args:
        filesource(str): File path for ImageJ rois as a zip file
        filetarget(str): File path to write tif stack of ROIS
        framesize(tuple, optional): Defaults to DEFAULT_FRAME_SIZE. [description]

    Returns:
        None

    TODO:
        * imageio tiff writing description seems to break- should write names of ROI as metadata for tiff
    """
    filesource = filesource if isinstance(filesource, str) else str(filesource.as_posix())
    filetarget = filetarget if isinstance(filetarget, str) else str(filetarget.as_posix())

    filesource = rf'{filesource}'
    filetarget = rf'{filetarget}'

    names, tiffstack = to_stack(roiread(Path(filesource)), framesize=framesize)

    mimwrite(Path(filetarget), tiffstack.astype(np.uint8))
    with open(filetarget+'.names', 'w') as f:
        f.write(';'.join(names))

    return None


def zip_to_single_array(filesource: str, framesize: tuple = DEFAULT_FRAME_SIZE):

    filesource = filesource if isinstance(filesource, str) else str(filesource.as_posix())
    filesource = rf'{filesource}'

    masks = np.zeros(framesize, dtype=bool)

    rois = roiread(filesource)

    for roi in rois:
        roi_array = to_array(roi, framesize=framesize)
        masks[roi_array] = True
    return masks


def zip_to_dict_pickle(filesource: str, filetarget: str, framesize: tuple = DEFAULT_FRAME_SIZE):

    filesource = filesource if isinstance(filesource, str) else str(filesource.as_posix())
    filetarget = filetarget if isinstance(filetarget, str) else str(filetarget.as_posix())

    filesource = rf'{filesource}'
    filetarget = rf'{filetarget}'

    names, tiffstack = to_stack(roiread(Path(filesource)), framesize=framesize)

    rois = {'rois': {},
            'framesize': framesize}
    for name, array in zip(names, tiffstack):
        y, x = np.where(array)
        rois['rois'][name] = {}
        rois['rois'][name]['x'] = x.tolist()
        rois['rois'][name]['y'] = y.tolist()

    with open(filetarget, 'wb') as outfile:
        pickle.dump(rois, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_dict_pickle(filesource: str):
    filesource = rf'{filesource}'

    with open(filesource, 'rb') as infile:
        rois = pickle.load(infile)

    roi_mat = np.zeros((len(rois['rois']), rois['framesize'][0], rois['framesize'][1]), dtype=bool)

    names = []

    for idx, (name, roi) in enumerate(rois['rois'].items()):
        names.append(name)
        roi_mat[idx, roi['y'], roi['x']] = True

    return names, roi_mat

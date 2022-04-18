from skimage.feature import match_template
from skimage.transform import resize
from scipy.signal import correlate2d
import numpy as np
from scipy.ndimage.interpolation import shift
from pathlib import Path
import logging


def register(avg_img, tiff_stack, options={}):
    """ Register tif stack using downsample registration

    Based on DownsampleReg (Theo Walker, 2014)

    Args:
        avg_img (numpy.ndarray): Template Image (y, x).
        tiff_stack (numpy.ndarray): Time series data to be registered (z, y, x).

    Returns:
        transform_spec (np array float (y,x): pixel shifts to register tiff_stack
    """
    offsets = np.empty((tiff_stack.shape[0], 2))

    if 'downsamplerates' not in options.keys():
        options['downsamplerates'] = [1/4, 1/2, 1]
    if 'maxmovement' not in options.keys():
        options['maxmovement'] = 1/16

    size_y, size_x = avg_img.shape
    max_movement = options['maxmovement']

    for r in range(len(options['downsamplerates'])):
        ds_rate = options['downsamplerates'][r]
        logging.debug('Running Downsample: %s', str(ds_rate))
        down_size_y = int(np.round(size_y * ds_rate))
        down_size_x = int(np.round(size_x * ds_rate))

        temp_img = resize(avg_img, (down_size_y, down_size_x))

        for f_idx, frame in enumerate(tiff_stack):
            reg_img = resize(frame, (down_size_y, down_size_x))
            best_corr_value = -1
            best_corr_x = 0
            best_corr_y = 0
            if r == 0:
                min_offset_y = int(-1 * np.round(max_movement * reg_img.shape[0]/2))
                max_offset_y = int(np.round(max_movement * reg_img.shape[0]/2))

                min_offset_x = int(-1 * np.round(max_movement * reg_img.shape[1]/2))
                max_offset_x = int(np.round(max_movement * reg_img.shape[1]/2))
            else:
                min_offset_y = int(offsets[f_idx, 0]*ds_rate - ds_rate/options['downsamplerates'][r-1]/2)
                max_offset_y = int(offsets[f_idx, 0]*ds_rate + ds_rate/options['downsamplerates'][r-1]/2)

                min_offset_x = int(offsets[f_idx, 1]*ds_rate - ds_rate/options['downsamplerates'][r-1]/2)
                max_offset_x = int(offsets[f_idx, 1]*ds_rate + ds_rate/options['downsamplerates'][r-1]/2)

            for y in range(min_offset_y, max_offset_y):
                for x in range(min_offset_x, max_offset_x):
                    sub_temp_y1 = np.max((y, 0))
                    sub_temp_y2 = reg_img.shape[0] + np.min((y, 0))
                    sub_temp_x1 = np.max((x, 0))
                    sub_temp_x2 = reg_img.shape[1] + np.min((x, 0))

                    sub_reg_y1 = np.max((-y, 0))
                    sub_reg_y2 = reg_img.shape[0] + np.min((-y, 0))
                    sub_reg_x1 = np.max((-x, 0))
                    sub_reg_x2 = reg_img.shape[1] + np.min((-x, 0))

                    sub_temp_img = temp_img[sub_temp_y1:sub_temp_y2, sub_temp_x1:sub_temp_x2]
                    sub_reg_img = reg_img[sub_reg_y1:sub_reg_y2, sub_reg_x1:sub_reg_x2]
                    corr_value = _corr2d(sub_temp_img, sub_reg_img)

                    if corr_value > best_corr_value:
                        best_corr_x = x
                        best_corr_y = y
                        best_corr_value = corr_value

            offsets[f_idx, 0] = best_corr_y * (1.0/ds_rate)
            offsets[f_idx, 1] = best_corr_x * (1.0/ds_rate)

    return offsets


def transform(img_stack, transform_spec):
    """Applies (y,x) translation to a series of images

    Args:
        img_stack (numpy array):  Uncorrected tiff stack (z, y, x)
        transform_spec (numpy array): Shifts to be applied to tiff stack in format (frameNum, (y,x))

    Returns:
        numpy.ndarray: Motion corrected tiff stack (z, y, x)
    """

    z, w, h = img_stack.shape

    for idx, frame in enumerate(img_stack):
        img_stack[idx, :, :] = shift(frame, transform_spec[idx, :])
    return img_stack.astype(np.int16)


def join(transform_list, transform_spec):
    """Appends the next set of transformations to a previous set.

    Args:
        transform_list (numpy.ndarray): Next set of frame by frame transformations.
        transform_spec (numpy.ndarray): Previous frame by frame transformations.

    Raises:
        ValueError: If the new transform_list isn't of a (n,2) numpy array.

    Returns:
        numpy.ndarray: Joined transformation specification.
    """

    if transform_list is None:
        return transform_spec
    elif isinstance(transform_list. np.ndarray) and transform_list.shape[1] == 2:
        return np.concatenate((transform_list, transform_spec), axis=0)
    else:
        raise ValueError('The transform list isn\'t a numpy array of dimensions (n,2)!')


def save(transform_list, target: Path):
    """Write the list of transformations to a file.

    Args:
        transform_list ([type]): [description]
        target (Path): [description]
    """

    np.savetxt(target, np.squeeze(transform_list), delimiter=',', fmt='%.3f', header=__name__)


def load():
    """Load frame by frame transformations from file.

    TODO:
        * Implement dftreg file loading.

    Raises:
        NotImplementedError: File loading is not yet supported
    """
    raise NotImplementedError('Transformation loading is not yet implemented.')


def create_template(img_stack):
    """Creates a template from the image stack.

    Args:
        img_stack (numpy.ndarray): image stack (t, y, x)

    Returns:
        numpy.ndarray: Mean Image 
    """

    return np.mean(img_stack, axis=0, dtype=np.float)


def _corr2d(template, img):

    am_bar = img - np.mean(img)
    bm_bar = template - np.mean(template)
    c_vect = am_bar * bm_bar
    d_vect = am_bar ** 2
    e_vect = bm_bar ** 2

    corr_val = np.sum(c_vect) / float(np.sqrt(np.sum(d_vect) * np.sum(e_vect)))
    return corr_val

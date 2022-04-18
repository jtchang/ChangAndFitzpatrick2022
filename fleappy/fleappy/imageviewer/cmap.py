import numpy as np
from skimage.color import rgb2hsv, hsv2rgb


def hsv_2d(huebins: int, satbins: int)->np.ndarray:
    LUT = np.ones((huebins, satbins, 3))
    for hue in range(LUT.shape[0]):
        LUT[hue, :, 0] = hue/LUT.shape[0]

    for saturation in range(LUT.shape[1]):
        LUT[:, saturation, 1] = saturation/LUT.shape[0]

    return LUT


def rgb_to_hsv(rgb_matrix):
    """[summary]

    Args:
        rgb_matrix ([type]): [description]

    Returns:
        [type]: [description]
    """

    return rgb2hsv(rgb_matrix)


def hsv_to_rgb(hsv_matrix):
    return hsv2rgb(hsv_matrix)


def rgb_to_hsl(rgb_matrix):

    if len(rgb_matrix.shape) is not 3:
        raise ValueError("Matrix must be of MxNx3 size")

    hsl_matrix = np.zeros(rgb_matrix.shape)
    return hsl_matrix


def _map_range(val, oldrange: tuple, newrange=(0, 1)):

    old_span = oldrange[1]-oldrange[0]
    new_span = newrange[1]-newrange[0]

    scaled_val = float(val-oldrange[0])/old_span

    return newrange[0] + (scaled_val * new_span)

from typing import Tuple, Dict
import numpy as np
from scipy.fft import fft2, fftshift, ifft2, next_fast_len, set_workers
import warnings


def kernel(array_size: Tuple[int, int], cutoffs: Tuple[float, float], resolution: float, t: float):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'overflow encountered')

        def kernel_func(rs, pb, x):
            return 1 / (1+np.exp(-((pb-np.sqrt(rs))/(x*pb))))

        x = np.arange(1-array_size[0]/2, array_size[0]/2+1, 1)
        y = np.arange(1-array_size[1]/2, array_size[1]/2+1, 1)
        y_v, x_v = np.meshgrid(y, x)

        distance_matrix = x_v**2+y_v**2
        scaled_array_size = (np.max(array_size) * resolution)

        filter_kernel = kernel_func(
            distance_matrix, scaled_array_size / cutoffs[0], t) if cutoffs[0] > 0 else np.ones(array_size)
        filter_kernel = filter_kernel-kernel_func(distance_matrix,
                                                  scaled_array_size/cutoffs[1], t) if cutoffs[1] > 0 else filter_kernel

    return filter_kernel


def filter(array, **kwargs):
    """Fermi bandpass filter.

    [description]

    Args:
        array ([type]): array of images to filter

    **kwargs    
        roi ([type], optional): Defaults to None. Region of interest to mask
        cutoffs (Tuple[float, float], optional): Defaults to (-1, -1). low and high filter cutoff in microns
        resolution (float, optional): Defaults to 1. Spatial Resolution in microns/pixel
        t (float, optional): Defaults to 0.05. Filter "Roll-off"
        filter_kernel ([type], optional): Defaults to None. Override filter kernel

    Returns:
        [type]: filtered frame or frame series
    """

    roi = kwargs['roi'] if 'roi' in kwargs else None
    cutoffs = kwargs['cutoffs'] if 'cutoffs' in kwargs else (-1, -1)
    resolution = kwargs['resolution'] if 'resolution' in kwargs else 1
    t = kwargs['t'] if 't' in kwargs else 0.05
    filter_kernel = kwargs['filter_kernel'] if 'filter_kernel' in kwargs else None

    pad_value = 1 if roi is None else 0

    if array.ndim == 2:
        roi = np.ones(array.shape, dtype=bool) if roi is None else roi

        # pad to square and fft
        array_start, padded_array = _pad_square(array, pad_value=pad_value)
        array_size = padded_array.shape
        with set_workers(4):
            if filter_kernel is None:
                filter_kernel = fftshift(kernel(array_size, cutoffs, resolution, t))
            filtered_array = ifft2(filter_kernel * fft2(padded_array))

        filtered_array = filtered_array[array_start[0]: array_start[0] + array.shape[0],
                                        array_start[1]: array_start[1] + array.shape[1]]
        filtered_array[~roi] = 0
    else:
        filtered_array = np.empty(array.shape)
        for idx, subarray in enumerate(array):
            filtered_array[idx, :] = filter(subarray, **kwargs)

    return np.real(filtered_array)


'''def _pad_square(a, pad_value=None):
    array_size = a.shape
    if array_size[0] == array_size[1]:
        return (0, 0), a
    elif array_size[0] > array_size[1]:
        prepad = int(np.floor((array_size[0]-array_size[1])/2))
        postpad = int(np.ceil((array_size[0]-array_size[1])/2))
        if pad_value == None:
            pad_value = np.nanmean(a[:])
        return (0, prepad), np.pad(a, ((0, 0), (prepad, postpad)), 'constant', constant_values=pad_value)
    else:
        idx, vector = _pad_square(np.transpose(a))
        return (idx[1], idx[0]), np.transpose(vector)'''


def _pad_square(img, pad_value=None):
    if pad_value == None:
        pad_value = np.nanmean(img)

    max_axis = np.max(img.shape)
    pad_size = next_fast_len(max_axis)

    pad_x = pad_size - img.shape[0]
    pad_y = pad_size - img.shape[1]

    img_pad = np.pad(img,
                     ((int(np.floor(pad_x/2)), int(np.ceil(pad_x/2))), (int(np.floor(pad_y/2)), int(np.ceil(pad_y/2)))),
                     mode='constant',
                     constant_values=pad_value)

    return (int(np.floor(pad_x/2)), int(np.floor(pad_y/2))), img_pad


def default_filter_params() -> Dict:
    """Default filtering parameters for epifluorescence imaging.

    Returns:
        Dict: [description]
    """
    return {'cutoffs': (-1, 1600),
            'resolution': 1000/172,
            't': 0.05,
            'roi': True}

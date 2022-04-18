import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.ndimage.interpolation import shift


def fix_scan_phase(frame, pixel_shift=None):
    new_frame = np.zeros(frame.shape)
    if pixel_shift is None:
        pixel_shift, _, _ = phase_cross_correlation(frame[::2, :], frame[1::2, :], upsample_factor=10)
        pixel_shift = pixel_shift[1]

    if pixel_shift == 0:
        return pixel_shift, frame

    new_frame[::2, :] = frame[::2, :]
    new_frame[1::2, :] = shift(frame[1::2], [0, pixel_shift])

    return pixel_shift, new_frame

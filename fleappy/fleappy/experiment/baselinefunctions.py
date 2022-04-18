import numpy as np
from scipy.ndimage.filters import percentile_filter as pct_filt


def percentile_filter(tseries, frame_rate: float = 1, percentile: float = 30, window_size: float = 60):
    """Filter times series using a rolling percentile filter

    Args:
        tseries (numpy.array): Time series to be filtered
        percentile (float): Percentile for cutoff
        window_size (float): Rolling window size (in seconds)

    Returns:
        numpy.array: filtered time series
    """
    window_elements = int(np.round(frame_rate * window_size))
    return pct_filt(tseries, percentile, size=(window_elements,))


def default_dff(signal, baseline):
    return (signal-baseline)/baseline

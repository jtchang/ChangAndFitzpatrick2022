import logging
import numpy as np
from scipy.signal import convolve2d, fftconvolve
from scipy.interpolate import interp1d, CubicSpline
from scipy.fftpack import fftshift, fft2, ifft2, next_fast_len
import scipy.fft
from scipy.ndimage import zoom
from skimage.transform import resize


def fit_wavelet(img, resolution, **kwargs):
    """This function estimates the local column spacing and bandedness of an REAL-VALUED FIELD with wavelet analysis.

    Based on the algorithm described in Keil et al. 2010 (https://doi.org/10.1073/pnas.0913020107)


    kwargs:
        roi (np.array): Boolean of roi
        k_base (float): width of the wavelet. a value of 2 captures more local structure, while 7 will reflect any periodicity
        k_interp (float): when using splines fits to assess the actual wavelength, what is the minimal step unit for the wavelength.
        min_wavelength (float): smallest wavelength for estimation [microns]
        max_wavelength (float): largest wavelength for estimation [microns]
        wavelength_steps (int):  step size for wavelength [microns]
        num_ors (int): number of wavelet orientations to be tested
        or_offset (float): global orientation offset
        use_fft (bool): where possible will convolve the wavelets using a FFT (must faster than a normal FFT)
        use_max (bool): uses the best fitting wavelet orientation, rather than the average to estimate the wavelength
        skip_bandedness (bool): only computes wavelength and not bandedness
        anisotropy (float): anisotropy of wavelet
    Args:
        img (np.array): Image to be fit
        resolution (float): spatial resolution (microns/pixel)
    Returns:
        local_wave [np.array]: local colum spacing as a function of position
        local_band [np.array]: local bandedness as a function of position
        kwargs (dict): settings used for the wavelength fitting
    """
    # mask non-finite values

    kwargs['roi'] = kwargs.get('roi', np.ones(img.shape, dtype=bool))    # roi
    kwargs['k_base'] = kwargs.get('k_base', 7)
    kwargs['min_wavelength'] = kwargs.get('min_wavelength', 400)
    kwargs['max_wavelength'] = kwargs.get('max_wavelength', 1000)
    kwargs['wavelength_step'] = kwargs.get('wavelength_step', 25)
    kwargs['interp_factor'] = kwargs.get('interp_factor', 5)
    kwargs['num_ors'] = kwargs.get('num_ors', 24)
    kwargs['anisotropy'] = kwargs.get('anisotropy', 1)

    img[~np.isfinite(img)] = 0

    # Wavelength Fitting
    logging.debug('Running Wavelengths...')
    wavelengths = np.arange(kwargs['min_wavelength'], kwargs['max_wavelength']+.001,
                            kwargs['wavelength_step'])
    wavelet_ors = np.linspace(0, np.pi, kwargs['num_ors']+1)[:-1]
    filter_size = np.max(img.shape)
    fit_results = np.full((img.shape[0],
                           img.shape[1],
                           wavelengths.size,
                           wavelet_ors.size),
                          np.nan,
                          dtype=np.float)

    for w_idx, wavelength in enumerate(wavelengths/resolution):
        logging.debug("Running wavelength %0.2f", wavelength)

        for or_idx, theta in enumerate(wavelet_ors):
            wavelet_filter = generate_wavelet_filter(filter_size,
                                                     wavelength,
                                                     kwargs['k_base'],
                                                     kwargs['anisotropy'],
                                                     theta)
            with scipy.fft.set_workers(2):
                fit_results[:, :, w_idx, or_idx] = np.abs(
                    fftconvolve(img, wavelet_filter, mode='same'))

    # Interpolation

    local_wave = np.full(img.shape, np.nan)

    logging.debug('Interpolating Results...')

    wavelength_interps = np.linspace(kwargs['min_wavelength'],
                                     kwargs['max_wavelength'],
                                     len(wavelengths)*kwargs['interp_factor'])

    roi_pts = np.argwhere(kwargs['roi'])

    for x_i, y_i in roi_pts:
        obs_interps = resize(fit_results[x_i, y_i],
                             (fit_results.shape[2]*kwargs['interp_factor'],
                              fit_results.shape[3]*kwargs['interp_factor']),
                             order=3, mode='constant', clip=False)
        local_wave[x_i, y_i] = wavelength_interps[np.unravel_index(obs_interps.argmax(), obs_interps.shape)[0]]
    kwargs.pop('roi')
    return local_wave, wavelength_interps, kwargs


def generate_wavelet_filter(filter_size, wavelength, k_base, aniso, theta):

    k_pixel = 2*np.pi / wavelength
    X, Y = np.meshgrid(range(filter_size), range(filter_size))
    X = X-filter_size//2
    Y = Y-filter_size//2

    wave = np.cos(np.cos(theta)*X*k_pixel + np.sin(theta)*Y*k_pixel) + \
        1j*np.sin(np.cos(theta)*X*k_pixel + np.sin(theta)*Y*k_pixel)

    k_ratio = k_base/k_pixel
    gaussian = 1/k_ratio * np.exp(-1 / (2*k_ratio**2) * (X**2 + Y**2/aniso**2))

    wavelet = gaussian * wave

    return wavelet - np.mean(wavelet)


def pad_map(img, pad_size, **kwargs):
    """[summary]

    Args:
        img (np.narray): image
        pad_size (int or tuple): Target size for the padded array (if not a tuple will pad the two )

    kwargs:
        constant_value (int, optional): [description]. Defaults to 0.

    Returns:
        img_pad [np.array]: Padded Image
        start_position (tuple):
    """

    kwargs['constant_value'] = kwargs.get('constant_value', 0)

    if not isinstance(img, np.ndarray) or img.ndim is not 2:
        raise TypeError('Must use a 2 dimensional numpy array!')

    if not isinstance(pad_size, tuple):
        pad_size = (pad_size, pad_size)

    pad_x = pad_size[0] - img.shape[0]
    pad_y = pad_size[1] - img.shape[1]

    img_pad = np.pad(img,
                     ((int(np.floor(pad_x/2)), int(np.ceil(pad_x/2))), (int(np.floor(pad_y/2)), int(np.ceil(pad_y/2)))),
                     mode='constant',
                     constant_values=kwargs['constant_value'])

    return img_pad, (int(np.floor(pad_x/2)), int(np.floor(pad_y/2)))


def remove_padding(img_pad, array_size, starting_position):
    """Removes Padding for images along first two dimensions.

    Args:
        img_pad ([type]): [description]
        array_size ([type]): [description]
        starting_position ([type]): [description]

    Raises:
        TypeError: [description]
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """
    if starting_position is not None:
        if img_pad.ndim >= 2:
            x_start = starting_position[0]
            x_stop = starting_position[0] + array_size[0]
            y_start = starting_position[1]
            y_stop = starting_position[1] + array_size[1]
            return img_pad[x_start:x_stop, y_start:y_stop, :]
        else:
            raise TypeError('Pad remove only supports arrays of 2 or more dimensions!')
    else:
        raise NotImplementedError('Requires a starting position')

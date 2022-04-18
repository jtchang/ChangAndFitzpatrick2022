import logging
import numpy as np

from pathlib import Path
from skimage.registration import phase_cross_correlation


from skimage.transform import AffineTransform, warp
from fleappy.filter import fermi
from fleappy.imgregistration.scanphase import fix_scan_phase


def register(template, tiff_stack, **kwargs):
    """Register time series tiff using phase cross correlation from skimage.

    Uses the phase cross correlation from skimage to match tiff_stack to template. Returns the transform_spec (y,x, phase). Fermi filtering is used to reduce high frequency noise.

    Args:
        template (numpy.ndarray (y,x)): Template image to register to.
        tiff_stack (numpy.ndarray (z,y,x)): Tif stack to register

    kwargs:
        maxmovement: Maximum frame movment
        tspec: Previous motion
        interframemax: Maximum frame to frame movement
        upsample_factor: Upsample for subpixel registration
        filter: Filter Settings
        scanphase: Whether to apply scan phase correction
    Returns:
        [type]: [description]
    """
    kwargs['maxmovement'] = kwargs.get('maxmovement', 10)
    kwargs['tspec'] = kwargs.get('tspec', np.empty([0, 3]))
    kwargs['interframemax'] = kwargs.get('interframemax', 10)
    kwargs['upsample_factor'] = kwargs.get('upsample_factor', 100)
    kwargs['scanphase'] = kwargs.get('scanphase', True)
    kwargs['scan_offset'] = kwargs.get('scan_offset', np.nan)
    kwargs['group_size'] = kwargs.get('group_size', 1)
    filter_settings = kwargs.get('filter', {'filter_type': None})

    number_of_frames = tiff_stack.shape[0]

    transform_spec = np.zeros((number_of_frames, 3))
    subsample_stack = _subsample_stack(tiff_stack, kwargs['group_size'])
    previous_shift = None

    for idx, frame in enumerate(subsample_stack):
        if kwargs['scanphase']:
            if np.isnan(kwargs['scan_offset']):
                transform_spec[idx*kwargs['group_size']:(idx+1)*kwargs['group_size'], 2], frame = fix_scan_phase(frame)
            else:
                transform_spec[idx*kwargs['group_size']:(idx+1)*kwargs['group_size'], 2], frame = fix_scan_phase(frame, kwargs['scan_offset'])
        else:
            transform_spec[idx*kwargs['group_size']:(idx+1)*kwargs['group_size'], 2] = 0
        if filter_settings['filter_type'] == 'fermi':
            frame = fermi.filter(frame, **filter_settings)
        xy_shift, _, _ = phase_cross_correlation(template,
                                                 frame,
                                                 upsample_factor=kwargs['upsample_factor'])
        tspec = -1 * np.flip(xy_shift)
        transform_spec[idx*kwargs['group_size']:(idx+1)*kwargs['group_size'], :2] = _clip_shifts(tspec,
                                                                                                 previous_shift,
                                                                                                 kwargs['interframemax'],
                                                                                                 kwargs['maxmovement'])

        previous_shift = transform_spec[idx, :]
    return transform_spec


def transform(img_stack, transform_spec):
    """Applies(y,x,phase) transform to a series of images.

    Args:
        img_stack (numpy.array): Uncorrected tif stack (t, y, x)
        transform_spec (numpy.array): Shifts to be applied to tiff stack in format (t, (y, x, phase))

    Returns:
        np.array: Corrected tif stack (t, y, x)
    """
    new_stack = np.zeros(img_stack.shape, dtype=img_stack.dtype)
    for idx, frame in enumerate(img_stack):
        _, frame = fix_scan_phase(frame, pixel_shift=transform_spec[idx, 2])
        new_stack[idx, :, :] = _shift_img(frame, transform_spec[idx, :2])

    return new_stack.astype('int16')


def join(transform_list, transform_spec):
    if transform_list is None:
        return transform_spec
    elif isinstance(transform_list, np.ndarray) and transform_list.shape[1] == 3:
        return np.concatenate((transform_list, transform_spec), axis=0)
    else:
        raise ValueError('The transform list isn\'t a numpy array of dimensions (n,3)!')


def save(transform_list, target: Path):
    """Write the list of transformations to a file.

    Args:
        transform_list ([type]): [description]
        target (Path): [description]
    """

    np.savetxt(target, np.squeeze(transform_list), delimiter=',', fmt='%.3f', header=__name__)


def load(fname):
    """Load frame by frame transformations from file.

    Args:
        fname ([type]): [description]

    Returns:
        np.array: Transformation list  t x (y,x,phase)
    """

    tspec = np.loadtxt(fname, delimiter=',')
    if tspec.shape[1] is not 3:
        raise ValueError('File does not contain appropriate transformations for this registration type!')
    return tspec


def create_template(img_stack, scanphase=True):
    """Creates a template from the image stack.

    Args:
        img_stack (np.array): Image Stack (t, y, x)
        scanphase (bool, optional): Whether to apply scan phase fix or not. Defaults to True.

    Returns:
        np.array: Templat image (y,x)
    """

    template = np.mean(img_stack[:img_stack.shape[0]//2, :, :], axis=0, dtype=np.float)
    scan_offset = 0
    if scanphase:
        x_shift, _ = fix_scan_phase(template[:, 50:-50])
        _, template = fix_scan_phase(template, x_shift)
        logging.info('Applying phase corection of %s', x_shift)
    return scan_offset, template


def _shift_img(image, vector):
    """ Translate an image using affine transformation.


    Args:
        image (np.array): Image to shift (y,x)
        vector (np.array): Shift to apply (y,x)

    Returns:
        np.array: The translated image
    """
    affine_transform = AffineTransform(translation=vector)
    shifted = warp(image,
                   affine_transform,
                   mode='constant', cval=0, preserve_range=True)

    return shifted


def _clip_shifts(xy_shift, previous_shift, inter_frame_max=None, maxmovement=None):
    """Clips the translations based on the maximum interframe and overall movements

    Args:
        xy_shift ([type]): [description]
        previous_shift ([type]): [description]
        inter_frame_max ([type], optional): [description]. Defaults to None.
        maxmovement ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if previous_shift is None or np.isnan(previous_shift[1]) or np.isnan(previous_shift[0]):
        return xy_shift

    if inter_frame_max is not None and np.linalg.norm(xy_shift[:2]-previous_shift[:2]) > inter_frame_max:
        xy_shift[:2] = previous_shift[:2]
    elif maxmovement is not None and np.any(np.abs(xy_shift[:2]) > maxmovement):
        xy_shift[:2] = previous_shift[:2]

    return xy_shift


def _subsample_stack(img, group_size):

    new_img = np.empty((img.shape[0]//group_size, img.shape[1], img.shape[2]))

    for idx in range(new_img.shape[0]):
        new_img[idx, :, :] = np.mean(img[idx*group_size:(idx+1)*group_size, :, :], axis=0)

    return new_img

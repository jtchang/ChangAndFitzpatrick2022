from skimage import data, segmentation
import matplotlib.pyplot as plt
import numpy as np


def get_contour(roi, template):

    mask = roi.mask.todense()
    clean_border = segmentation.clear_border(mask).astype(np.int)
    template = segmentation.mark_boundaries(template, clean_border, color=)


def plot_contours(roi_array: np.ndarray, template: np.ndarray, fig: plt.figure = None, ax: plt.axis = None):
    """Plot contours of roi.

    Args:
        roi_array (numpy.ndarray): Array of roi masks.
        template (numpy.ndarray): Template image.
        fig (matplotlib.pyplot.figure, optional): Defaults to None. Figure to plot to.
        ax (matplotlib.pyplot.axis, optional): Defaults to None. Axis to plot to.
    """

    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    elif ax is None and fig is not None:
        ax = fig.add_subplot(111)
    z, _, _ = roi_array.shape
    np.random.seed(128)
    colors = np.random.rand(3, z)

    for idx, img_slice in enumerate(roi_array):
        clean_border = segmentation.clear_border(img_slice).astype(np.int)
        template = segmentation.mark_boundaries(template, clean_border, color=colors[:, idx], mode='thick')
    ax.imshow(template)

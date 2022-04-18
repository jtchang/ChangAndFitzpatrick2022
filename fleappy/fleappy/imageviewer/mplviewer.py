import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import skimage.io as io
import skimage.measure as measure


def image_projection(stack, type = 'mean', clip=99, display=True, cmap='gray'):
    """Displays an image projection using matplotlib of an image series.
    
    Args:
        stack (numpy array): image stack of dimensions (z, y, x)
        type (str, optional): Defaults to 'mean'. Type of projection to perform.
        clip (int, optional): Defaults to 99. Clipping percentile for image display
        display (bool, optional): Defaults to True. Whether or not to display the projection.
    Returns:
        [numpy array]: projection of specified type as (y,x)
    """

    if type is 'mean':
        projection = np.mean(stack, axis=0)
    elif type is 'median':
        projection = np.median(stack, axis=0)
    elif type is 'std':
        projection = np.std(stack, axis=0)
    if clip > 0:
        clip_value = np.percentile(projection, clip)
        projection[projection > clip_value] = clip_value
        projection= projection / clip_value
    if display:
        plt.imshow(projection, cmap=cmap)
    return projection

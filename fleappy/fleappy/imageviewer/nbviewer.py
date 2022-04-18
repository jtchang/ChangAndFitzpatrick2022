import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import skimage.io as io
import skimage.measure as measure
from pathlib import Path


class InlineViewer():
    """ View tiff stack in a Jupyter notebook.

    Attributes:
        filename (str) : Filename for 
        anim (matplotlib ArtistAnimation) : Frame Animation of the tiff stack
        ds_factor (tuple) = Defaults to (10,1,1). Down sampling factor in (z,y,x)
    """

    def __init__(self):
        self.anim = None
        self.filename = None
        self.dsFactor = (10, 1, 1)

    def load_file(self, filename: str, dsFactor=(10, 1, 1)):
        """Load tif file and display time series data

        Args:
            filename (str): String to tif file path
            dsFactor (tuple, optional):  Defaults to (10,1,1). Downsampling factor in (z, y, x).

        Returns:
            (IPython.display.HTML) : HTML for embedded image movie
        """
        assert isinstance(filename, str) and filename.endswith(
            '.tif') and '\\' not in filename, 'Specify path to a .tif file as a string using unix style!'
        file_to_open = Path(filename)
        assert file_to_open.exists(), 'File does not exist!'

        img_stack = io.imread(file_to_open)
        self.filename = filename
        return self.load_array(img_stack, dsFactor=dsFactor)

    def load_array(self, img_stack, dsFactor=(10, 1, 1)):
        """Display image time series data from a numpy array.

        Args:
            img_stack (numpy.ndarray): Image data in (z,y,x).
            dsFactor (tuple, optional): Defaults to (10,1,1). Downsampling factor in (z, y, x).
        Returns:
            (IPython.display.HTML) : HTML for embedded image move
        """

        self.dsFactor = dsFactor

        img_stack = measure.block_reduce(
            img_stack, dsFactor, func=np.mean)
        tmax, _, _ = img_stack.shape
        img_stack = img_stack / np.percentile(img_stack[:], 99)
        imgs = []
        fig = plt.figure()
        for i in range(0, tmax):
            im = plt.imshow(img_stack[i, :, :], animated=True, cmap='gray')
            _ = im.axes.get_xaxis().set_visible(False)
            _ = im.axes.get_yaxis().set_visible(False)
            imgs.append([im])

        self.anim = animation.ArtistAnimation(
            fig, imgs, interval=50, blit=True)
        plt.close(fig)
        return self.view_stack()

    def view_stack(self):
        """View the embedded tif movie.

        Returns:
            (IPython.display.HTML) : HTML for embedded image movie
        """

        return HTML(self.anim.to_jshtml())

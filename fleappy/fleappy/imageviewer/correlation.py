import numpy as np
from scipy.stats import pearsonr


def correlation_image(imgstack, neigboring_pixels=1):
    corr_image = np.empty((imgstack.shape[1], imgstack.shape[2]))
    for row in range(imgstack.shape[1]):
        for col in range(imgstack.shape[2]):
            pixel_tseries = imgstack[:, row, col]
            corr = []
            for neighboring_row in range(-1*neighboring_pixels+row, neighboring_pixels+row, 1):
                for neighboring_col in range(-1*neighboring_pixels+col, neighboring_pixels+col, 1):
                    if neighboring_row < 1 or neighboring_col < 1 or (neighboring_row == 0 and neighboring_col == 0):
                        continue
                    else:
                        r, _ = pearsonr(pixel_tseries, imgstack[:, neighboring_row, neighboring_col])
                        corr = corr + []

            corr_image[row, col] = np.mean(corr)

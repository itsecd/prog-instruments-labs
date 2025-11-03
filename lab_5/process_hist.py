import cv2
import logging

import matplotlib.pyplot as plt
import numpy as np

from process_image import *


logger = logging.getLogger("histogram")


def make_hist(img: np.ndarray) -> tuple:
    """
    make histogram of image
    :param img: original image
    :return: list with histograms
    """
    logger.info("HISTOGRAM_CALCULATION_STARTED - Calculating histograms for BGR channels")

    try:
        if img.size == 0:
            logger.error("HISTOGRAM_CALCULATION_FAILED - Empty image provided")
            raise ValueError("Cannot calculate histogram for empty image")
        
        hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])

        logger.debug(
            "HISTOGRAM_STATS - Blue max: %d, Green max: %d, Red max: %d", 
            np.max(hist_b), 
            np.max(hist_g), 
            np.max(hist_r)
        )
        logger.info("HISTOGRAM_CALCULATION_COMPLETED - Histograms calculated successfully")
        
        return hist_b, hist_g, hist_r
        
    except Exception as e:
        logger.error("HISTOGRAM_CALCULATION_FAILED - Error: %s", str(e))
        raise


def print_hist(hist: tuple) -> None:
    """
    build and show histogram
    :param hist: tuple with histograms for blue, green and red colours
    """
    logger.info("HISTOGRAM_DISPLAY_STARTED - Displaying histogram plot")

    plt.figure()
    plt.title("Histogram of image")
    plt.xlabel("Values of pixels")
    plt.ylabel("Frequency of colour")

    plt.plot(hist[0], color = 'blue', label = 'Blue channel')
    plt.plot(hist[1], color = 'green', label = 'Green channel')
    plt.plot(hist[2], color = 'red', label = 'Red channel')

    plt.xlim([0, 256])
    plt.legend()

    plt.tight_layout()
    plt.show()

    logger.info("HISTOGRAM_DISPLAYED_SUCCESSFULLY - Histogram shown to user")
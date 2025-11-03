import cv2
import matplotlib.pyplot as plt
import numpy as np

from process_image import *

def make_hist(img: np.ndarray) -> tuple:
    """
    make histogram of image
    :param img: original image
    :return: list with histograms
    """
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
    return hist_b, hist_g, hist_r

def print_hist(hist: tuple) -> None:
    """
    build and show histogram
    :param hist: tuple with histograms for blue, green and red colours
    """
    plt.figure()
    plt.title("Histogram of image")
    plt.xlabel("Values of pixels")
    plt.ylabel("Frequency of colour")

    plt.plot(hist[0], color = 'blue', label = 'Blue channel')
    plt.plot(hist[1], color = 'green', label = 'Green channel')
    plt.plot(hist[2], color = 'red', label = 'Red channel')

    plt.xlim([0, 256])
    plt.legend()
    plt.show()
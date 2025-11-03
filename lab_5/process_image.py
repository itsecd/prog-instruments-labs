import cv2
import numpy as np

def read_image(img: str) -> np.ndarray:
    """
    read image from file
    :param img: path to image
    :return: image
    """
    img = cv2.imread(img)
    return img

def get_size(img: np.ndarray) -> tuple:
    """
    print size of image
    :param img: image
    """
    return img.shape[0], img.shape[1]

def invert(img: np.ndarray) -> np.ndarray:
    """
    make inverted pixels
    :param img: original image
    :return: changed image
    """    
    return 255  - img
    
def print_differences(img: np.ndarray, invert_img: np.ndarray) -> None:
    """
    print original and inverted images
    :param img: original image
    :param invert_img: inverted image
    """
    cv2.imshow('Original image', img)
    cv2.waitKey(0)

    cv2.imshow('Inverted image', invert_img)
    cv2.waitKey(0)

def save_data(path: str, img: np.ndarray) -> None:
    """
    Save image to file
    :param path: Path to file
    :param img: Saving image
    """
    try:
        cv2.imwrite(path, img)
    except:
        raise Exception('Can not save image')
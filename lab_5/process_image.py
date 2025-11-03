import cv2
import logging

import numpy as np


logger = logging.getLogger("image_processing")


def read_image(img_path: str) -> np.ndarray:
    """
    read image from file
    :param img_path: path to image
    :return: image
    """
    logger.info("IMAGE_READ_STARTED - Reading image from: %s", img_path)

    try:
        img = cv2.imread(img_path)
        if img is None:
            logger.error("IMAGE_READ_FAILED - Cannot read image from: %s", img_path)
            raise FileNotFoundError(f"Cannot read image from {img_path}")
        
        logger.info(
            "IMAGE_READ_SUCCESSFUL - Image loaded successfully, shape: %s", 
            img.shape
        )
        return img
    
    except Exception as e:
        logger.error("IMAGE_READ_ERROR - Unexpected error: %s", str(e))


def get_size(img: np.ndarray) -> tuple:
    """
    print size of image
    :param img: image
    :return: width and height of image
    """
    logger.debug("GETTING_IMAGE_SIZE - Image shape: %s", img.shape)
    
    height, width = img.shape[0], img.shape[1]
    logger.info("IMAGE_SIZE - Width: %d, Height: %d", width, height)
    return width, height


def invert(img: np.ndarray) -> np.ndarray:
    """
    make inverted pixels
    :param img: original image
    :return: changed image
    """
    logger.info("IMAGE_INVERSION_STARTED - Inverting image pixels")

    try:
        if img.size == 0:
            logger.error("IMAGE_INVERSION_FAILED - Empty image provided")
            raise ValueError("Cannot invert empty image")
        
        inverted_img = 255 - img
        logger.info("IMAGE_INVERSION_COMPLETED - Image inverted successfully")
        return inverted_img
        
    except Exception as e:
        logger.error("IMAGE_INVERSION_FAILED - Error: %s", str(e))
    

def print_differences(img: np.ndarray, invert_img: np.ndarray) -> None:
    """
    print original and inverted images
    :param img: original image
    :param invert_img: inverted image
    """
    logger.info("IMAGE_DISPLAY_STARTED - Displaying original and processed images")
    
    try:
        cv2.imshow('Original image', img)
        cv2.waitKey(0)

        cv2.imshow('Inverted image', invert_img)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()

        logger.info("IMAGE_DISPLAY_COMPLETED - Images displayed successfully")
        
    except Exception as e:
        logger.error("IMAGE_DISPLAY_FAILED - Error: %s", str(e))
        raise


def save_data(path: str, img: np.ndarray) -> None:
    """
    Save image to file
    :param path: Path to file
    :param img: Saving image
    """
    logger.info("IMAGE_SAVE_STARTED - Saving image to: %s", path)

    try:
        success = cv2.imwrite(path, img)
        if success:
            logger.info("IMAGE_SAVE_SUCCESSFUL - Image saved successfully to: %s", path)
        else:
            logger.error("IMAGE_SAVE_FAILED - Failed to save image to: %s", path)
            raise Exception(f'Cannot save image to {path}')
        
    except Exception as e:
        logger.error("IMAGE_SAVE_ERROR - Error: %s", str(e))
        raise


def get_image_info(img: np.ndarray) -> dict:
    """
    Get information about image
    param img: image
    :return: image info
    """
    logger.debug("GETTING_IMAGE_INFO - Analyzing image properties")
    
    info = {
        'shape': img.shape,
        'dtype': str(img.dtype),
        'min_value': np.min(img),
        'max_value': np.max(img),
        'mean_value': np.mean(img)
    }
    
    logger.info("IMAGE_INFO_RETRIEVED - Image analysis completed")
    return info
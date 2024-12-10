import logging
from enum import Enum
import numpy as np
import cv2


logging.basicConfig(filename='fusing_methods.log',
                    level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RuleType(Enum):
    """
    Enumerable for fusing method choice
    """
    average = 1
    variance_weighted = 2
    minimum_error = 3
    max = 4
    min = 5
    weight = 6
    power_transformation = 7

class FusingBase:
    """
    Base class for single pixel fusing methods
    """
    @staticmethod
    def calculate_intensity(images: list, errors: list):
        logger.debug("Calculating intensity in FusingBase")
        pass

class AverageFusing(FusingBase):
    """
    The most primitive method
    """
    @staticmethod
    def calculate_intensity(images: list, errors: list):
        logger.info("Calculating average intensity")
        result = np.mean(images, axis=0)
        logger.debug("Average intensity calculated")
        return result

class MaxFusing(FusingBase):
    @staticmethod
    def calculate_intensity(images: list, errors: list):
        logger.info("Calculating max intensity")
        result = np.max(images, axis=0)
        logger.debug("Max intensity calculated")
        return result

class MinFusing(FusingBase):
    @staticmethod
    def calculate_intensity(images: list, errors: list):
        logger.info("Calculating min intensity")
        result = np.min(images, axis=0)
        logger.debug("Min intensity calculated")
        return result

class WeightFusing(FusingBase):
    @staticmethod
    def calculate_intensity(images: list, errors: list):
        logger.info("Calculating weighted intensity")
        tmp = images[0]
        for i in range(len(images)-1):
            weight_coeff = np.sum(images[i+1])/(np.sum(tmp) + np.sum(images[i+1]))
            tmp = weight_coeff * tmp + (1 - weight_coeff) * images[i+1]
            logger.debug(f"Weight coefficient for image {i+1}: {weight_coeff}")
        result = tmp
        logger.debug("Weighted intensity calculated")
        return result

class PowerTransformationFusing(FusingBase):
    @staticmethod
    def calculate_intensity(images: list, errors: list):
        logger.info("Calculating intensity using power transformation")
        tmp = images[0]
        image_depth = 8
        for i in range(len(images) - 1):
            power = 1 + images[i+1] / np.power(2, image_depth)
            tmp = np.power(tmp, power)
            logger.debug(f"Power transformation for image {i+1}: {power}")
        result = np.where(tmp == float('inf'), 255, 0)
        logger.debug("Power transformation intensity calculated")
        return result

class VarianceFusing(FusingBase):
    """
    Method described in Maksimov A.I., Sergeev V.V.
    Optimal fusing of video sequence images //
    Proceedings of ITNT 2020 - 6th IEEE International Conference
     on Information Technology and Nanotechnology. — 2020.
    """
    @staticmethod
    def calculate_intensity(images: list, errors: list):
        logger.info("Calculating intensity using variance method")
        upper = 0
        lower = 0
        avg = np.array([])
        tmp_pixels = []
        pixels = []
        tmp_errors = []
        err = []
        for lvl in range(images[0].shape[0]):
            for element in range(images[0].shape[1]):
                for count_of_img in range(len(images)):
                    tmp_pixels.append(images[count_of_img][lvl][element])
                    tmp_errors.append(errors[count_of_img][0][lvl]['i'])
                pixels.append(tmp_pixels.copy())
                err.append(tmp_errors.copy())
                tmp_pixels.clear()
                tmp_errors.clear()
        for intensity, variance in zip(pixels, err):
            for i in range(len(intensity)):
                if variance[i] == 0:
                    avg = np.append(avg, intensity[i])
                    continue
                upper += intensity[i] / variance[i]
                lower += 1 / variance[i]
                tmp_pixels.append(upper)
                tmp_errors.append(lower)
                upper = 0
                lower = 0
        if len(avg):
            logger.debug("Returning mean of average intensities")
            return np.mean(avg)
        else:
            logger.debug("Returning computed intensities based on variance")
            return [val1 / val2 for val1, val2 in zip(tmp_pixels, tmp_errors)]

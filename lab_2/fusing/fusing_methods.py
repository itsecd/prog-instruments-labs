from enum import Enum
import numpy as np
import cv2


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
    # and so on


class FusingBase:
    """
    Base class for single pixel fusing methods
    """
    @staticmethod
    def calculate_intensity(images: list, errors: list):
        pass


class AverageFusing(FusingBase):
    """
    The most primitive method
    """
    @staticmethod
    def calculate_intensity(images: list, errors: list):
        return np.mean(images, axis=0)


class MaxFusing(FusingBase):
    @staticmethod
    def calculate_intensity(images: list, errors: list):
        return np.max(images, axis=0)

class MinFusing(FusingBase):
    @staticmethod
    def calculate_intensity(images: list, errors: list):
        return np.min(images, axis=0)


class WeightFusing(FusingBase):
    @staticmethod
    def calculate_intensity(images: list, errors: list):
        tmp = images[0]
        for i in range(len(images)-1):
            weight_coeff = np.sum(images[i+1])/(np.sum(tmp) + np.sum(images[i+1]))
            tmp = weight_coeff*tmp + (1-weight_coeff)*images[i+1]
        result = tmp
        return result


class PowerTransformationFusing(FusingBase):
    @staticmethod
    def calculate_intensity(images: list, errors: list):
        # tmp_pixels = []
        # pixels = []
        # image_depth = 256
        # result = []
        # reshape = 0
        # for lvl in range(images[0].shape[0]):
        #     for element in range(images[0].shape[1]):
        #         for count_of_img in range(len(images)):
        #             tmp_pixels.append(images[count_of_img][lvl][element])
        #         pixels.append(tmp_pixels.copy())
        #         tmp_pixels.clear()
        # for i in range(len(pixels)):
        #     tmp = pixels[i][0]
        #     for j in range(len(pixels[i])):
        #         power = 1 - pixels[i][j]/image_depth
        #         tmp = np.power(tmp, power)
        #     result.append(tmp)
        # for i in range(500, 550):
        #     if len(result) % i == 0:
        #         reshape = i
        # result = np.array(result).reshape((reshape, reshape))
        # threshold = 2
        # _, thresholded_image = cv2.threshold(result, threshold, 255, cv2.THRESH_BINARY)
        tmp = images[0]
        image_depth = 8
        for i in range(len(images) - 1):
            power = 1 + images[i+1]/np.power(2, image_depth)
            tmp = np.power(tmp, power)
        result = tmp
        return np.where(result == float('inf'), 255, 0)


class VarianceFusing(FusingBase):
    """
    Method described in Maksimov A.I., Sergeev V.V.
    Optimal fusing of video sequence images //
    Proceedings of ITNT 2020 - 6th IEEE International Conference
     on Information Technology and Nanotechnology. — 2020.
    """
    @staticmethod
    def calculate_intensity(images: list, errors: list):
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
            return np.mean(avg)
        else:
            return [val1 / val2 for val1, val2 in zip(tmp_pixels, tmp_errors)]

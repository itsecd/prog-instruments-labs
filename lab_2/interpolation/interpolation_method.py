from enum import Enum
import cv2
import numpy as np
from scipy import interpolate


class RuleType(Enum):
    interpn = 1
    regulargridinterpolator = 2
    rectbivariatespline = 3


class InterpolationBase:
    @staticmethod
    def interpolation(increase_image: np.ndarray, image: np.ndarray, scaling_factor: int):
        pass


class Interpn(InterpolationBase):
    @staticmethod
    def interpolation(increase_image: np.ndarray, image: np.ndarray, scaling_factor: int):
        x = np.linspace(0, increase_image.shape[0] - 1, increase_image.shape[0])
        y = np.linspace(0, increase_image.shape[1] - 1, increase_image.shape[1])
        points = (x, y)
        values = cv2.resize(image, (increase_image.shape[1], increase_image.shape[0]))
        xi = ([(i, j) for i, row in enumerate(values) for j, _ in enumerate(row)
              if i % int(increase_image.shape[1]/image.shape[1]) != 0
              or j % int(increase_image.shape[0]/image.shape[0]) != 0])
        interp_value = interpolate.interpn(points, values, xi, method='linear')
        counter = 0
        for interp_pixel in xi:
            increase_image[interp_pixel[0], interp_pixel[1]] = interp_value[counter]
            counter += 1
        return increase_image


class RegularGridInterpolator(InterpolationBase):
    @staticmethod
    def interpolation(increase_image: np.ndarray, image: np.ndarray, scaling_factor: int):
        x = np.linspace(0, increase_image.shape[0] - 1, increase_image.shape[0])
        y = np.linspace(0, increase_image.shape[1] - 1, increase_image.shape[1])
        points = (x, y)
        values = cv2.resize(image, (increase_image.shape[1], increase_image.shape[0]))
        interp_value = interpolate.RegularGridInterpolator(points, values)
        return interp_value


class RectBivariateSpline(InterpolationBase):
    @staticmethod
    def interpolation(increase_image: np.ndarray, image: np.ndarray, scaling_factor: int):
        x = np.linspace(0, increase_image.shape[0] - 1, increase_image.shape[0])
        y = np.linspace(0, increase_image.shape[1] - 1, increase_image.shape[1])
        values = increase_image
        interp_func = interpolate.RectBivariateSpline(x, y, values)
        interp_values = interp_func(x, y)
        return interp_values

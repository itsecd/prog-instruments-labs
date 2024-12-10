import json
import cv2 as cv
import numpy as np
from os import listdir
from os.path import splitext, join


class ImageStack:
    """"
    Represents a set of two 3d arrays - images and error fields - of H*W*N shape, where
    H - height of image,
    W - width of image,
    N - number of images
    """
    def __init__(self, path_to_images: str, path_to_errors: str):
        """"
        To initialize stack correctly, images and corresponding errors
        should be previously named according to the same rule,
         ex. 0001.png, 0001.json
        :param path_to_images: Path to a folder containing images
        :param path_to_errors: Path to a folder containing error fields
        """
        self._images = self._load_images(path_to_images)
        self._errors = self._load_errors(path_to_errors)
        # if not self._validate_stack():
        #     raise ValueError('ImageStack is invalid')

    @property
    def images(self):
        return self._images

    @property
    def errors(self):
        return self._errors

    @staticmethod
    def _load_images(path_to_images: str):
        image_array = []
        files = sorted([f for f in listdir(path_to_images) if splitext(f)[1] in ('.png', '.jpg')])
        for f in files:
            image_array.append(cv.imread(join(path_to_images, f), cv.IMREAD_GRAYSCALE))
        return image_array

    @staticmethod
    def _load_errors(path_to_errors: str):
        errors_array = []
        files = sorted([f for f in listdir(path_to_errors) if splitext(f)[1] in ('.json', '.npy')])
        for f in files:
            with open(join(path_to_errors, f), "r") as file:
                errors_array.append(json.load(file))
        result = np.array(errors_array)
        return result

    def _validate_stack(self):
        count = 0
        if len(self._images) == len(self._errors):
            for i in range(len(self._images)):
                if self._images[i].shape == self._errors[i].shape:
                    count += 1
            if count == len(self._images):
                return True
        return False

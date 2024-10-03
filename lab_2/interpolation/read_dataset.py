import cv2 as cv
from os import listdir
from os.path import splitext, join


class ReadData:
    def __init__(self, path_to_images: str):
        self._images = self._read_data(path_to_images)

    @property
    def images(self):
        return self._images

    @staticmethod
    def _read_data(path_to_images: str):
        image_array = []
        files = sorted([f for f in listdir(path_to_images) if splitext(f)[1] in ('.png', '.jpg')])
        for f in files:
            image_array.append(cv.imread(join(path_to_images, f), cv.IMREAD_GRAYSCALE))
        return image_array

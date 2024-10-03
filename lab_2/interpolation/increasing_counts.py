import numpy as np


class IncreasingCounts:
    def __init__(self, data: np.ndarray, scaling_factor: int):
        self._increase_image = self.increasing(data, scaling_factor)

    @property
    def increase_image(self):
        return self._increase_image

    @staticmethod
    def increasing(data: np.ndarray, scaling_factor: int):
        increase_data = []
        for i in range(len(data)):
            height, width = data[i].shape
            new_height, new_width = height * scaling_factor, width * scaling_factor
            new_data = np.zeros((new_height, new_width), dtype=np.uint8)
            new_data[::scaling_factor, ::scaling_factor] = data[i]
            increase_data.append(new_data)
        return increase_data

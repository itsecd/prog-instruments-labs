from interpolation.increasing_counts import IncreasingCounts
from interpolation.read_dataset import ReadData
from interpolation.interpolation_method import Interpn, RegularGridInterpolator, RectBivariateSpline
import cv2
import numpy as np
import copy
import itertools
import json



class InterpolationModule:
    def __init__(self, image: np.array, scale: int):
        self._image = image
        self._scale = scale

    def interpolation(self):
        images = copy.copy(self._image.images)
        increase_data = IncreasingCounts(images, self._scale)
        for i in range(len(increase_data.increase_image)):
            cv2.imwrite(f'../increase_dataset/{self._scale}/increase_dataset{i}.png', increase_data.increase_image[i])
            interpn_data = Interpn.interpolation(increase_data.increase_image[i], data.images[i], self._scale)
            cv2.imwrite(f'../interpolation_dataset/interpn/{self._scale}/im{i}.png', interpn_data)
            rgi_data = RegularGridInterpolator.interpolation(increase_data.increase_image[i], data.images[i],
                                                             self._scale)
            cv2.imwrite(f'../interpolation_dataset/RGI/{self._scale}/im{i}.png', rgi_data.values)
            rbs_data = RectBivariateSpline.interpolation(increase_data.increase_image[i], data.images[i], self._scale)
            cv2.imwrite(f'../interpolation_dataset/RBS/{self._scale}/im{i}.png', rbs_data)


if __name__ == '__main__':
    path = "../dataset/x0.3333"
    data = ReadData(path)
    increase = 3
    interpolation_data = InterpolationModule(data, increase)
    interpolation_data.interpolation()
    orig_img = ReadData('shift_orig_img')
    for j in range(2, 11):
        interp_data = ReadData(f'../interpolation_dataset/RBS/{j}')
        increase_data = ReadData(f'../increase_dataset/{j}')
        shift = np.array([0, 131, -131])
        h, w = interp_data.images[0].shape[:2]
        next_img = 0
        result_im = []
        result_interp = []
        result_increase = []
        tmp_dispersion = 0
        forward_list = list(itertools.combinations_with_replacement(shift, 2))
        reversed_list = [tuple(reversed(item)) for item in forward_list]
        combinations = sorted(list(set(forward_list + reversed_list)))
        for i in combinations:
            cropped_image = orig_img.images[next_img][256+i[1]-35:256+i[1]+35, 256+i[0]-120:256+i[0]+120]
            cropped_interp = interp_data.images[next_img][256+i[1]-35:256+i[1]+35, 256+i[0]-120:256+i[0]+120]
            cropped_increase = increase_data.images[next_img][256+i[1]-35:256+i[1]+35, 256+i[0]-120:256+i[0]+120]
            result_im.append(cropped_image)
            result_interp.append(cropped_interp)
            result_increase.append(cropped_increase)
            cv2.imwrite(f'./cropped_image/result_im{next_img}.png', result_im[next_img])
            cv2.imwrite(f'./cropped_interp/RBS/{j}/result_interp{next_img}.png', result_interp[next_img])
            cv2.imwrite(f'./cropped_increase/{j}/result_increase{next_img}.png', result_increase[next_img])
            next_img += 1
        for i in range(len(result_interp)):
            tmp_dispersion += (np.mean((result_interp[i]-result_im[i])**2))/(np.mean(result_im[i]**2))
            result_increase = ReadData(f'./cropped_increase/{j}')
            interp_data = ReadData(f'./cropped_interp/RBS/{j}')
            h, w = result_increase.images[0].shape[:2]
            M_x = np.zeros((h, w))
            for next_interp_point_x in range(1, j):
                interp_point_coord = np.array([(x, y) for x in range(0, h, j) for y in range(next_interp_point_x, w, j)])
                orig_point_coord = np.array([(x, y) for x in range(0, h, j) for y in range(0, w, j)])
                for step_orig in range(len(orig_point_coord)-h):
                    tmp = (result_increase.images[i][orig_point_coord[step_orig+1, 0], orig_point_coord[step_orig+1, 1]] *
                           ((orig_point_coord[step_orig + 1, 1] - interp_point_coord[step_orig, 1]) /
                            (orig_point_coord[step_orig + 1, 1] - orig_point_coord[step_orig, 1]))) + \
                          (result_increase.images[i][orig_point_coord[step_orig, 0], orig_point_coord[step_orig, 1]] *
                           ((interp_point_coord[step_orig, 1] - orig_point_coord[step_orig, 1]) /
                            (orig_point_coord[step_orig + 1, 1] - orig_point_coord[step_orig, 1])))
                    M_x[interp_point_coord[step_orig+next_interp_point_x, 0],
                        interp_point_coord[step_orig+next_interp_point_x, 1]] = tmp
            M_y = M_x + result_increase.images[i]
            for next_interp_point_y in range(1, j):
                interp_point_coord = np.array([(x, y) for x in range(next_interp_point_y, h, j) for y in range(0, w, 1)])
                orig_point_coord = np.array([(x, y) for x in range(0, h, j) for y in range(0, w, 1)])
                for step_orig in range(len(orig_point_coord) - w):
                    tmp = (M_y[orig_point_coord[step_orig, 0], orig_point_coord[step_orig, 1]] *
                           ((orig_point_coord[step_orig + h, 0] - interp_point_coord[step_orig, 0]) /
                            (orig_point_coord[step_orig + h, 0] - orig_point_coord[step_orig, 0]))) + \
                          (M_y[orig_point_coord[step_orig + h, 0], orig_point_coord[step_orig + h, 1]] *
                           ((interp_point_coord[step_orig, 0] - orig_point_coord[step_orig, 0]) /
                            (orig_point_coord[step_orig + h, 0] - orig_point_coord[step_orig, 0])))
                    M_y[interp_point_coord[step_orig, 0],
                        interp_point_coord[step_orig, 1]] = tmp
            error = interp_data.images[i] - M_y
            cv2.imwrite(f'../interpolation_error/RGI/{j}/cut_im{i}.png', error)
            with open(f'../interpolation_error/RGI/{j}/im{i}.json', 'w') as f:
                json.dump(error.tolist(), f)
        print(f'increase in {j}:{(tmp_dispersion / len(result_interp)) }')

        error_data = ReadData(f'../interpolation_error/RBS/{j}')
        next_img = 0
        result_err = []
        for i in combinations:
            cropped_err = error_data.images[next_img][256+i[1]-35:256+i[1]+35, 256+i[0]-120:256+i[0]+120]
            print(f'err{j}{i}', np.mean(cropped_err))
            result_err.append(cropped_err)
            cv2.imwrite(f'../interpolation_error/cropped_err/RBS/{j}/result_err{i}.png', result_err[next_img])
            next_img += 1
        print('avg_err', np.mean(result_err))

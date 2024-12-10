from fusing.fusing_methods import RuleType, FusingBase, AverageFusing, VarianceFusing, MaxFusing, MinFusing,\
    WeightFusing, PowerTransformationFusing
from fusing.image_stack import ImageStack
import cv2 as cv
import numpy as np
import sys
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


class FusingModule:
    """
    Fuses a stack of registered enhanced images with corresponding error fields
    into a single image via chosen fusing method
    """
    def __init__(self, stack: ImageStack, rule_type: int):
        """
        Initialization of fusing module instance
        :param stack: ImageStack object to process
        :param rule_type: RuleType enum value to choose the fusing method
        """
        self._image_stack = stack
        match rule_type:
            case RuleType.average.value:
                self._rule = AverageFusing()
            case RuleType.variance_weighted.value:
                self._rule = VarianceFusing()
            case RuleType.max.value:
                self._rule = MaxFusing()
            case RuleType.min.value:
                self._rule = MinFusing()
            case RuleType.weight.value:
                self._rule = WeightFusing()
            case RuleType.power_transformation.value:
                self._rule = PowerTransformationFusing()
            case _:
                self._rule = FusingBase()

    def fuse_image_stack(self):
        # list_of_pixels = []
        # list_of_errors = []
        # res = []
        # tmp = []
        # result = []
        # size_images = len(self._image_stack.images)
        # for lvl in range(self._image_stack.images[0].shape[0]):
        #     for element in range(self._image_stack.images[0].shape[1]):
        #         for count_of_img in range(size_images):
        #             list_of_pixels.append(self._image_stack.images[count_of_img][lvl][element])
        #             list_of_errors.append(self._image_stack.errors[count_of_img][lvl][element])
        #         tmp.append(self._rule.calculate_intensity(list_of_pixels, list_of_errors))
        #for count_of_img in range(size_images):
        # tmp.append(self._rule.calculate_intensity(self._image_stack.images, self._image_stack.errors))
            # list_of_pixels.clear()
            # list_of_errors.clear()
        # res.append(tmp)
        # tmp = []
        # result = np.array(res)
        return self._rule.calculate_intensity(self._image_stack.images, self._image_stack.errors)


if __name__ == "__main__":
    #path_to_images = sys.argv[1]
    for i in range(2,11):
        path_to_images = f'../registration_dataset/skimage/RGI/{i}'
        path_to_errors = '../interpolation_error'
        fusing = FusingModule(ImageStack(path_to_images, path_to_errors), 7)
        cv.imwrite(f'../result_image/skimage/power_transformation/RGI/result_image{i}.png', np.floor(fusing.fuse_image_stack()))
    # tmp_dispersion = 0
    # orig_img = cv.imread('../prepare_datasets/orig_number.png', cv.IMREAD_GRAYSCALE)
    # orig_ssim = ssim(orig_img, orig_img)
    # orig_mse = mean_squared_error(orig_img, orig_img)
    # print('ssim:', orig_ssim)
    # print('mse:', orig_mse)
    # for j in range(2, 11):
    #     fusing_data = cv.imread(f'../result_image/shift/power_transformation/interpn/result_image{j}.png', cv.IMREAD_GRAYSCALE)
    #     if j == 3 or j == 9:
    #         tmp_dispersion += (np.mean((fusing_data[:-1, :-1] - orig_img) ** 2)) / (np.mean(fusing_data[:-1, :-1] ** 2))
    #         fus_ssim = ssim(orig_img, fusing_data[:-1, :-1])
    #         fus_mse = mean_squared_error(orig_img, fusing_data[:-1, :-1])
    #         print(f'ssim img{j}:', fus_ssim)
    #         print(f'mse img{j}:', fus_mse)
    #     elif j == 5:
    #         tmp_dispersion += (np.mean((fusing_data[:-3, :-3] - orig_img) ** 2)) / (np.mean(fusing_data[:-3, :-3] ** 2))
    #         fus_ssim = ssim(orig_img, fusing_data[:-3, :-3])
    #         fus_mse = mean_squared_error(orig_img, fusing_data[:-3, :-3])
    #         print(f'ssim img{j}:', fus_ssim)
    #         print(f'mse img{j}:', fus_mse)
    #     elif j == 6:
    #         tmp_dispersion += (np.mean((fusing_data[:-4, :-4] - orig_img) ** 2)) / (np.mean(fusing_data[:-4, :-4] ** 2))
    #         fus_ssim = ssim(orig_img, fusing_data[:-4, :-4])
    #         fus_mse = mean_squared_error(orig_img, fusing_data[:-4, :-4])
    #         print(f'ssim img{j}:', fus_ssim)
    #         print(f'mse img{j}:', fus_mse)
    #     elif j == 7:
    #         tmp_dispersion += (np.mean((fusing_data[:-6, :-6] - orig_img) ** 2)) / (np.mean(fusing_data[:-6, :-6] ** 2))
    #         fus_ssim = ssim(orig_img, fusing_data[:-6, :-6])
    #         fus_mse = mean_squared_error(orig_img, fusing_data[:-6, :-6])
    #         print(f'ssim img{j}:', fus_ssim)
    #         print(f'mse img{j}:', fus_mse)
    #     elif j == 10:
    #         tmp_dispersion += (np.mean((fusing_data[:-8, :-8] - orig_img) ** 2)) / (np.mean(fusing_data[:-8, :-8] ** 2))
    #         fus_ssim = ssim(orig_img, fusing_data[:-8, :-8])
    #         fus_mse = mean_squared_error(orig_img, fusing_data[:-8, :-8])
    #         print(f'ssim img{j}:', fus_ssim)
    #         print(f'mse img{j}:', fus_mse)
    #     else:
    #         tmp_dispersion += (np.mean((fusing_data - orig_img) ** 2)) / (np.mean(fusing_data ** 2))
    #         fus_ssim = ssim(orig_img, fusing_data)
    #         fus_mse = mean_squared_error(orig_img, fusing_data)
    #         print(f'ssim img{j}:', fus_ssim)
    #         print(f'mse img{j}:', fus_mse)
    # print(f'error:{(1 / 9) * tmp_dispersion}')
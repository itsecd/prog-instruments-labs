import logging
from fusing.fusing_methods import RuleType, FusingBase, AverageFusing, VarianceFusing, MaxFusing, MinFusing, \
    WeightFusing, PowerTransformationFusing
from fusing.image_stack import ImageStack
import cv2 as cv
import numpy as np
import sys
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


logging.basicConfig(filename='fusing_module.log',
                    level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.info("Initializing FusingModule with rule type: %s", rule_type)
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
                logger.warning("Unknown rule type, defaulting to FusingBase.")
                self._rule = FusingBase()

    def fuse_image_stack(self):
        logger.info("Fusing image stack using %s", self._rule.__class__.__name__)
        result = self._rule.calculate_intensity(self._image_stack.images, self._image_stack.errors)
        logger.debug("Fused image stack result shape: %s", result.shape)
        return result


if __name__ == "__main__":
    # path_to_images = sys.argv[1]
    for i in range(2, 11):
        path_to_images = f'../registration_dataset/skimage/RGI/{i}'
        path_to_errors = '../interpolation_error'
        logger.info("Processing images from: %s", path_to_images)
        fusing = FusingModule(ImageStack(path_to_images, path_to_errors), 7)
        result_image = np.floor(fusing.fuse_image_stack())
        cv.imwrite(f'../result_image/skimage/power_transformation/RGI/result_image{i}.png', result_image)
        logger.info("Saved fused image to: ../result_image/skimage/power_transformation/RGI/result_image%s.png", i)

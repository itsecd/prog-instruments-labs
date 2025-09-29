import json
import numpy as np
import cv2 as cv

errors_mas = np.array([[1,2,3],[4,5,6],[7,8,9]])
np.save("errors_massive", errors_mas)

class Image_formation:
    def forming(self):
        formed_img = Pixel_avg()
        return formed_img.avg("Img_name")

    def save_img(self):
        img = self.forming()
        cv.imwrite('result.png', img)


class Deserialize_errors:
    pass

class Deserialize_npy(Deserialize_errors):
    def deserialize(self, file_name):
        return np.load(f"{file_name}.npy")

class Deserialize_JSON(Deserialize_errors):
    def deserialize(self, file_name):
        with open(f"{file_name}.json", "r") as f:
            errors = json.load(f)
        return errors

class Pixel_formation:
    pass

class Pixel_avg(Pixel_formation):
    def read_images(self,number_of_images,img_name):
        """Не знаю как посчитать количество изображений приходящих в модуль, поэтому 5"""
        image = []
        for i in range(number_of_images):
            """"""
            img = cv.imread(f'{img_name}{i}.tif', cv.IMREAD_GRAYSCALE)
            assert img is not None, "file could not be read, check with os.path.exists()"
            image.append(img)
        return image

    def avg(self,img_name):
        number_of_images = 5
        sum_pixel = []
        set_of_img = self.read_images(number_of_images,img_name)
        """Извените за говнокод поправлю в ближайшее время(когда найду метод для сложения элементов массивов)"""
        for j in range(len(set_of_img)/number_of_images):
            for i in range(number_of_images):
                sum_pixel[j] += set_of_img[i][j]
        avg = sum_pixel/number_of_images
        return avg


if __name__ == '__main__':
    test2 = Deserialize_JSON()
    print(test2.deserialize("films"))
    test = Deserialize_npy()
    print(test.deserialize("errors_massive"))

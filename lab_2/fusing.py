from tqdm import tqdm
import numpy as np
from skimage import img_as_ubyte, img_as_float
from skimage.color import rgb2gray
from skimage import io


def fusing_gui(files, additional_channel, progress_bar_info):
    import math
    progress_step = 100 / (files[0].shape[0] * files[0].shape[1]) / 2
    progress_bar_info[0]['value'] = 0
    progress_bar_info[1].config(text="0")
    progress_bar_info[2].update_idletasks()
    a = 0
    b = 0
    ones = np.ones((additional_channel[1].shape[0], additional_channel[1].shape[1]))
    # error_helper = np.ones((additional_channel[1].shape[0], additional_channel[1].shape[1])) / 1000000000
    for m in tqdm(range(0, len(files)), desc="Комплексирование: "):
        offset_img = img_as_ubyte(rgb2gray(files[m]))
        disp_err = additional_channel[m]# + error_helper
        a += offset_img / disp_err
        b += ones / disp_err
        progress_bar_info[0]['value'] += progress_step
        progress_bar_info[1].config(text=round(progress_bar_info[0]['value']))
        progress_bar_info[2].update_idletasks()
    pixel_matrix = a / b
    for i in range(0, pixel_matrix.shape[0]):
        for j in range(0, pixel_matrix.shape[1]):
            flag_nan = True
            if additional_channel[0][i][j] == 0:
                pixel_matrix[i][j] = files[0][i][j]
            if pixel_matrix[i][j] == 0:
                pixel_matrix[i][j] = files[0][i][j]
            if math.isnan(pixel_matrix[i][j]):
                for m in range(0, len(files)):
                    if flag_nan:
                        flag_nan = False
                        pixel_matrix[i][j] = 0
                    pixel_matrix[i][j] += files[m][i][j] / len(files)
            if pixel_matrix[i][j] < 0.0000005 or pixel_matrix[i][j] > 255:
                pixel_matrix[i][j] = files[0][i][j]
        progress_bar_info[0]['value'] += progress_step
        progress_bar_info[1].config(text=round(progress_bar_info[0]['value']))
        progress_bar_info[2].update_idletasks()
    pixel_matrix = img_as_ubyte((pixel_matrix - np.min(pixel_matrix)) / (np.max(pixel_matrix) - np.min(pixel_matrix)))
    progress_bar_info[0]['value'] = 100
    progress_bar_info[1].config(text=progress_bar_info[0]['value'])
    progress_bar_info[2].update_idletasks()
    return pixel_matrix

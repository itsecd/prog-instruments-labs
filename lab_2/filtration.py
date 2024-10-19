from tqdm import tqdm
import numpy as np
from skimage import img_as_ubyte, img_as_float
from skimage.color import rgb2gray
# from skimage import io


def wiener(files, progress_bar, progress_label, root):
    from skimage import restoration
    progress_step = 100 / len(files)
    progress_bar['value'] = 0
    progress_label.config(text="0")
    root.update_idletasks()
    result_array = []
    psf = np.ones((5, 5)) / 25
    for i in tqdm(range(0, len(files)), desc="Фильтрация: "):
        img = img_as_float(rgb2gray(files[i]))
        restored_img, _ = restoration.unsupervised_wiener(img, psf)
        restored_img = img_as_ubyte(
            (restored_img - np.min(restored_img)) / (np.max(restored_img) - np.min(restored_img)))
        result_array.append(restored_img)
        progress_bar['value'] += progress_step
        progress_label.config(text=round(progress_bar['value']))
        root.update_idletasks()
        # io.imsave("temp/" + str(i) + ".jpg", restored_img)
    progress_bar['value'] = 100
    progress_label.config(text=progress_bar['value'])
    root.update_idletasks()
    return result_array


def gauss(files, progress_bar, progress_label, root):
    from scipy import ndimage
    progress_step = 100 / len(files)
    progress_bar['value'] = 0
    progress_label.config(text="0")
    root.update_idletasks()
    result_array = []
    for i in tqdm(range(0, len(files)), desc="Фильтрация: "):
        restored_img = ndimage.gaussian_filter(files[i], 2)
        result_array.append(restored_img)
        progress_bar['value'] += progress_step
        progress_label.config(text=round(progress_bar['value']))
        root.update_idletasks()
        # io.imsave("temp/" + str(i) + ".jpg", restored_img)
    progress_bar['value'] = 100
    progress_label.config(text=progress_bar['value'])
    root.update_idletasks()
    return result_array


def median(files, progress_bar, progress_label, root):
    from scipy import ndimage
    progress_step = 100 / len(files)
    progress_bar['value'] = 0
    progress_label.config(text="0")
    root.update_idletasks()
    result_array = []
    for i in tqdm(range(0, len(files)), desc="Фильтрация: "):
        restored_img = ndimage.median_filter(files[i], 3)
        result_array.append(restored_img)
        progress_bar['value'] += progress_step
        progress_label.config(text=round(progress_bar['value']))
        root.update_idletasks()
        # io.imsave("temp/" + str(i) + ".jpg", restored_img)
    progress_bar['value'] = 100
    progress_label.config(text=progress_bar['value'])
    root.update_idletasks()
    return result_array


def contrast(files, progress_bar, progress_label, root):
    import cv2
    progress_step = 100 / len(files)
    progress_bar['value'] = 0
    progress_label.config(text="0")
    root.update_idletasks()
    result_array = []
    for i in tqdm(range(0, len(files)), desc="Фильтрация: "):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        restored_img = clahe.apply(files[i])
        result_array.append(restored_img)
        progress_bar['value'] += progress_step
        progress_label.config(text=round(progress_bar['value']))
        root.update_idletasks()
        # io.imsave("temp/" + str(i) + ".jpg", restored_img)
    progress_bar['value'] = 100
    progress_label.config(text=progress_bar['value'])
    root.update_idletasks()
    return result_array


def sharpen(files, progress_bar, progress_label, root):
    import cv2
    sharpen_mask = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])
    progress_step = 100 / len(files)
    progress_bar['value'] = 0
    progress_label.config(text="0")
    root.update_idletasks()
    result_array = []
    for i in tqdm(range(0, len(files)), desc="Фильтрация: "):
        restored_img = cv2.filter2D(files[i], -1, sharpen_mask)
        result_array.append(restored_img)
        progress_bar['value'] += progress_step
        progress_label.config(text=round(progress_bar['value']))
        root.update_idletasks()
        # io.imsave("temp/" + str(i) + ".jpg", restored_img)
    progress_bar['value'] = 100
    progress_label.config(text=progress_bar['value'])
    root.update_idletasks()
    return result_array


def denoise(files, progress_bar, progress_label, root):
    from skimage.restoration import denoise_nl_means, estimate_sigma
    progress_step = 100 / len(files)
    progress_bar['value'] = 0
    progress_label.config(text="0")
    root.update_idletasks()
    result_array = []
    patch_kw = dict(patch_size=2, patch_distance=2)
    for i in tqdm(range(0, len(files)), desc="Фильтрация: "):
        sigma_est = np.mean(estimate_sigma(files[i]))
        img = img_as_float(files[i])
        restored_img = denoise_nl_means(img, h=0.8 * sigma_est, sigma=sigma_est, fast_mode=False, **patch_kw)
        restored_img = img_as_ubyte(
            (restored_img - np.min(restored_img)) / (np.max(restored_img) - np.min(restored_img)))
        result_array.append(restored_img)
        progress_bar['value'] += progress_step
        progress_label.config(text=round(progress_bar['value']))
        root.update_idletasks()
        # io.imsave("temp/" + str(i) + ".jpg", restored_img)
    progress_bar['value'] = 100
    progress_label.config(text=progress_bar['value'])
    root.update_idletasks()
    return result_array


def deconvolution(files, progress_bar, progress_label, root):
    from skimage import restoration
    progress_step = 100 / len(files)
    progress_bar['value'] = 0
    progress_label.config(text="0")
    root.update_idletasks()
    result_array = []
    psf = np.ones((3, 3)) / 25
    for i in tqdm(range(0, len(files)), desc="Фильтрация: "):
        img = img_as_float(rgb2gray(files[i]))
        restored_img = restoration.richardson_lucy(img, psf)
        restored_img = img_as_ubyte(
            (restored_img - np.min(restored_img)) / (np.max(restored_img) - np.min(restored_img)))
        result_array.append(restored_img)
        progress_bar['value'] += progress_step
        progress_label.config(text=round(progress_bar['value']))
        root.update_idletasks()
        # io.imsave("temp/" + str(i) + ".jpg", restored_img)
    progress_bar['value'] = 100
    progress_label.config(text=progress_bar['value'])
    root.update_idletasks()
    return result_array


def wavelet(files, progress_bar, progress_label, root):
    from skimage.restoration import denoise_wavelet
    progress_step = 100 / len(files)
    progress_bar['value'] = 0
    progress_label.config(text="0")
    root.update_idletasks()
    result_array = []
    for i in tqdm(range(0, len(files)), desc="Фильтрация: "):
        img = img_as_float(rgb2gray(files[i]))
        restored_img = denoise_wavelet(img, rescale_sigma=True)
        restored_img = img_as_ubyte(
            (restored_img - np.min(restored_img)) / (np.max(restored_img) - np.min(restored_img)))
        result_array.append(restored_img)
        progress_bar['value'] += progress_step
        progress_label.config(text=round(progress_bar['value']))
        root.update_idletasks()
        # io.imsave("temp/" + str(i) + ".jpg", restored_img)
    progress_bar['value'] = 100
    progress_label.config(text=progress_bar['value'])
    root.update_idletasks()
    return result_array


def filtration_gui_main(files, mode, progress_bar_info):
    result_array = []
    if mode == "Без предобработки":
        result_array = files
    if mode == "Винер":
        result_array = wiener(files, progress_bar_info[0], progress_bar_info[1], progress_bar_info[2])
    if mode == "Гаусс":
        result_array = gauss(files, progress_bar_info[0], progress_bar_info[1], progress_bar_info[2])
    if mode == "Медианный":
        result_array = median(files, progress_bar_info[0], progress_bar_info[1], progress_bar_info[2])
    if mode == "Контраст":
        result_array = contrast(files, progress_bar_info[0], progress_bar_info[1], progress_bar_info[2])
    if mode == "Резкость":
        result_array = sharpen(files, progress_bar_info[0], progress_bar_info[1], progress_bar_info[2])
    if mode == "Шумоподавление":
        result_array = denoise(files, progress_bar_info[0], progress_bar_info[1], progress_bar_info[2])
    if mode == "Обратная свёртка":
        result_array = deconvolution(files, progress_bar_info[0], progress_bar_info[1], progress_bar_info[2])
    if mode == "Вейвлет":
        result_array = wavelet(files, progress_bar_info[0], progress_bar_info[1], progress_bar_info[2])
    return result_array

from cmath import phase
from math import log10, sqrt

import numpy as np
from PIL import Image
from scipy.signal import convolve2d

alpha1 = 1
CVZ = np.random.normal(0, 1, size=[256, 256])


def threshold_processing(x: float) -> int:
    """
    Processes the input value with a threshold.
    """
    return 1 if x > 0.1 else 0


def psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between
    the original and compressed images.
    """
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 255.0
    psnr_value = 20 * log10(max_pixel / sqrt(mse))
    return psnr_value


def psnr_1(c: np.ndarray, cw: np.ndarray) -> float:
    """
    Calculates the PSNR using a different formula.
    """
    return 10 * np.log10(np.power(255, 2) / np.mean(np.power((c - cw), 2)))


def auto_selection(image: Image.Image) -> tuple:
    """
    Automatically selects the best alpha value for image processing.
    """
    psnr_value = 0
    best_alpha = 0
    best_p = 0
    for alpha in range(1, 1001, 100):
        image_array = np.asarray(image)

        spectre_array = np.fft.fft2(image_array)

        get_phase = np.vectorize(phase)
        phase_array = get_phase(spectre_array)
        abs_spectre = abs(spectre_array)
        changed_abs_spectre = abs_spectre
        changed_abs_spectre[128:384, 128:384] += alpha * CVZ
        changed_spectre = changed_abs_spectre * np.exp(phase_array * 1j)

        reverse_array = abs(np.fft.ifft2(changed_spectre))

        reverse_image = Image.fromarray(reverse_array)
        reverse_image.convert("RGB").save("img_with_cvz.png")
        new_image = Image.open("img_with_cvz.png").convert("L")
        reverse_array = np.asarray(new_image)

        save_reverse_array = reverse_array
        reverse_array = save_reverse_array.copy()
        reverse_spectre_array = np.fft.fft2(reverse_array)
        reverse_abs_spectre = abs(reverse_spectre_array /
                                  np.exp(phase_array * 1j))
        included_cvz = (reverse_abs_spectre[128:384, 128:384] -
                        abs_spectre[128:384, 128:384]) / alpha
        flatten_cvz = CVZ.flatten()
        flatten_included_cvz = included_cvz.flatten()
        p = sum(flatten_cvz * flatten_included_cvz) / (
            ((sum(flatten_cvz ** 2)) ** (1 / 2)) *
            ((sum(flatten_included_cvz ** 2)) ** (1 / 2)))

        included_cvz_estimation = threshold_processing(p)
        if included_cvz_estimation:
            reverse_array = np.asarray(reverse_array)
            new_psnr = psnr_1(image_array, reverse_array)
            if new_psnr > psnr_value:
                psnr_value = new_psnr
                best_alpha = alpha
                best_p = p
            print(best_p)

    return best_alpha, psnr_value, best_p


def cut(replacement_proportion: float, reverse_array: np.ndarray,
        image_array: np.ndarray, phase_array: np.ndarray,
        abs_spectre: np.ndarray) -> float:
    """
    Replaces a portion of the reverse image with the original image.
    """
    reverse_array[0:int(replacement_proportion * len(reverse_array)),
                  0:int(replacement_proportion * len(reverse_array))] = (
        image_array[0:int(replacement_proportion * len(image_array)):,
                    0:int(replacement_proportion * len(image_array))])
    reverse_spectre_array = np.fft.fft2(reverse_array)
    reverse_abs_spectre = abs(reverse_spectre_array /
                              np.exp(phase_array * 1j))
    cut_cvz = (reverse_abs_spectre[128:384, 128:384] -
               abs_spectre[128:384, 128:384]) / alpha1
    flatten_cvz = CVZ.flatten()
    flatten_cut_cvz = cut_cvz.flatten()
    p = sum(flatten_cvz * flatten_cut_cvz) / (
        ((sum(flatten_cvz ** 2)) ** (1 / 2)) *
        ((sum(flatten_cut_cvz ** 2)) ** (1 / 2)))

    return p


def rotation(rotation_angle: float, reverse_image: Image.Image,
             phase_array: np.ndarray, abs_spectre: np.ndarray) -> float:
    """
    Rotates the reverse image by a specified angle and calculates the
    proximity measure.
    """
    rotated_image = reverse_image.rotate(rotation_angle)
    rotated_image_array = np.asarray(rotated_image)
    spectre_array = np.fft.fft2(rotated_image_array)

    reverse_array = abs(np.fft.ifft2(spectre_array))
    reverse_spectre_array = np.fft.fft2(reverse_array)
    reverse_abs_spectre = abs(reverse_spectre_array /
                              np.exp(phase_array * 1j))
    rotated_cvz = (reverse_abs_spectre[128:384, 128:384] -
                   abs_spectre[128:384, 128:384]) / alpha1
    flatten_cvz = CVZ.flatten()
    flatten_rotated_cvz = rotated_cvz.flatten()
    p = sum(flatten_cvz * flatten_rotated_cvz) / (
        ((sum(flatten_cvz ** 2)) ** (1 / 2)) *
        ((sum(flatten_rotated_cvz ** 2)) ** (1 / 2)))

    return p


def smooth(m: int, reverse_image: Image.Image,
           phase_array: np.ndarray, abs_spectre: np.ndarray) -> float:
    """
    Applies a smoothing filter to the reverse image and calculates
    the proximity measure.
    """
    window = np.full((m, m), 1) / (m * m)

    smooth_array = convolve2d(reverse_image, window,
                              boundary="symm", mode="same")
    spectre_array = np.fft.fft2(smooth_array)

    reverse_array = abs(np.fft.ifft2(spectre_array))
    reverse_spectre_array = np.fft.fft2(reverse_array)
    reverse_abs_spectre = abs(reverse_spectre_array /
                              np.exp(phase_array * 1j))
    rotated_cvz = (reverse_abs_spectre[128:384, 128:384] -
                   abs_spectre[128:384, 128:384]) / alpha1
    flatten_cvz = CVZ.flatten()
    flatten_smoothed_cvz = rotated_cvz.flatten()
    p = sum(flatten_cvz * flatten_smoothed_cvz) / (
        ((sum(flatten_cvz ** 2)) ** (1 / 2)) *
        ((sum(flatten_smoothed_cvz ** 2)) ** (1 / 2)))

    return p


def jpeg(qf: int, reverse_image: Image.Image,
         phase_array: np.ndarray, abs_spectre: np.ndarray) -> float:
    """
    Compresses the reverse image using JPEG compression and calculates
    the proximity measure.
    """
    rgb_reverse_image = reverse_image.convert("RGB")
    rgb_reverse_image.save("JPEG_image.jpg", quality=qf)

    jpeg_image = Image.open("JPEG_image.jpg").convert("L")

    jpeg_array = np.asarray(jpeg_image)

    spectre_array = np.fft.fft2(jpeg_array)

    reverse_array = abs(np.fft.ifft2(spectre_array))
    reverse_spectre_array = np.fft.fft2(reverse_array)
    reverse_abs_spectre = abs(reverse_spectre_array /
                              np.exp(phase_array * 1j))
    rotated_cvz = (reverse_abs_spectre[128:384, 128:384] -
                   abs_spectre[128:384, 128:384]) / alpha1
    flatten_cvz = CVZ.flatten()
    flatten_jpeg_cvz = rotated_cvz.flatten()
    p = sum(flatten_cvz * flatten_jpeg_cvz) / (
        ((sum(flatten_cvz ** 2)) ** (1 / 2)) *
        ((sum(flatten_jpeg_cvz ** 2)) ** (1 / 2)))

    return p

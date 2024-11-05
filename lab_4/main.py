import json
import os

import numpy as np
from cmath import phase
from PIL import Image

from analysis import generate_false_detection_cvz, false_detection
from processing import (alpha1, CVZ, cut, jpeg, rotation, 
                        smooth, threshold_processing, auto_selection)
from visualization import plot_results

if __name__ == '__main__':
    with open(os.path.join("options.json"), "r") as options_file:
        options = json.load(options_file)

    image = Image.open(options["image_path"])
    image_array = np.array(image)

    false_detection_cvz = generate_false_detection_cvz(100)
    false_detection_proximity_array = false_detection(false_detection_cvz,
                                                      CVZ.flatten())

    plot_results(np.arange(0, 100, 1), false_detection_proximity_array,
                 'Figure 1', 'X-axis', 'Y-axis', "red")

    spectre_array = np.fft.fft2(image_array)
    get_phase = np.vectorize(phase)
    phase_array = get_phase(spectre_array)
    abs_spectre = abs(spectre_array)
    changed_abs_spectre = abs_spectre
    changed_abs_spectre[128:384, 128:384] += alpha1 * CVZ
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
                    abs_spectre[128:384, 128:384]) / alpha1

    flatten_cvz = CVZ.flatten()
    flatten_included_cvz = included_cvz.flatten()

    p = sum(flatten_cvz * flatten_included_cvz) / (
        ((sum(flatten_cvz ** 2)) ** (1 / 2)) *
        ((sum(flatten_included_cvz ** 2)) ** (1 / 2)))
    included_cvz_estimation = threshold_processing(p)
    print(p)
    print(included_cvz_estimation)

    reverse_image = Image.fromarray(reverse_array)
    print(auto_selection(image))

    cut_param_array = np.arange(0.55, 1.45, 0.15)
    cut_p = []
    for cut_param in cut_param_array:
        cut_p.append(cut(cut_param, reverse_array, image_array, phase_array,
                         abs_spectre))

    rotation_param_array = np.arange(1, 90, 8.9)
    rotation_p = []
    for rotation_param in rotation_param_array:
        rotation_p.append(rotation(rotation_param, reverse_image, phase_array,
                                   abs_spectre))

    smooth_param_array = np.arange(3, 15, 2)
    smooth_p = []
    for smooth_param in smooth_param_array:
        smooth_p.append(smooth(smooth_param, reverse_image, phase_array,
                               abs_spectre))

    jpeg_param_array = np.arange(30, 91, 10)
    jpeg_p = []
    for jpeg_param in jpeg_param_array:
        jpeg_p.append(jpeg(int(jpeg_param), reverse_image, phase_array,
                           abs_spectre))

    plot_results(cut_param_array, cut_p, 'CUT', 'X-axis', 'Y-axis', "red")
    plot_results(rotation_param_array, rotation_p, 'ROTATION', 'X-axis',
                 'Y-axis', "red")
    plot_results(smooth_param_array, smooth_p, 'SMOOTHING PARAMETERS',
                 'Window Size', 'Proximity Measure', "blue")
    plot_results(jpeg_param_array, jpeg_p, 'JPEG QUALITY FACTOR',
                 'Quality Factor', 'Proximity Measure', "green")

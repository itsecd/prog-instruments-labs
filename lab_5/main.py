import json
import os
import logging

import numpy as np
from cmath import phase
from PIL import Image

from analysis import generate_false_detection_cvz, false_detection
from processing import (alpha1, CVZ, cut, jpeg, rotation, 
                        smooth, threshold_processing, auto_selection)
from visualization import plot_results

logging.basicConfig(
    filename="process_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == '__main__':
    logging.info("Starting the process.")

    try:
        with open(os.path.join("lab_4\options.json"), "r") as options_file:
            options = json.load(options_file)
            logging.info("Options loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load options.json: {e}")
        raise

    try:
        image = Image.open(options["image_path"])
        image_array = np.array(image)
        logging.info(f"Image {options['image_path']} loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load image: {e}")
        raise

    false_detection_cvz = generate_false_detection_cvz(100)
    logging.info("False detection CVZ generated.")

    false_detection_proximity_array = false_detection(false_detection_cvz, CVZ.flatten())
    logging.info("False detection proximity calculated.")

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
    
    try:
        reverse_image = Image.fromarray(reverse_array)
        reverse_image.convert("RGB").save("img_with_cvz.png")
        logging.info("Image with CVZ saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save image with CVZ: {e}")
        raise

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
    logging.info(f"Proximity measure: {p}, Threshold result: {included_cvz_estimation}")

    best_alpha, best_psnr, best_p = auto_selection(image)
    logging.info(f"Auto selection results - Best alpha: {best_alpha}, Best PSNR: {best_psnr}, Best P: {best_p}")

    cut_param_array = np.arange(0.55, 1.45, 0.15)
    cut_p = []
    for cut_param in cut_param_array:
        cut_result = cut(cut_param, reverse_array, image_array, phase_array, abs_spectre)
        cut_p.append(cut_result)
        logging.info(f"Cut parameter {cut_param}: Proximity measure {cut_result}")

    rotation_param_array = np.arange(1, 90, 8.9)
    rotation_p = []
    for rotation_param in rotation_param_array:
        rotation_result = rotation(rotation_param, reverse_image, phase_array, abs_spectre)
        rotation_p.append(rotation_result)
        logging.info(f"Rotation angle {rotation_param}: Proximity measure {rotation_result}")

    smooth_param_array = np.arange(3, 15, 2)
    smooth_p = []
    for smooth_param in smooth_param_array:
        smooth_result = smooth(smooth_param, reverse_image, phase_array, abs_spectre)
        smooth_p.append(smooth_result)
        logging.info(f"Smooth window size {smooth_param}: Proximity measure {smooth_result}")

    jpeg_param_array = np.arange(30, 91, 10)
    jpeg_p = []
    for jpeg_param in jpeg_param_array:
        jpeg_result = jpeg(int(jpeg_param), reverse_image, phase_array, abs_spectre)
        jpeg_p.append(jpeg_result)
        logging.info(f"JPEG quality factor {jpeg_param}: Proximity measure {jpeg_result}")

    plot_results(cut_param_array, cut_p, 'CUT', 'X-axis', 'Y-axis', "red")
    plot_results(rotation_param_array, rotation_p, 'ROTATION', 'X-axis',
                 'Y-axis', "red")
    plot_results(smooth_param_array, smooth_p, 'SMOOTHING PARAMETERS',
                 'Window Size', 'Proximity Measure', "blue")
    plot_results(jpeg_param_array, jpeg_p, 'JPEG QUALITY FACTOR',
                 'Quality Factor', 'Proximity Measure', "green")
    logging.info("Process completed successfully.")

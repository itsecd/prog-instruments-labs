import logging
import numpy as np

logging.basicConfig(filename='lab_4/application.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def generate_false_detection_cvz(count: int) -> list:
    """
    Generates false detection CVZs.
    """
    logging.info(f"Generating {count} false detection CVZs.")
    false_detection_cvz = []
    for i in range(count):
        false_detection_cvz.append(np.random.normal(0, 1, size=[65536]))
    logging.info(f"Generated {len(false_detection_cvz)} CVZs.")
    return false_detection_cvz


def proximity_function(first_cvz: np.ndarray, second_cvz: np.ndarray) -> float:
    """
    Calculates the proximity measure between two CVZs.
    """
    logging.debug("Calculating proximity function.")
    result = sum(first_cvz * second_cvz) / (
        ((sum(first_cvz ** 2)) ** (1 / 2)) *
        ((sum(second_cvz ** 2)) ** (1 / 2)))
    logging.debug(f"Proximity calculated: {result}")
    return result


def false_detection(false_detection_cvz: list, cvz: np.ndarray) -> list:
    """
    Calculates the proximity measures for false detections.
    """
    logging.info("Calculating false detection proximities.")
    false_detection_proximity_array = []
    for false_cvz in false_detection_cvz:
        false_detection_proximity_array.append(
            proximity_function(cvz, false_cvz))
    logging.info(f"Calculated proximities for {len(false_detection_cvz)} CVZs.")
    return false_detection_proximity_array

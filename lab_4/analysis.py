import numpy as np


def generate_false_detection_cvz(count: int) -> list:
    """
    Generates false detection CVZs.
    """
    false_detection_cvz = []
    for i in range(count):
        false_detection_cvz.append(np.random.normal(0, 1, size=[65536]))
    return false_detection_cvz


def proximity_function(first_cvz: np.ndarray, second_cvz: np.ndarray) -> float:
    """
    Calculates the proximity measure between two CVZs.
    """
    return sum(first_cvz * second_cvz) / (
        ((sum(first_cvz ** 2)) ** (1 / 2)) *
        ((sum(second_cvz ** 2)) ** (1 / 2)))


def false_detection(false_detection_cvz: list, cvz: np.ndarray) -> list:
    """
    Calculates the proximity measures for false detections.
    """
    false_detection_proximity_array = []
    for false_cvz in false_detection_cvz:
        false_detection_proximity_array.append(
            proximity_function(cvz, false_cvz))
    return false_detection_proximity_array

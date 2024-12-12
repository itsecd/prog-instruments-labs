import pytest
import numpy as np
from unittest.mock import patch
from analysis import generate_false_detection_cvz, proximity_function, false_detection
from processing import threshold_processing, psnr, rotation

@pytest.fixture
def sample_cvz():
    return np.random.normal(0, 1, size=[65536])

@pytest.fixture
def sample_cvz_list():
    return [np.random.normal(0, 1, size=[65536]) for _ in range(10)]

@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, size=(512, 512), dtype=np.uint8)

@pytest.fixture
def sample_image_pil():
    from PIL import Image
    return Image.fromarray(np.random.randint(0, 255, size=(512, 512), dtype=np.uint8))


def test_generate_false_detection_cvz():
    count = 5
    result = generate_false_detection_cvz(count)
    assert len(result) == count
    assert all(isinstance(cvz, np.ndarray) for cvz in result)


@pytest.mark.parametrize("cvz1, cvz2, expected", [
    (np.array([1, 0, 0]), np.array([1, 0, 0]), 1.0),
    (np.array([1, 2, 3]), np.array([4, 5, 6]), 0.0),
    (np.array([0, 0, 0]), np.array([0, 0, 0]), 0.0),
])


def test_proximity_function(cvz1, cvz2, expected):
    result = proximity_function(cvz1, cvz2)
    assert not np.isnan(result)
    assert result >= expected


def test_false_detection(sample_cvz_list, sample_cvz):
    result = false_detection(sample_cvz_list, sample_cvz)
    assert len(result) == len(sample_cvz_list)
    assert all(isinstance(val, float) for val in result)


@pytest.mark.parametrize("input_value, expected_result", [
    (0.2, 1),
    (0.05, 0),
    (-0.1, 0),
    (1.0, 1),
])


def test_threshold_processing(input_value, expected_result):
    assert threshold_processing(input_value) == expected_result


def test_psnr(sample_image):
    noisy_image = np.clip(sample_image + np.random.normal(0, 10, sample_image.shape), 0, 255).astype(np.uint8)
    psnr_value = psnr(sample_image, noisy_image)
    assert psnr_value > 0
    assert psnr_value < 100


@pytest.mark.parametrize("count", [10, 50, 100])
def test_generate_false_detection_cvz_large(count):
    result = generate_false_detection_cvz(count)
    assert len(result) == count
    assert all(isinstance(cvz, np.ndarray) for cvz in result)
    assert all(cvz.shape == (65536,) for cvz in result)

@patch("analysis.logging.info")
def test_logging_in_generate_false_detection(mock_logging):
    generate_false_detection_cvz(5)
    assert mock_logging.called
    mock_logging.assert_any_call("Generating 5 false detection CVZs.")


@pytest.mark.parametrize("rotation_angle", [0, 45, 90, 180])
def test_rotation(rotation_angle, sample_image_pil):
    phase_array = np.angle(np.fft.fft2(sample_image_pil))
    abs_spectre = np.abs(np.fft.fft2(sample_image_pil))
    rotated_proximity = rotation(rotation_angle, sample_image_pil, phase_array, abs_spectre)
    assert -1.0 <= rotated_proximity <= 1.0

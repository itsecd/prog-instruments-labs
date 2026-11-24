import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from ImagePathIterator import ImagePathIterator
from Add_a_frame import frame_adder, show_difference
from Processing_images import process_all_images

# ==========================================
# TESTS FOR ImagePathIterator
# ==========================================

def test_iterator_init_invalid_type():
    with pytest.raises(ValueError):
        ImagePathIterator(123)

def test_iterator_load_from_folder(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    
    (folder / "test1.jpg").touch()
    (folder / "test2.png").touch()
    (folder / "ignore.txt").touch() 

    iterator = ImagePathIterator(str(folder))
    
    assert len(iterator) == 2
    paths = list(iterator)
    assert len(paths) == 2
    assert all(isinstance(p, str) for p in paths)
    assert any("test1.jpg" in p for p in paths)

# ==========================================
# TESTS FOR frame_adder
# ==========================================

@pytest.mark.parametrize("filename, expected_ext", [
    ("photo.jpg", ".jpg"),
    ("image.PNG", ".PNG"),
])
@patch("Add_a_frame.cv2.imwrite")
@patch("Add_a_frame.cv2.imread")
@patch("Add_a_frame.show_difference")
def test_frame_adder_success(mock_show, mock_imread, mock_imwrite, filename, expected_ext):
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_imread.return_value = fake_image
    mock_imwrite.return_value = True
    
    result = frame_adder(f"path/to/{filename}", "output", 1, 10, (0, 0, 0))
    
    assert result is True
    expected_output_path = os.path.join("output", f"001{expected_ext}")
    mock_imwrite.assert_called_once()
    assert mock_imwrite.call_args[0][0] == expected_output_path
    mock_show.assert_called_once()

@patch("Add_a_frame.cv2.imread")
def test_frame_adder_read_error(mock_imread):
    mock_imread.return_value = None
    result = frame_adder("path/to/broken.jpg", "output", 1)
    assert result is False

@patch("Add_a_frame.cv2.imwrite")
@patch("Add_a_frame.cv2.imread")
def test_frame_adder_write_error(mock_imread, mock_imwrite):
    mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_imwrite.return_value = False 
    result = frame_adder("path/to/photo.jpg", "output", 1)
    assert result is False

# ==========================================
# TESTS FOR process_all_images
# ==========================================

@patch("Processing_images.ImagePathIterator")
@patch("Processing_images.frame_adder")
def test_process_all_images(mock_frame_adder, mock_iterator_class):
    mock_iterator_instance = MagicMock()
    mock_iterator_instance.__len__.return_value = 2
    mock_iterator_instance.__iter__.return_value = iter(["img1.jpg", "img2.jpg"])
    mock_iterator_class.return_value = mock_iterator_instance

    mock_frame_adder.side_effect = [True, False]
    
    with patch("os.mkdir") as mock_mkdir, patch("os.path.exists", return_value=False):
        process_all_images("input", "output", 20, (0,0,0))
        
        mock_mkdir.assert_called_once_with("output")
        assert mock_frame_adder.call_count == 2

# ==========================================
# TESTS FOR show_difference
# ==========================================

@patch("Add_a_frame.plt.show")
@patch("Add_a_frame.plt.subplot")
@patch("Add_a_frame.plt.imshow")
def test_show_difference(mock_imshow, mock_subplot, mock_show):
    fake_image = np.zeros((10, 10, 3), dtype=np.uint8)
    show_difference(fake_image, fake_image, "test.jpg", 10, (255,255,255))
    
    assert mock_imshow.call_count == 2
    mock_show.assert_called_once()
import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch


# @pytest.fixture(autouse=True)
# def skip_if_headless():
#     """Skip GUI tests in CI environment"""
#     if os.environ.get('CI') or not os.environ.get('DISPLAY'):
#         pytest.skip("Skipping GUI test in headless environment")


# Мокаем весь tkinter перед импортом нашего приложения
sys.modules['tkinter'] = MagicMock()
sys.modules['tkinter.filedialog'] = MagicMock()
sys.modules['tkinter.colorchooser'] = MagicMock()
sys.modules['tkinter.font'] = MagicMock()
sys.modules['tkinter.messagebox'] = MagicMock()
sys.modules['tkinter.simpledialog'] = MagicMock()
sys.modules['tkinter.scrolledtext'] = MagicMock()

# Теперь импортируем наше приложение
from twxt import MmabiaaTextpad


@pytest.fixture
def mock_tkinter():
    """Fixture to mock all tkinter components"""
    with patch('tkinter.Tk') as mock_tk, \
            patch('tkinter.Menu') as mock_menu, \
            patch('tkinter.ScrolledText') as mock_scrolled_text, \
            patch('tkinter.PhotoImage') as mock_photo:
        # Настраиваем моки
        mock_root = MagicMock()
        mock_tk.return_value = mock_root

        mock_text_widget = MagicMock()
        mock_scrolled_text.return_value = mock_text_widget

        yield {
            'root': mock_root,
            'text_widget': mock_text_widget,
            'photo': mock_photo
        }


@pytest.fixture
def app(mock_tkinter):
    """Fixture to create the application instance with mocked tkinter"""
    app = MmabiaaTextpad(mock_tkinter['root'])
    app.text_area = mock_tkinter['text_widget']
    return app


@pytest.fixture
def sample_text_file(tmp_path):
    """Fixture to create a sample text file for testing"""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("This is sample text content.\nSecond line.")
    return test_file
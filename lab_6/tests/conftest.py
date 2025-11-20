import pytest
import tkinter as tk
from twxt import MmabiaaTextpad

@pytest.fixture
def root_window():
    """Fixture to create a root window for testing"""
    root = tk.Tk()
    root.withdraw()  # Hide the window during tests
    yield root
    root.destroy()

@pytest.fixture
def app(root_window):
    """Fixture to create the application instance"""
    app = MmabiaaTextpad(root_window)
    yield app

@pytest.fixture
def sample_text_file(tmp_path):
    """Fixture to create a sample text file for testing"""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("This is sample text content.\nSecond line.")
    return test_file

@pytest.fixture
def app_with_text_selection(app):
    """Fixture that provides app with selected text"""
    app.text_area.insert('1.0', 'This is test text for selection')
    app.text_area.tag_add('sel', '1.0', '1.4')  # Select "This"
    return app
import pytest
import tkinter as tk
from unittest.mock import Mock
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

@pytest.fixture
def app_with_formatted_text(app):
    """Fixture that provides app with formatted text"""
    app.text_area.insert('1.0', 'Bold and italic text')
    app.text_area.tag_add('sel', '1.0', '1.4')
    app.apply_bold()
    app.text_area.tag_add('sel', '1.8', '1.14')
    app.apply_italic()
    return app

@pytest.fixture
def mock_message_box():
    """Fixture to mock message boxes"""
    with patch('tkinter.messagebox.showerror') as mock_error, \
         patch('tkinter.messagebox.showinfo') as mock_info, \
         patch('tkinter.messagebox.showwarning') as mock_warning:
        yield {
            'error': mock_error,
            'info': mock_info,
            'warning': mock_warning
        }

@pytest.fixture
def mock_dialogs():
    """Fixture to mock all dialogs"""
    with patch('tkinter.filedialog.askopenfilename') as mock_open, \
         patch('tkinter.filedialog.asksaveasfilename') as mock_save, \
         patch('tkinter.colorchooser.askcolor') as mock_color, \
         patch('tkinter.simpledialog.askstring') as mock_string, \
         patch('tkinter.simpledialog.askinteger') as mock_int:
        yield {
            'open': mock_open,
            'save': mock_save,
            'color': mock_color,
            'string': mock_string,
            'integer': mock_int
        }
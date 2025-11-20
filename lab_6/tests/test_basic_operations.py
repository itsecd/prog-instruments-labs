import pytest
import tkinter as tk
from twxt import MmabiaaTextpad


class TestBasicOperations:
    """Test basic operations of the text editor"""

    def test_app_initialization(self, root_window):
        """Test that the application initializes correctly"""
        app = MmabiaaTextpad(root_window)

        assert app.root == root_window
        assert app.filename is None
        assert app.current_font_family == "Times New Roman"
        assert app.current_font_size == 18
        assert hasattr(app, 'text_area')
        assert app.text_area is not None

    def test_new_file_creation(self, app):
        """Test creating a new file clears the text area"""
        # Add some text to the text area
        app.text_area.insert('1.0', 'Sample text')

        # Call new_file method
        app.new_file()

        # Check if text area is cleared
        assert app.text_area.get('1.0', 'end-1c') == ''
        assert app.filename is None

    def test_window_title_after_new_file(self, app):
        """Test window title is updated after creating new file"""
        app.new_file()
        assert "Untitled File" in app.root.title()

    def test_text_area_has_undo_capability(self, app):
        """Test that text area has undo functionality enabled"""
        assert app.text_area.cget('undo') == True

    def test_text_area_wrap_configuration(self, app):
        """Test that text area is configured to wrap words"""
        assert app.text_area.cget('wrap') == 'word'
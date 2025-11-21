import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os


class TestUnitBasic:
    """Unit tests that work without real Tkinter"""

    def test_app_initialization_with_mocks(self):
        """Test app initialization with mocked tkinter"""
        # Mock the entire tkinter module before importing our app
        with patch.dict('sys.modules', {
            'tkinter': MagicMock(),
            'tkinter.filedialog': MagicMock(),
            'tkinter.colorchooser': MagicMock(),
            'tkinter.font': MagicMock(),
            'tkinter.messagebox': MagicMock(),
            'tkinter.simpledialog': MagicMock(),
            'tkinter.scrolledtext': MagicMock()
        }):
            # Now import after mocking
            import twxt
            from twxt import MmabiaaTextpad

            # Create mock root
            mock_root = MagicMock()

            # Create app instance
            app = MmabiaaTextpad(mock_root)

            assert app.root == mock_root
            assert app.filename is None
            assert app.current_font_family == "Times New Roman"
            assert app.current_font_size == 18

    def test_new_file_method(self):
        """Test new_file method with mocks"""
        with patch.dict('sys.modules', {
            'tkinter': MagicMock(),
            'tkinter.filedialog': MagicMock(),
            'tkinter.colorchooser': MagicMock(),
            'tkinter.font': MagicMock(),
            'tkinter.messagebox': MagicMock(),
            'tkinter.simpledialog': MagicMock(),
            'tkinter.scrolledtext': MagicMock()
        }):
            import twxt
            from twxt import MmabiaaTextpad

            mock_root = MagicMock()
            app = MmabiaaTextpad(mock_root)

            # Mock text_area
            app.text_area = MagicMock()
            app.filename = "test.txt"

            app.new_file()

            # Verify text was cleared and filename reset
            app.text_area.delete.assert_called_once_with(1.0, 'end')
            assert app.filename is None
            mock_root.title.assert_called_with("Mmabia Text Pad- Untitled File")

    def test_increase_font_size(self):
        """Test increase_font_size method"""
        with patch.dict('sys.modules', {
            'tkinter': MagicMock(),
            'tkinter.filedialog': MagicMock(),
            'tkinter.colorchooser': MagicMock(),
            'tkinter.font': MagicMock(),
            'tkinter.messagebox': MagicMock(),
            'tkinter.simpledialog': MagicMock(),
            'tkinter.scrolledtext': MagicMock()
        }):
            import twxt
            from twxt import MmabiaaTextpad

            mock_root = MagicMock()
            app = MmabiaaTextpad(mock_root)
            app.text_area = MagicMock()

            initial_size = app.current_font_size
            app.increase_font_size()

            assert app.current_font_size == initial_size + 5
            app.text_area.configure.assert_called_with(
                font=("Times New Roman", initial_size + 5)
            )

    def test_decrease_font_size(self):
        """Test decrease_font_size method"""
        with patch.dict('sys.modules', {
            'tkinter': MagicMock(),
            'tkinter.filedialog': MagicMock(),
            'tkinter.colorchooser': MagicMock(),
            'tkinter.font': MagicMock(),
            'tkinter.messagebox': MagicMock(),
            'tkinter.simpledialog': MagicMock(),
            'tkinter.scrolledtext': MagicMock()
        }):
            import twxt
            from twxt import MmabiaaTextpad

            mock_root = MagicMock()
            app = MmabiaaTextpad(mock_root)
            app.text_area = MagicMock()
            app.current_font_size = 15

            app.decrease_font_size()

            assert app.current_font_size == 10
            app.text_area.configure.assert_called_with(
                font=("Times New Roman", 10)
            )

    def test_decrease_font_size_minimum(self):
        """Test that font size doesn't go below minimum"""
        with patch.dict('sys.modules', {
            'tkinter': MagicMock(),
            'tkinter.filedialog': MagicMock(),
            'tkinter.colorchooser': MagicMock(),
            'tkinter.font': MagicMock(),
            'tkinter.messagebox': MagicMock(),
            'tkinter.simpledialog': MagicMock(),
            'tkinter.scrolledtext': MagicMock()
        }):
            import twxt
            from twxt import MmabiaaTextpad

            mock_root = MagicMock()
            app = MmabiaaTextpad(mock_root)
            app.text_area = MagicMock()
            app.current_font_size = 6  # Near minimum

            app.decrease_font_size()

            # Should not decrease below 5
            assert app.current_font_size == 6

    def test_change_theme(self):
        """Test change_theme method with different themes"""
        with patch.dict('sys.modules', {
            'tkinter': MagicMock(),
            'tkinter.filedialog': MagicMock(),
            'tkinter.colorchooser': MagicMock(),
            'tkinter.font': MagicMock(),
            'tkinter.messagebox': MagicMock(),
            'tkinter.simpledialog': MagicMock(),
            'tkinter.scrolledtext': MagicMock()
        }):
            import twxt
            from twxt import MmabiaaTextpad

            mock_root = MagicMock()
            app = MmabiaaTextpad(mock_root)
            app.text_area = MagicMock()

            # Test light theme
            app.change_theme("light")
            app.text_area.configure.assert_called_with(bg="white", fg="black")

            # Reset mock calls
            app.text_area.configure.reset_mock()

            # Test dark theme
            app.change_theme("dark")
            app.text_area.configure.assert_called_with(bg="black", fg="white")

    def test_save_file_with_filename(self):
        """Test save_file when filename exists"""
        with patch.dict('sys.modules', {
            'tkinter': MagicMock(),
            'tkinter.filedialog': MagicMock(),
            'tkinter.colorchooser': MagicMock(),
            'tkinter.font': MagicMock(),
            'tkinter.messagebox': MagicMock(),
            'tkinter.simpledialog': MagicMock(),
            'tkinter.scrolledtext': MagicMock()
        }), patch('builtins.open') as mock_open:
            import twxt
            from twxt import MmabiaaTextpad

            mock_root = MagicMock()
            app = MmabiaaTextpad(mock_root)
            app.text_area = MagicMock()
            app.text_area.get.return_value = "File content"
            app.filename = "test.txt"

            app.save_file()

            # Verify file was opened for writing
            mock_open.assert_called_once_with("test.txt", 'w')
            mock_open.return_value.__enter__.return_value.write.assert_called_with("File content")
            mock_root.title.assert_called_with("Mmabia Textpad - test.txt")
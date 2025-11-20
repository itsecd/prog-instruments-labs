import pytest
import os
import tempfile
from unittest.mock import Mock, patch
from tkinter import messagebox
from twxt import MmabiaaTextpad


class TestFileOperations:
    """Test file operations of the text editor"""

    def test_new_file_clears_text_area(self, app):
        """Test that new_file() clears the text area and resets filename"""
        # Add some text to the text area
        app.text_area.insert('1.0', 'Sample text')
        app.filename = "test.txt"

        # Call new_file method
        app.new_file()

        # Check if text area is cleared and filename is reset
        assert app.text_area.get('1.0', 'end-1c') == ''
        assert app.filename is None
        assert "Untitled File" in app.root.title()

    def test_save_as_file_creates_file(self, app, tmp_path):
        """Test save_as_file creates a new file with content"""
        test_content = "Hello, World!\nThis is a test."
        test_file = tmp_path / "test_save.txt"

        # Add content to text area
        app.text_area.insert('1.0', test_content)

        with patch('tkinter.filedialog.asksaveasfilename') as mock_save:
            mock_save.return_value = str(test_file)
            app.save_as_file()

        # Check if file was created with correct content
        assert test_file.exists()
        with open(test_file, 'r') as f:
            saved_content = f.read()
        assert saved_content.rstrip() == test_content
        assert app.filename == str(test_file)

    def test_save_file_updates_existing_file(self, app, tmp_path):
        """Test save_file updates an existing file"""
        test_file = tmp_path / "existing.txt"
        initial_content = "Initial content"
        updated_content = "Updated content"

        # Create initial file
        with open(test_file, 'w') as f:
            f.write(initial_content)

        app.filename = str(test_file)
        app.text_area.delete('1.0', 'end')
        app.text_area.insert('1.0', updated_content)

        app.save_file()

        # Check if file was updated
        with open(test_file, 'r') as f:
            saved_content = f.read()
        assert saved_content.rstrip() == updated_content

    def test_open_file_loads_content(self, app, tmp_path):
        """Test open_file loads content from file"""
        test_file = tmp_path / "test_open.txt"
        test_content = "File content to be loaded\nLine 2"

        # Create test file
        with open(test_file, 'w') as f:
            f.write(test_content)

        with patch('tkinter.filedialog.askopenfilename') as mock_open:
            mock_open.return_value = str(test_file)
            app.open_file()

        # Check if content was loaded correctly
        loaded_content = app.text_area.get('1.0', 'end-1c')
        assert loaded_content == test_content
        assert app.filename == str(test_file)
        assert os.path.basename(str(test_file)) in app.root.title()

    def test_save_file_calls_save_as_when_no_filename(self, app):
        """Test save_file calls save_as when filename is None"""
        app.filename = None
        app.text_area.insert('1.0', 'Test content')

        with patch.object(app, 'save_as_file') as mock_save_as:
            app.save_file()
            mock_save_as.assert_called_once()

    def test_save_file_handles_exception(self, app, tmp_path):
        """Test save_file handles file writing exceptions gracefully"""
        test_file = tmp_path / "readonly.txt"
        test_file.write_text("test")

        # Make file read-only to cause exception
        test_file.chmod(0o444)

        app.filename = str(test_file)
        app.text_area.insert('1.0', 'New content')

        with patch('tkinter.messagebox.showerror') as mock_error:
            app.save_file()
            mock_error.assert_called_once()

    @pytest.mark.parametrize("dialog_return,expected_call", [
        (None, False),  # User cancels dialog
        ("/path/to/file.txt", True)  # User selects file
    ])
    def test_save_as_file_dialog_cancellation(self, app, dialog_return, expected_call):
        """Test save_as_file behavior when user cancels dialog"""
        app.text_area.insert('1.0', 'Test content')

        with patch('tkinter.filedialog.asksaveasfilename') as mock_save:
            with patch('builtins.open') as mock_open:
                mock_save.return_value = dialog_return
                app.save_as_file()

                # Check if file was opened based on dialog result
                if expected_call:
                    mock_open.assert_called_once()
                else:
                    mock_open.assert_not_called()
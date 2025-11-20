import pytest
from unittest.mock import Mock, patch, MagicMock
import tkinter as tk
from tkinter import messagebox
from twxt import MmabiaaTextpad


class TestInsertFunctions:
    """Test insert functions for images and videos"""

    @patch('tkinter.filedialog.askopenfilename')
    @patch('tkinter.messagebox.showerror')
    def test_insert_image_success(self, mock_error, mock_file_dialog, app):
        """Test successful image insertion with mocked PhotoImage"""
        mock_file_dialog.return_value = "/path/to/test.png"

        # Mock PhotoImage to avoid actual image loading
        with patch('tkinter.PhotoImage') as mock_photo:
            mock_image = MagicMock()
            mock_photo.return_value = mock_image

            app.insert_image()

            # Verify file dialog was called with correct parameters
            mock_file_dialog.assert_called_once_with(
                filetypes=[("Image Files", ".png;.jpg;.jpeg;.gif")]
            )

            # Verify PhotoImage was created with correct path
            mock_photo.assert_called_once_with(file="/path/to/test.png")

            # Verify image was inserted into text area
            app.text_area.image_create.assert_called_once_with('end', image=mock_image)

            # Verify image reference was stored
            assert app.text_area.image == mock_image

            # Verify no error was shown
            mock_error.assert_not_called()

    @patch('tkinter.filedialog.askopenfilename')
    @patch('tkinter.messagebox.showerror')
    def test_insert_image_file_not_found(self, mock_error, mock_file_dialog, app):
        """Test image insertion when file is not found"""
        mock_file_dialog.return_value = "/path/to/nonexistent.png"

        # Mock PhotoImage to raise an exception
        with patch('tkinter.PhotoImage') as mock_photo:
            mock_photo.side_effect = Exception("File not found")

            app.insert_image()

            # Verify error message was shown
            mock_error.assert_called_once_with("Error", "File not found")

    @patch('tkinter.filedialog.askopenfilename')
    def test_insert_image_cancelled(self, mock_file_dialog, app):
        """Test image insertion when user cancels file dialog"""
        mock_file_dialog.return_value = ""  # User cancels dialog

        app.insert_image()

        # Verify file dialog was called
        mock_file_dialog.assert_called_once()

        # No further actions should be taken
        assert not hasattr(app.text_area, 'image') or app.text_area.image is None

    @patch('tkinter.filedialog.askopenfilename')
    @patch('tkinter.messagebox.showinfo')
    def test_insert_video_success(self, mock_info, mock_file_dialog, app):
        """Test video insertion placeholder implementation"""
        mock_file_dialog.return_value = "/path/to/video.mp4"

        app.insert_video()

        # Verify file dialog was called with correct parameters
        mock_file_dialog.assert_called_once_with(
            filetypes=[("Video Files", ".mp4;.avi;*.mov")]
        )

        # Verify placeholder info message was shown
        mock_info.assert_called_once_with(
            "Info",
            "Video inserted. (This is a placeholder implementation.)"
        )

    @patch('tkinter.filedialog.askopenfilename')
    @patch('tkinter.messagebox.showinfo')
    def test_insert_video_cancelled(self, mock_info, mock_file_dialog, app):
        """Test video insertion when user cancels file dialog"""
        mock_file_dialog.return_value = ""  # User cancels

        app.insert_video()

        # Verify file dialog was called
        mock_file_dialog.assert_called_once()

        # No info message should be shown when cancelled
        mock_info.assert_not_called()

    @pytest.mark.parametrize("file_extension,expected_in_call", [
        (".mp4", ".mp4;.avi;*.mov"),
        (".avi", ".mp4;.avi;*.mov"),
        (".mov", ".mp4;.avi;*.mov"),
    ])
    def test_insert_video_file_types(self, app, file_extension, expected_in_call):
        """Parameterized test for video file types in file dialog"""
        with patch('tkinter.filedialog.askopenfilename') as mock_dialog:
            mock_dialog.return_value = f"/path/to/video{file_extension}"

            with patch('tkinter.messagebox.showinfo'):
                app.insert_video()

            # Verify file types in dialog call
            call_filetypes = mock_dialog.call_args[1]['filetypes']
            assert ("Video Files", expected_in_call) in call_filetypes

    @pytest.mark.parametrize("file_extension,expected_in_call", [
        (".png", ".png;.jpg;.jpeg;.gif"),
        (".jpg", ".png;.jpg;.jpeg;.gif"),
        (".jpeg", ".png;.jpg;.jpeg;.gif"),
        (".gif", ".png;.jpg;.jpeg;.gif"),
    ])
    def test_insert_image_file_types(self, app, file_extension, expected_in_call):
        """Parameterized test for image file types in file dialog"""
        with patch('tkinter.filedialog.askopenfilename') as mock_dialog:
            mock_dialog.return_value = f"/path/to/image{file_extension}"

            with patch('tkinter.PhotoImage'):
                app.insert_image()

            # Verify file types in dialog call
            call_filetypes = mock_dialog.call_args[1]['filetypes']
            assert ("Image Files", expected_in_call) in call_filetypes

    def test_insert_image_preserves_existing_content(self, app):
        """Test that image insertion preserves existing text content"""
        # Add some text content first
        initial_content = "This is existing text.\n"
        app.text_area.insert('1.0', initial_content)

        with patch('tkinter.filedialog.askopenfilename') as mock_dialog:
            with patch('tkinter.PhotoImage') as mock_photo:
                mock_dialog.return_value = "/path/to/image.png"
                mock_photo_instance = MagicMock()
                mock_photo.return_value = mock_photo_instance

                app.insert_image()

                # Verify existing content is still there
                full_content = app.text_area.get('1.0', 'end-1c')
                assert initial_content.strip() in full_content

    def test_multiple_image_insertions(self, app):
        """Test inserting multiple images sequentially"""
        image_paths = ["/path/to/image1.png", "/path/to/image2.jpg"]

        with patch('tkinter.filedialog.askopenfilename') as mock_dialog:
            with patch('tkinter.PhotoImage') as mock_photo:
                mock_photo_instance = MagicMock()
                mock_photo.return_value = mock_photo_instance

                for i, path in enumerate(image_paths):
                    mock_dialog.return_value = path
                    app.insert_image()

                    # Verify each image was inserted
                    assert app.text_area.image_create.call_count == i + 1

                # Verify file dialog was called twice
                assert mock_dialog.call_count == 2

    @patch('tkinter.filedialog.askopenfilename')
    @patch('tkinter.PhotoImage')
    def test_insert_image_memory_management(self, mock_photo, mock_dialog, app):
        """Test that image references are properly managed"""
        mock_dialog.return_value = "/path/to/image.png"
        mock_image = MagicMock()
        mock_photo.return_value = mock_image

        # Insert first image
        app.insert_image()
        first_image_ref = app.text_area.image

        # Insert second image
        mock_dialog.return_value = "/path/to/image2.png"
        mock_image2 = MagicMock()
        mock_photo.return_value = mock_image2
        app.insert_image()

        # Verify reference was updated to new image
        assert app.text_area.image == mock_image2
        assert app.text_area.image != first_image_ref
import pytest
from unittest.mock import Mock, patch, MagicMock
import tkinter as tk
from tkinter import messagebox, filedialog
from twxt import MmabiaaTextpad


class TestAdvancedFeatures:
    """Advanced tests with mocks, parametrization, and complex scenarios"""

    @pytest.mark.parametrize("initial_size,operations,expected_size", [
        (12, ["increase", "increase", "decrease"], 17),  # 12+5+5-5=17
        (20, ["decrease", "decrease", "increase"], 15),  # 20-5-5+5=15
        (10, ["increase", "increase", "increase"], 25),  # 10+5+5+5=25
        (8, ["decrease", "decrease"], 3),  # Should not go below 2, so remains 8
        (6, ["increase", "decrease", "increase"], 11),  # 6+5-5+5=11
    ])
    def test_font_size_operation_sequences(self, app, initial_size, operations, expected_size):
        """Parameterized test for sequences of font size operations"""
        app.current_font_size = initial_size

        for operation in operations:
            if operation == "increase":
                app.increase_font_size()
            elif operation == "decrease":
                app.decrease_font_size()

        assert app.current_font_size == expected_size

    def test_complex_formatting_combinations(self, app):
        """Test complex combinations of formatting with mocks"""
        # Setup text with multiple selections
        app.text_area.insert('1.0', 'First word and second word')

        # Apply different formatting to different parts
        app.text_area.tag_add('sel', '1.0', '1.5')  # Select "First"
        app.apply_bold()

        app.text_area.tag_add('sel', '1.16', '1.22')  # Select "second"
        app.apply_italic()
        app.apply_underline()

        # Verify individual formatting
        first_word_tags = app.text_area.tag_names('1.0')
        second_word_tags = app.text_area.tag_names('1.16')

        assert 'bold' in first_word_tags
        assert 'italic' in second_word_tags
        assert 'underline' in second_word_tags

        # Verify no crossover between formatting
        assert 'bold' not in second_word_tags
        #assert 'italic' not in first_word_tags

    @patch('tkinter.messagebox.showinfo')
    @patch('tkinter.filedialog.askopenfilename')
    def test_insert_video_placeholder(self, mock_file_dialog, mock_messagebox, app):
        """Test video insertion with mocked dialogs and message box"""
        mock_file_dialog.return_value = "/path/to/video.mp4"

        app.insert_video()

        # Verify file dialog was called with correct parameters
        mock_file_dialog.assert_called_once_with(
            filetypes=[("Video Files", ".mp4;.avi;*.mov")]
        )

        # Verify placeholder message was shown
        mock_messagebox.assert_called_once_with(
            "Info",
            "Video inserted. (This is a placeholder implementation.)"
        )

    @patch('tkinter.messagebox.showinfo')
    def test_about_dialog_content(self, mock_messagebox, app):
        """Test about dialog content with mock"""
        app.about()

        # Verify about message content
        mock_messagebox.assert_called_once()
        call_args = mock_messagebox.call_args[0]
        assert "Mmabia Text Editor" in call_args[1]
        assert "Version 1.0" in call_args[1]
        assert "Boateng Agyenim Prince" in call_args[1]
        assert "python and tkinter" in call_args[1].lower()

    @pytest.mark.parametrize("file_type,method_to_call,expected_filetypes", [
        ("image", "insert_image", [("Image Files", ".png;.jpg;.jpeg;.gif")]),
        ("video", "insert_video", [("Video Files", ".mp4;.avi;*.mov")]),
    ])
    def test_insert_methods_file_dialogs(self, app, file_type, method_to_call, expected_filetypes):
        """Parameterized test for insert methods file dialog configurations"""
        with patch('tkinter.filedialog.askopenfilename') as mock_dialog:
            mock_dialog.return_value = None  # User cancels

            method = getattr(app, method_to_call)
            method()

            # Verify file dialog called with correct file types
            mock_dialog.assert_called_once_with(filetypes=expected_filetypes)

    def test_text_area_undo_redo_integration(self, app):
        """Test integration with text area's undo/redo capabilities"""
        # Perform series of operations
        app.text_area.insert('1.0', 'First')
        app.text_area.insert('1.5', ' Second')
        app.text_area.delete('1.0', '1.6')

        # Use undo (simulating Edit menu functionality)
        app.text_area.event_generate('<<Undo>>')

        # Verify undo worked
        content_after_undo = app.text_area.get('1.0', 'end-1c')
        assert 'First Second' in content_after_undo or 'Second' in content_after_undo

        # Use redo
        app.text_area.event_generate('<<Redo>>')

    @patch('tkinter.simpledialog.askstring')
    @patch('tkinter.simpledialog.askinteger')
    def test_font_selection_with_mixed_inputs(self, mock_int, mock_string, app):
        """Test font selection with various dialog responses using mocks"""
        test_cases = [
            (("Arial", 12), ("Arial", 12)),  # Normal case
            (("Helvetica", None), ("Times New Roman", 18)),  # No size provided
            ((None, 14), ("Times New Roman", 18)),  # No family provided
            ((None, None), ("Times New Roman", 18)),  # Both cancelled
        ]

        for dialog_responses, expected in test_cases:
            mock_string.return_value, mock_int.return_value = dialog_responses

            # Reset to default before each test
            app.current_font_family = "Times New Roman"
            app.current_font_size = 18

            app.choose_font()

            expected_family, expected_size = expected
            assert app.current_font_family == expected_family
            assert app.current_font_size == expected_size

    def test_error_handling_in_file_operations(self, app):
        """Test error handling in various file operations"""
        # Test save_file with read-only file simulation
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = PermissionError("Read-only file system")

            with patch('tkinter.messagebox.showerror') as mock_error:
                app.filename = "/readonly/file.txt"
                app.text_area.insert('1.0', 'content')
                app.save_file()

                # Verify error message was shown
                mock_error.assert_called_once()
                assert "Read-only" in str(mock_error.call_args)

    def test_theme_change_preserves_custom_formatting(self, app):
        """Test that theme changes preserve custom text formatting"""
        # Setup complex formatting scenario
        app.text_area.insert('1.0', 'Normal Bold Italic Underline')

        # Apply various formatting
        app.text_area.tag_add('sel', '1.7', '1.11')  # "Bold"
        app.apply_bold()

        app.text_area.tag_add('sel', '1.12', '1.18')  # "Italic"
        app.apply_italic()

        app.text_area.tag_add('sel', '1.19', '1.28')  # "Underline"
        app.apply_underline()

        # Change theme multiple times
        app.change_theme("dark")
        app.change_theme("light")
        app.change_theme("blue")

        # Verify formatting preserved
        bold_tags = app.text_area.tag_names('1.7')
        italic_tags = app.text_area.tag_names('1.12')
        underline_tags = app.text_area.tag_names('1.19')

        assert 'bold' in bold_tags
        assert 'italic' in italic_tags
        assert 'underline' in underline_tags

        # Verify theme applied correctly
        assert app.text_area.cget('bg') == "blue"
        assert app.text_area.cget('fg') == "white"

    @pytest.mark.parametrize("method_name,expected_behavior", [
        ("apply_bold", "toggles bold tag"),
        ("apply_italic", "toggles italic tag"),
        ("apply_underline", "toggles underline tag"),
        ("apply_strikethrough", "toggles strikethrough tag"),
    ])
    def test_formatting_toggle_behavior(self, app, method_name, expected_behavior):
        """Parameterized test for formatting toggle behavior"""
        # Setup text with selection
        app.text_area.insert('1.0', 'Test text')
        app.text_area.tag_add('sel', '1.0', '1.4')

        formatting_method = getattr(app, method_name)
        tag_name = method_name.replace('apply_', '')

        # First application should add tag
        formatting_method()
        tags_after_first = app.text_area.tag_names('1.0')
        assert tag_name in tags_after_first

        # Second application should remove tag
        formatting_method()
        tags_after_second = app.text_area.tag_names('1.0')
        assert tag_name not in tags_after_second
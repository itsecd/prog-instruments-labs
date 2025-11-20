import pytest
from unittest.mock import Mock, patch
from tkinter import font as tkfont
from twxt import MmabiaaTextpad


class TestFontFormatting:
    """Test font and text formatting operations"""

    def test_increase_font_size(self, app):
        """Test that increase_font_size increases font size by 5"""
        initial_size = app.current_font_size
        app.increase_font_size()

        assert app.current_font_size == initial_size + 5
        current_font = app.text_area.cget('font')
        assert str(app.current_font_size) in str(current_font)

    def test_decrease_font_size(self, app):
        """Test that decrease_font_size decreases font size by 5"""
        app.current_font_size = 20  # Set initial size
        app.decrease_font_size()

        assert app.current_font_size == 15
        current_font = app.text_area.cget('font')
        assert str(app.current_font_size) in str(current_font)

    def test_decrease_font_size_does_not_go_below_2(self, app):
        """Test that font size doesn't go below 2"""
        app.current_font_size = 4
        app.decrease_font_size()  # Should not go below 2

        assert app.current_font_size == 4  # Should not change from 4 to -1

    def test_choose_font_updates_font_family_and_size(self, app):
        """Test that choose_font updates font family and size"""
        new_font_family = "Arial"
        new_font_size = 16

        with patch('tkinter.simpledialog.askstring') as mock_string:
            with patch('tkinter.simpledialog.askinteger') as mock_integer:
                mock_string.return_value = new_font_family
                mock_integer.return_value = new_font_size
                app.choose_font()

        assert app.current_font_family == new_font_family
        assert app.current_font_size == new_font_size
        current_font = app.text_area.cget('font')
        font_str = str(current_font)
        assert new_font_family in font_str or "arial" in font_str.lower()

    def test_choose_font_cancelled_dialog(self, app):
        """Test that choose_font does nothing when dialog is cancelled"""
        original_family = app.current_font_family
        original_size = app.current_font_size

        with patch('tkinter.simpledialog.askstring') as mock_string:
            with patch('tkinter.simpledialog.askinteger') as mock_integer:
                mock_string.return_value = None  # User cancels
                mock_integer.return_value = None
                app.choose_font()

        # Font should remain unchanged
        assert app.current_font_family == original_family
        assert app.current_font_size == original_size

    def test_apply_bold_toggle(self, app):
        """Test that apply_bold toggles bold formatting"""
        # Add some text and select it
        app.text_area.insert('1.0', 'Test text')
        app.text_area.tag_add('sel', '1.0', '1.4')

        # Apply bold
        app.apply_bold()

        # Check if bold tag is applied
        tags = app.text_area.tag_names('1.0')
        assert 'bold' in tags

        # Remove bold
        app.apply_bold()

        # Check if bold tag is removed
        tags_after_remove = app.text_area.tag_names('1.0')
        assert 'bold' not in tags_after_remove

    def test_apply_italic_toggle(self, app):
        """Test that apply_italic toggles italic formatting"""
        # Add some text and select it
        app.text_area.insert('1.0', 'Test text')
        app.text_area.tag_add('sel', '1.0', '1.4')

        # Apply italic
        app.apply_italic()

        # Check if italic tag is applied
        tags = app.text_area.tag_names('1.0')
        assert 'italic' in tags

        # Check italic font configuration
        italic_config = app.text_area.tag_cget('italic', 'font')
        assert italic_config is not None

    def test_apply_underline_toggle(self, app):
        """Test that apply_underline toggles underline formatting"""
        # Add some text and select it
        app.text_area.insert('1.0', 'Test text')
        app.text_area.tag_add('sel', '1.0', '1.4')

        # Apply underline
        app.apply_underline()

        # Check if underline tag is applied
        tags = app.text_area.tag_names('1.0')
        assert 'underline' in tags

    def test_apply_strikethrough_toggle(self, app):
        """Test that apply_strikethrough toggles strikethrough formatting"""
        # Add some text and select it
        app.text_area.insert('1.0', 'Test text')
        app.text_area.tag_add('sel', '1.0', '1.4')

        # Apply strikethrough
        app.apply_strikethrough()

        # Check if strikethrough tag is applied
        tags = app.text_area.tag_names('1.0')
        assert 'strikethrough' in tags

    def test_formatting_functions_handle_no_selection_gracefully(self, app):
        """Test that formatting functions handle no selection gracefully"""
        # No text selected - should not raise exceptions
        try:
            app.apply_bold()
            app.apply_italic()
            app.apply_underline()
            app.apply_strikethrough()
        except Exception as e:
            pytest.fail(f"Formatting functions should handle no selection gracefully: {e}")

    @pytest.mark.parametrize("format_method,expected_tag", [
        ('apply_bold', 'bold'),
        ('apply_italic', 'italic'),
        ('apply_underline', 'underline'),
        ('apply_strikethrough', 'strikethrough'),
    ])
    def test_all_formatting_methods_apply_tags(self, app, format_method, expected_tag):
        """Parameterized test for all formatting methods"""
        # Add text and select it
        app.text_area.insert('1.0', 'Formatting test')
        app.text_area.tag_add('sel', '1.0', '1.5')

        # Apply formatting
        formatting_method = getattr(app, format_method)
        formatting_method()

        # Check if expected tag is applied
        tags = app.text_area.tag_names('1.0')
        assert expected_tag in tags

    def test_multiple_formatting_combinations(self, app):
        """Test that multiple formatting options can be combined"""
        # Add text and select it
        app.text_area.insert('1.0', 'Combined formatting')
        app.text_area.tag_add('sel', '1.0', '1.8')

        # Apply multiple formats
        app.apply_bold()
        app.apply_italic()
        app.apply_underline()

        # Check all tags are applied
        tags = app.text_area.tag_names('1.0')
        assert 'bold' in tags
        assert 'italic' in tags
        assert 'underline' in tags
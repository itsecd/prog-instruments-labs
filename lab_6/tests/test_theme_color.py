import pytest
from unittest.mock import Mock, patch
from tkinter import colorchooser
from twxt import MmabiaaTextpad


class TestThemeAndColor:
    """Test theme and color operations with parametrization"""

    @pytest.mark.parametrize("theme,expected_bg,expected_fg", [
        ("light", "white", "black"),
        ("dark", "black", "white"),
        ("gray", "grey", "white"),
        ("green", "green", "black"),
        ("blue", "blue", "white"),
        ("purple", "purple", "white"),
        ("orange", "orange", "black"),
        ("yellow", "yellow", "black"),
        ("pink", "pink", "black"),
        ("brown", "brown", "white"),
        ("cyan", "cyan", "black"),
        ("magenta", "magenta", "white"),
        ("custom", "aqua", "white"),
    ])
    def test_change_theme_all_variants(self, app, theme, expected_bg, expected_fg):
        """Parameterized test for all theme variants"""
        app.change_theme(theme)

        # Check background and foreground colors
        actual_bg = app.text_area.cget('bg')
        actual_fg = app.text_area.cget('fg')

        assert actual_bg == expected_bg
        assert actual_fg == expected_fg

    def test_choose_color_applies_color(self, app):
        """Test that choose_color applies selected color to text area"""
        selected_color = "#FF0000"  # Red

        with patch('tkinter.colorchooser.askcolor') as mock_chooser:
            mock_chooser.return_value = ((255, 0, 0), selected_color)
            app.choose_color()

        # Check if foreground color was updated
        actual_fg = app.text_area.cget('fg')
        assert actual_fg == selected_color

    def test_choose_color_cancelled(self, app):
        """Test that choose_color does nothing when color dialog is cancelled"""
        original_fg = app.text_area.cget('fg')

        with patch('tkinter.colorchooser.askcolor') as mock_chooser:
            mock_chooser.return_value = (None, None)  # User cancels
            app.choose_color()

        # Foreground color should remain unchanged
        actual_fg = app.text_area.cget('fg')
        assert actual_fg == original_fg

    def test_theme_change_preserves_text_content(self, app):
        """Test that changing theme doesn't affect text content"""
        test_content = "This text should remain after theme change"
        app.text_area.insert('1.0', test_content)

        # Change to dark theme
        app.change_theme("dark")

        # Check content is preserved
        content_after = app.text_area.get('1.0', 'end-1c')
        assert content_after == test_content

        # Change to light theme
        app.change_theme("light")

        # Check content is still preserved
        content_after_second = app.text_area.get('1.0', 'end-1c')
        assert content_after_second == test_content

    def test_multiple_theme_changes(self, app):
        """Test multiple consecutive theme changes work correctly"""
        themes_to_test = ["light", "dark", "blue", "green", "light"]
        expected_colors = [
            ("white", "black"),
            ("black", "white"),
            ("blue", "white"),
            ("green", "black"),
            ("white", "black")
        ]

        for i, theme in enumerate(themes_to_test):
            app.change_theme(theme)
            actual_bg = app.text_area.cget('bg')
            actual_fg = app.text_area.cget('fg')

            expected_bg, expected_fg = expected_colors[i]
            assert actual_bg == expected_bg
            assert actual_fg == expected_fg

    def test_theme_after_text_operations(self, app):
        """Test theme changes work correctly after text operations"""
        # Perform some text operations
        app.text_area.insert('1.0', 'Sample text')
        app.text_area.delete('1.0', '1.3')  # Delete "Sam"
        app.text_area.insert('1.0', 'Test')  # Insert "Test"

        # Change theme
        app.change_theme("dark")

        # Verify theme applied correctly
        assert app.text_area.cget('bg') == "black"
        assert app.text_area.cget('fg') == "white"

        # Verify text operations are preserved
        content = app.text_area.get('1.0', 'end-1c')
        assert "Testple text" in content or "Test" in content

    @pytest.mark.parametrize("color_input,expected_result", [
        (((255, 0, 0), "#ff0000"), "#ff0000"),  # Red
        (((0, 255, 0), "#00ff00"), "#00ff00"),  # Green
        (((0, 0, 255), "#0000ff"), "#0000ff"),  # Blue
        (((255, 255, 0), "#ffff00"), "#ffff00"),  # Yellow
        ((None, None), None),  # Cancelled
    ])
    def test_choose_color_parameterized(self, app, color_input, expected_result):
        """Parameterized test for choose_color with different color inputs"""
        original_fg = app.text_area.cget('fg')

        with patch('tkinter.colorchooser.askcolor') as mock_chooser:
            mock_chooser.return_value = color_input
            app.choose_color()

        actual_fg = app.text_area.cget('fg')

        if expected_result:
            assert actual_fg == expected_result
        else:
            # Color should remain unchanged when dialog cancelled
            assert actual_fg == original_fg

    def test_theme_with_formatting_tags(self, app):
        """Test that theme changes work with existing text formatting"""
        # Add text with formatting
        app.text_area.insert('1.0', 'Formatted text')
        app.text_area.tag_add('sel', '1.0', '1.10')
        app.apply_bold()

        # Change theme
        app.change_theme("purple")

        # Verify theme applied
        assert app.text_area.cget('bg') == "purple"
        assert app.text_area.cget('fg') == "white"

        # Verify formatting is preserved
        tags = app.text_area.tag_names('1.0')
        assert 'bold' in tags

    def test_unknown_theme_handling(self, app):
        """Test behavior with unknown theme name"""
        original_bg = app.text_area.cget('bg')
        original_fg = app.text_area.cget('fg')

        # Try to change to unknown theme
        app.change_theme("unknown_theme")

        # Should not change colors for unknown theme
        assert app.text_area.cget('bg') == original_bg
        assert app.text_area.cget('fg') == original_fg
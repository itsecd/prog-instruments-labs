import pytest
from unittest.mock import MagicMock


class TestSimpleUnit:
    """Simple unit tests that test pure logic without GUI dependencies"""

    def test_font_size_increase(self):
        """Test font size increase logic"""
        from twxt import MmabiaaTextpad

        # Mock root and text_area
        mock_root = MagicMock()
        mock_text = MagicMock()

        app = MmabiaaTextpad(mock_root)
        app.text_area = mock_text

        initial_size = app.current_font_size
        app.increase_font_size()

        assert app.current_font_size == initial_size + 5

    def test_font_size_decrease(self):
        """Test font size decrease logic"""
        from twxt import MmabiaaTextpad

        mock_root = MagicMock()
        mock_text = MagicMock()

        app = MmabiaaTextpad(mock_root)
        app.text_area = mock_text
        app.current_font_size = 15

        app.decrease_font_size()

        assert app.current_font_size == 10

    def test_font_size_minimum(self):
        """Test font size doesn't go below minimum"""
        from twxt import MmabiaaTextpad

        mock_root = MagicMock()
        mock_text = MagicMock()

        app = MmabiaaTextpad(mock_root)
        app.text_area = mock_text
        app.current_font_size = 6

        app.decrease_font_size()

        assert app.current_font_size == 6  # Should not decrease

    @pytest.mark.parametrize("theme,expected_bg,expected_fg", [
        ("light", "white", "black"),
        ("dark", "black", "white"),
        ("blue", "blue", "white"),
    ])
    def test_theme_logic(self, theme, expected_bg, expected_fg):
        """Test theme color logic"""
        from twxt import MmabiaaTextpad

        mock_root = MagicMock()
        mock_text = MagicMock()

        app = MmabiaaTextpad(mock_root)
        app.text_area = mock_text

        app.change_theme(theme)

        # Verify the correct colors would be applied
        mock_text.configure.assert_called_with(bg=expected_bg, fg=expected_fg)

    def test_new_file_logic(self):
        """Test new file logic"""
        from twxt import MmabiaaTextpad

        mock_root = MagicMock()
        mock_text = MagicMock()

        app = MmabiaaTextpad(mock_root)
        app.text_area = mock_text
        app.filename = "test.txt"

        app.new_file()

        assert app.filename is None
        mock_root.title.assert_called_with("Mmabia Text Pad- Untitled File")
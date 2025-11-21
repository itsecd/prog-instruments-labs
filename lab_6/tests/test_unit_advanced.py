import pytest
from unittest.mock import Mock, MagicMock, patch


class TestUnitAdvanced:
    """Advanced unit tests with parametrization"""

    @pytest.mark.parametrize("initial_size,operations,expected_size", [
        (12, ["increase", "increase", "decrease"], 17),
        (20, ["decrease", "decrease", "increase"], 15),
        (10, ["increase", "increase", "increase"], 25),
        (8, ["decrease", "decrease"], 3),
        (6, ["increase", "decrease", "increase"], 11),
    ])
    def test_font_size_operation_sequences(self, initial_size, operations, expected_size):
        """Parameterized test for font size operation sequences"""
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
            app.current_font_size = initial_size

            for operation in operations:
                if operation == "increase":
                    app.increase_font_size()
                elif operation == "decrease":
                    app.decrease_font_size()

            assert app.current_font_size == expected_size

    @pytest.mark.parametrize("theme,expected_bg,expected_fg", [
        ("light", "white", "black"),
        ("dark", "black", "white"),
        ("blue", "blue", "white"),
        ("green", "green", "black"),
    ])
    def test_change_theme_parameterized(self, theme, expected_bg, expected_fg):
        """Parameterized test for theme changes"""
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

            app.change_theme(theme)

            app.text_area.configure.assert_called_with(bg=expected_bg, fg=expected_fg)
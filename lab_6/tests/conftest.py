import pytest
import sys
import os

# Добавляем путь к корневой директории lab_6
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from twxt import MmabiaaTextpad
except ImportError:
    # Альтернативный способ импорта
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from twxt import MmabiaaTextpad

@pytest.fixture
def mock_tkinter():
    """Mock tkinter for testing"""
    from unittest.mock import MagicMock
    return {
        'root': MagicMock(),
        'text_widget': MagicMock()
    }

@pytest.fixture
def app(mock_tkinter):
    """Create app instance with mocked tkinter"""
    app = MmabiaaTextpad(mock_tkinter['root'])
    app.text_area = mock_tkinter['text_widget']
    return app
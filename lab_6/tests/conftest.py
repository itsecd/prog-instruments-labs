import pytest
import tkinter as tk
from twxt import MmabiaaTextpad

@pytest.fixture
def root_window():
    """Fixture to create a root window for testing"""
    root = tk.Tk()
    root.withdraw()  # Hide the window during tests
    yield root
    root.destroy()

@pytest.fixture
def app(root_window):
    """Fixture to create the application instance"""
    app = MmabiaaTextpad(root_window)
    yield app
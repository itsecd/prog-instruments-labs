import pytest
from unittest.mock import Mock, patch
from main import CustomQPushButton, ClickableLabel
from PySide6.QtWidgets import QApplication, QWidget

@pytest.fixture(scope="session")
def qapplication():
    """Создает QApplication для тестов."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

@pytest.fixture
def parent_widget(qapplication):
    """Фикстура для создания родительского виджета."""
    parent = QWidget()
    parent.full_formula = ClickableLabel("0")
    parent.alone_sign = ClickableLabel("0")
    return parent

@pytest.fixture
def button(parent_widget):
    """Фикстура для создания кнопки."""
    return CustomQPushButton("Test", parent_widget)

def test_button_initialization(button, parent_widget):
    """Проверка инициализации кнопки."""
    assert button.text() == "Test"
    assert parent_widget.full_formula.text() == "0"

def test_button_set_text_on_click(button):
    """Проверка установки текста при клике."""
    button.setText("Clicked")
    assert button.text() == "Clicked"

def test_button_is_enabled_by_default(parent_widget):
    """Проверка, что кнопка по умолчанию доступна для клика."""
    btn = CustomQPushButton("Test", parent_widget)
    assert btn.isEnabled()



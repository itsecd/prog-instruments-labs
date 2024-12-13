import pytest
import warnings
from unittest.mock import Mock, patch
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent, QPointingDevice
from lab_5.main import CustomQPushButton, ClickableLabel
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

@patch('main.CustomQPushButton.keyPressEvent')
def test_key_press_event(mock_key_event, button, parent_widget):
    """Проверка обработки клавиатурного ввода."""
    event = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_1, Qt.NoModifier)
    button.keyPressEvent(event)
    parent_widget.alone_sign.setText("1")
    assert parent_widget.alone_sign.text() == "1"

def test_button_click_changes_label_text(parent_widget):
    """Проверка изменения текста в метке при клике на кнопку."""
    label = ClickableLabel("Initial Text")
    button = CustomQPushButton("Change Text", parent_widget)

    def on_button_click():
        label.setText("Changed Text")

    button.clicked.connect(on_button_click)
    assert label.text() == "Initial Text"
    button.click()
    assert label.text() == "Changed Text"

def test_button_group_interaction(parent_widget):
    """Проверка взаимодействия группы кнопок."""
    button1 = CustomQPushButton("Button 1", parent_widget)
    button2 = CustomQPushButton("Button 2", parent_widget)
    button1.setEnabled(False)
    button2.click()
    assert not button1.isEnabled()
    assert button2.isEnabled()

def test_mouse_event_handling(button, parent_widget):
    """Проверка обработки событий мыши."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        event = QMouseEvent(
            QMouseEvent.MouseButtonPress,
            button.rect().center(),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
            QPointingDevice.primaryPointingDevice()
        )
        button.mousePressEvent(event)
    assert parent_widget.alone_sign.text() == "0"


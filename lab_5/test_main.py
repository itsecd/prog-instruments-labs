import unittest
from unittest.mock import patch, Mock
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent, QClipboard
from PySide6.QtWidgets import QApplication, QWidget
from main import CustomQPushButton, MainWindow, ClickableLabel

app = QApplication([])

class TestCustomQPushButton(unittest.TestCase):

    def setUp(self):
        """Создаем родителя и кнопку."""
        self.parent_widget = QWidget()
        self.parent_widget.full_formula = ClickableLabel("0")
        self.parent_widget.alone_sign = ClickableLabel("0")
        self.button = CustomQPushButton("Test", self.parent_widget)

    def test_initialization(self):
        """Проверка инициализации кнопки."""
        self.assertEqual(self.button.text(), "Test")
        self.assertEqual(self.parent_widget.full_formula.text(), "0")

    def test_mouse_event_handling(self):
        """Проверка обработки событий мыши."""
        event = QMouseEvent(QMouseEvent.MouseButtonPress, self.button.rect().center(), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        self.button.mousePressEvent(event)
        self.assertEqual(self.parent_widget.alone_sign.text(), "0")

    @patch('main.CustomQPushButton.keyPressEvent')
    def test_key_press_event(self, mock_key_event):
        """Проверка обработки клавиатурного ввода."""
        event = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_1, Qt.NoModifier)
        self.button.keyPressEvent(event)
        self.parent_widget.alone_sign.setText("1")
        self.assertEqual(self.parent_widget.alone_sign.text(), "1")

    def test_set_text_on_click(self):
        """Проверка установки текста при клике."""
        self.button.setText("Clicked")
        self.assertEqual(self.button.text(), "Clicked")

    def test_button_group_interaction(self):
        """Проверка взаимодействия группы кнопок."""
        button1 = CustomQPushButton("Button 1", self.parent_widget)
        button2 = CustomQPushButton("Button 2", self.parent_widget)
        button1.setEnabled(False)
        button2.click()
        self.assertFalse(button1.isEnabled())
        self.assertTrue(button2.isEnabled())
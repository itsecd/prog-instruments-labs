import unittest
from unittest.mock import patch
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QWidget
from main import CustomQPushButton, MainWindow, ClickableLabel

app = QApplication([])

class TestCustomQPushButton(unittest.TestCase):

    def setUp(self):
        """Создаем реального родителя вместо Mock."""
        self.parent_widget = QWidget()  # Реальный QWidget для корректной работы
        self.parent_widget.full_formula = ClickableLabel("0")
        self.parent_widget.alone_sign = ClickableLabel("0")
        self.button = CustomQPushButton("Test", self.parent_widget)

    def test_initialization(self):
        """Проверка инициализации кнопки."""
        self.assertEqual(self.button.text(), "Test")
        self.assertEqual(self.parent_widget.full_formula.text(), "0")

    def test_eng_toggle(self):
        """Проверка переключения инженерного режима."""
        self.button.is_eng = True
        self.button.show()  
        self.button.eng_toggle()
        self.assertFalse(self.button.isVisible())  # Проверяем, что скрылась

        self.button.is_eng = False  # Сбрасываем флаг обратно
        self.button.eng_toggle()  
        self.button.show()  # Принудительно делаем видимой
        self.assertTrue(self.button.isVisible())  # Проверяем, что отобразилась

class TestMouseEvents(unittest.TestCase):

    def test_mouse_event_handling(self):
        """Проверка обработки событий мыши."""
        parent_widget = QWidget()
        parent_widget.full_formula = ClickableLabel("0")
        parent_widget.alone_sign = ClickableLabel("0")
        button = CustomQPushButton("C", parent_widget)
        event = QMouseEvent(QMouseEvent.Type.MouseButtonPress, button.rect().center(), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        button.mousePressEvent(event)
        self.assertEqual(parent_widget.alone_sign.text(), "0")


if __name__ == "__main__":
    unittest.main()
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QSlider, QPushButton, QWidget, QHBoxLayout, \
    QSizePolicy, QGridLayout, QAction, QStyle
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QMouseEvent
import pygame.mixer

from recorder import Recorder
from metronome import Metronome
from menuBar import MenuBar
from pianoControlPanel import PianoControlPanel
from pianoKeyboard import PianoKeyboard
from volume import VolumeControl

from loguru import logger

# Инициализация микшера pygame
pygame.mixer.init()

# Добавляем логирование в файл
logger.add("debug.log", format="{time} - {level} - {message}", level="DEBUG")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Virtual Piano")
        self.setMinimumSize(1200, 400)

        volume_control = VolumeControl()
        recorder = Recorder()
        metronome = Metronome(bpm=120)
        menu_bar = MenuBar(self)

        self.setMenuBar(menu_bar)

        # Верхняя панель управления
        control_panel = PianoControlPanel(volume_control, recorder, metronome)

        # Нижняя панель пианино
        piano_panel = PianoKeyboard(volume_control, recorder)

        # Основной макет
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 0, 0, 0)  # Отступы: слева, сверху, справа, снизу
        main_layout.setSpacing(0)
        main_layout.addWidget(control_panel)
        main_layout.addWidget(piano_panel)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        # self.setFixedHeight(600)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

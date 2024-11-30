from PyQt5.QtWidgets import QWidget, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QMouseEvent
from loguru import logger
import pygame.mixer

# Путь к папке со звуками
SOUND_PATH = "sounds/"

class PianoKeyboard(QWidget):
    def __init__(self, volume_control, recorder):
        super().__init__()
        self.volume_control = volume_control
        self.recorder = recorder

        # Конфигурация клавиш
        self.num_white_keys = 36
        self.num_black_keys = self.num_white_keys // 7 * 5

        self.white_notes = ["C-1", "D-1", "E-1", "F-1", "G-1", "A-1", "B-1",
                            "C0", "D0", "E0", "F0", "G0", "A0", "B0",
                            "C1", "D1", "E1", "F1", "G1", "A1", "B1",
                            "C2", "D2", "E2", "F2", "G2", "A2", "B2",
                            "C3", "D3", "E3", "F3", "G3", "A3", "B3",
                            "C4"]
        self.black_notes = ["C#-1", "D#-1", "F#-1", "G#-1", "A#-1",
                            "C#0", "D#0", "F#0", "G#0", "A#0",
                            "C#1", "D#1", "F#1", "G#1", "A#1",
                            "C#2", "D#2", "F#2", "G#2", "A#2",
                            "C#3", "D#3", "F#3", "G#3", "A#3"]

        self.black_key_offsets = [1, 2, 4, 5, 6]
        self.pressed_white_keys = [False] * self.num_white_keys
        self.pressed_black_keys = [False] * self.num_black_keys

        self.sound_map = {}
        self.load_sounds()
        self.setMinimumHeight(200)

        piano_layout = QHBoxLayout()
        piano_layout.setContentsMargins(0, 0, 0, 0)
        piano_layout.setSpacing(0)
        self.setLayout(piano_layout)

    def load_sounds(self):
        """Загрузка звуков в словарь."""
        notes = self.white_notes + self.black_notes
        for note in notes:
            file_path = SOUND_PATH + note + ".wav"
            try:
                self.sound_map[note] = pygame.mixer.Sound(file_path)
            except pygame.error:
                logger.error(f"Не удалось загрузить звук {note}")
            except FileNotFoundError:
                logger.error(f"Звуковой файл {note}.wav не был найден")


    def play_sound(self, note):
        """Проигрывание звука с учетом громкости."""
        if note in self.sound_map:
            try:
                sound = self.sound_map[note]
                sound.set_volume(self.volume_control.get_volume())
                sound.play()
                self.recorder.add_note_event(note)
            except Exception as e:
                logger.error(f"Ошибка при воспроизведении звука для {note}: {e}")

    def mousePressEvent(self, event: QMouseEvent):
        """Обработка нажатий клавиш."""
        x, y = event.x(), event.y()
        width = self.width()
        height = self.height()
        white_key_width = width // self.num_white_keys
        black_key_width = int(white_key_width * 0.6)
        black_key_height = int(height * 0.6)

        note = None

        # Проверяем черные клавиши
        if y <= black_key_height:
            for i in range(self.num_black_keys):
                octave = i // 5
                black_key_x = (self.black_key_offsets[i % 5] + octave * 7) * white_key_width - black_key_width // 2
                if black_key_x <= x < black_key_x + black_key_width:
                    note = self.black_notes[i]
                    self.pressed_black_keys[i] = True
                    break

        # Проверяем белые клавиши
        if not note:
            white_key_index = x // white_key_width
            if 0 <= white_key_index < len(self.white_notes):
                note = self.white_notes[white_key_index]
                self.pressed_white_keys[white_key_index] = True

        # Проигрываем звук
        if note:
            self.play_sound(note)

        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Сбрасываем нажатие клавиш."""
        self.pressed_white_keys = [False] * self.num_white_keys
        self.pressed_black_keys = [False] * self.num_black_keys
        self.update()

    def paintEvent(self, event):
        """Отрисовка клавиш."""
        painter = QPainter(self)
        width = self.width()
        height = self.height()

        white_key_width = width // self.num_white_keys
        black_key_width = int(white_key_width * 0.6)
        black_key_height = int(height * 0.6)

        # Рисуем белые клавиши
        for i in range(self.num_white_keys):
            color = QColor(220, 220, 220) if self.pressed_white_keys[i] else QColor(255, 255, 255)
            painter.setBrush(color)
            painter.setPen(Qt.black)
            painter.drawRect(i * white_key_width, 0, white_key_width, height)

        # Рисуем черные клавиши
        for i in range(self.num_black_keys):
            octave = i // 5
            x = (self.black_key_offsets[i % 5] + octave * 7) * white_key_width - black_key_width // 2
            color = QColor(50, 50, 50) if self.pressed_black_keys[i] else QColor(0, 0, 0)
            painter.setBrush(color)
            painter.setPen(Qt.black)
            painter.drawRect(x, 0, black_key_width, black_key_height)


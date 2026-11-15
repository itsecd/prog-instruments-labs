import sys
import os
import logging
import logging.handlers
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                             QSlider, QFrame, QGroupBox)
from PyQt5.QtCore import Qt, QUrl, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtGui import QFont, QPalette, QColor
import pandas as pd
from pathlib import Path

#НАСТРОЙКА ЛОГИРОВАНИЯ
logger = logging.getLogger("player")
logger.setLevel(logging.DEBUG)

# Формат с миллисекундами (ISO8601-подобный)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Ротация файла: 1 МБ, 3 бэкапа
file_handler = logging.handlers.RotatingFileHandler(
    "player.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Консольный обработчик — только INFO и выше
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

#ИТЕРАТОР АУДИОФАЙЛОВ
class AudioIterator:
    """Итератор для аудиофайлов"""
    def __init__(self, annotation_file=None, dataset_path=None):
        self.annotation_file = annotation_file
        self.dataset_path = dataset_path
        self.data = []
        self.current_index = -1
        
        if annotation_file:
            self.load_from_csv()
        elif dataset_path:
            self.load_from_folder()
    
    def load_from_csv(self):
        """Загрузка данных из CSV файла"""
        try:
            logger.info("Loading annotations from CSV file: %s",
                        os.path.basename(self.annotation_file))
            df = pd.read_csv(self.annotation_file)
            if 'relative_path' in df.columns:
                base_dir = Path(self.annotation_file).parent
                self.data = [
                    str(base_dir / Path(row['relative_path'])) 
                    for _, row in df.iterrows()
                ]
            elif 'absolute_path' in df.columns:
                self.data = df['absolute_path'].tolist()
            logger.info("Loaded %d audio file(s) from CSV", len(self.data))
            if not self.data:
                logger.warning("CSV file contains no valid audio paths")
        except Exception as e:
            logger.error("Failed to load CSV file '%s': %s",
                         os.path.basename(self.annotation_file), e)
            self.data = []
    
    def load_from_folder(self):
        """Загрузка данных из папки"""
        if self.dataset_path:
            logger.info("Scanning folder for audio files: %s",
                        os.path.basename(self.dataset_path))
            audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a']
            self.data = [
                str(file) for file in Path(self.dataset_path).rglob('*')
                if file.suffix.lower() in audio_extensions
            ]
            logger.info("Found %d audio file(s) in folder", len(self.data))
            if not self.data:
                logger.warning("No audio files found in selected folder")
    
    def next(self):
        """Получить следующий файл"""
        if not self.data:
            logger.warning("Attempted to get next file, but dataset is empty")
            return None
        
        self.current_index = (self.current_index + 1) % len(self.data)
        logger.debug("Iterator moved to next index: %d/%d",
                     self.current_index + 1, len(self.data))
        return self.data[self.current_index]
    
    def previous(self):
        """Получить предыдущий файл"""
        if not self.data:
            logger.warning("Attempted to get previous file, but dataset is empty")
            return None
        
        self.current_index = (self.current_index - 1) % len(self.data)
        logger.debug("Iterator moved to previous index: %d/%d",
                     self.current_index + 1, len(self.data))
        return self.data[self.current_index]
    
    def current(self):
        """Получить текущий файл"""
        if not self.data or self.current_index < 0:
            return None
        return self.data[self.current_index]
    
    def __len__(self):
        return len(self.data)

#ВИДЖЕТЫ (без изменений)
class ModernButton(QPushButton):
    def __init__(self, text, icon=None, color="#7E57C2", hover_color="#9575CD"):
        super().__init__(text)
        self.normal_color = color
        self.hover_color = hover_color
        self.setMinimumHeight(45)
        self.setFont(QFont("Segoe UI", 10, QFont.Medium))
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.normal_color};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: medium;
            }}
            QPushButton:hover {{
                background-color: {self.hover_color};
            }}
            QPushButton:pressed {{
                background-color: #673AB7;
            }}
            QPushButton:disabled {{
                background-color: #B39DDB;
                color: #EDE7F6;
            }}
        """)
        if icon:
            self.setIcon(icon)

class ModernSlider(QSlider):
    def __init__(self, orientation=Qt.Horizontal):
        super().__init__(orientation)
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px; background: #D1C4E9; border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #7E57C2; border-radius: 3px;
            }
            QSlider::add-page:horizontal {
                background: #D1C4E9; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: white; width: 18px; height: 18px;
                margin: -6px 0; border-radius: 9px;
                border: 2px solid #7E57C2;
            }
            QSlider::handle:horizontal:hover {
                background: #F3E5F5; border: 2px solid #9575CD;
            }
        """)

class TrackInfoWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            TrackInfoWidget {
                background-color: #F3E5F5;
                border-radius: 12px;
                border: 2px solid #D1C4E9;
                padding: 20px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        self.track_name = QLabel("Название трека не загружено")
        self.track_name.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.track_name.setStyleSheet("color: #4527A0;")
        self.track_name.setAlignment(Qt.AlignCenter)
        self.track_name.setWordWrap(True)
        self.file_info = QLabel("Файл: -")
        self.file_info.setFont(QFont("Segoe UI", 10))
        self.file_info.setStyleSheet("color: #5E35B1;")
        self.file_info.setAlignment(Qt.AlignCenter)
        self.duration_info = QLabel("Длительность: -")
        self.duration_info.setFont(QFont("Segoe UI", 10))
        self.duration_info.setStyleSheet("color: #5E35B1;")
        self.duration_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.track_name)
        layout.addWidget(self.file_info)
        layout.addWidget(self.duration_info)
        self.setLayout(layout)
    
    def update_info(self, track_name, file_name, duration):
        self.track_name.setText(f"{track_name}")
        self.file_info.setText(f"Файл: {file_name}")
        self.duration_info.setText(f"Длительность: {duration}")

#ГЛАВНОЕ ОКНО
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.audio_iterator = None
        self.media_player = QMediaPlayer()
        self.is_playing = False
        logger.info("Application started")
        self.setup_palette()
        self.init_ui()
        self.setup_media_player()
    
    def setup_palette(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(250, 245, 255))
        palette.setColor(QPalette.WindowText, QColor(33, 33, 33))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(248, 245, 252))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(33, 33, 33))
        palette.setColor(QPalette.Text, QColor(33, 33, 33))
        palette.setColor(QPalette.Button, QColor(237, 231, 246))
        palette.setColor(QPalette.ButtonText, QColor(69, 39, 160))
        palette.setColor(QPalette.BrightText, Qt.white)
        palette.setColor(QPalette.Link, QColor(94, 53, 177))
        palette.setColor(QPalette.Highlight, QColor(126, 87, 194))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        self.setPalette(palette)
    
    def init_ui(self):
        self.setWindowTitle("Аудиоплеер")
        self.setGeometry(100, 100, 900, 700)
        central_widget = QWidget()
        central_widget.setStyleSheet("QWidget { background-color: #FAF5FF; }")
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        central_widget.setLayout(main_layout)
        
        header_label = QLabel("Слушать музыку скачать онлайн без вирусов")
        header_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
        header_label.setStyleSheet("""
            QLabel {
                color: white; padding: 15px; border-radius: 12px;
                background-color: #7E57C2;
            }
        """)
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)
        
        # Панель загрузки
        load_group = QGroupBox("Загрузка данных")
        load_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        load_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #D1C4E9; border-radius: 12px;
                margin-top: 10px; padding-top: 20px; background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 15px;
                padding: 0 15px 0 15px; color: #5E35B1;
            }
        """)
        load_layout = QHBoxLayout()
        load_layout.setSpacing(15)
        self.btn_load_csv = ModernButton("Загрузить CSV файл", color="#7E57C2", hover_color="#9575CD")
        self.btn_load_csv.clicked.connect(self.load_csv_file)
        self.btn_load_folder = ModernButton("Загрузить папку", color="#7E57C2", hover_color="#9575CD")
        self.btn_load_folder.clicked.connect(self.load_folder)
        load_layout.addWidget(self.btn_load_csv)
        load_layout.addWidget(self.btn_load_folder)
        load_layout.addStretch()
        load_group.setLayout(load_layout)
        main_layout.addWidget(load_group)
        
        self.track_info_widget = TrackInfoWidget()
        main_layout.addWidget(self.track_info_widget)
        
        # Панель управления
        control_group = QGroupBox("Управление воспроизведением")
        control_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        control_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #D1C4E9; border-radius: 12px;
                margin-top: 10px; padding-top: 20px; background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 15px;
                padding: 0 15px 0 15px; color: #5E35B1;
            }
        """)
        control_layout = QVBoxLayout()
        control_layout.setSpacing(15)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)
        
        self.btn_prev = ModernButton("⏮", color="#9575CD", hover_color="#7E57C2")
        self.btn_prev.clicked.connect(self.prev_track)
        self.btn_prev.setMinimumSize(60, 60)
        self.btn_prev.setFont(QFont("Segoe UI", 14))
        self.btn_prev.setEnabled(False)
        
        self.btn_play = ModernButton("▶", color="#5E35B1", hover_color="#4527A0")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setMinimumSize(80, 80)
        self.btn_play.setFont(QFont("Segoe UI", 16))
        self.btn_play.setEnabled(False)
        
        self.btn_next = ModernButton("⏭", color="#9575CD", hover_color="#7E57C2")
        self.btn_next.clicked.connect(self.next_track)
        self.btn_next.setMinimumSize(60, 60)
        self.btn_next.setFont(QFont("Segoe UI", 14))
        self.btn_next.setEnabled(False)
        
        button_layout.addStretch()
        button_layout.addWidget(self.btn_prev)
        button_layout.addWidget(self.btn_play)
        button_layout.addWidget(self.btn_next)
        button_layout.addStretch()
        
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(8)
        self.slider_progress = ModernSlider(Qt.Horizontal)
        self.slider_progress.sliderMoved.connect(self.set_position)
        self.slider_progress.setEnabled(False)
        
        time_layout = QHBoxLayout()
        self.lbl_current_time = QLabel("00:00")
        self.lbl_current_time.setFont(QFont("Segoe UI", 10, QFont.Medium))
        self.lbl_current_time.setStyleSheet("color: #5E35B1;")
        self.lbl_total_time = QLabel("00:00")
        self.lbl_total_time.setFont(QFont("Segoe UI", 10, QFont.Medium))
        self.lbl_total_time.setStyleSheet("color: #5E35B1;")
        self.lbl_total_time.setAlignment(Qt.AlignRight)
        time_layout.addWidget(self.lbl_current_time)
        time_layout.addStretch()
        time_layout.addWidget(self.lbl_total_time)
        
        volume_layout = QHBoxLayout()
        volume_label = QLabel("Громкость:")
        volume_label.setFont(QFont("Segoe UI", 10, QFont.Medium))
        volume_label.setStyleSheet("color: #5E35B1;")
        self.slider_volume = ModernSlider(Qt.Horizontal)
        self.slider_volume.setRange(0, 100)
        self.slider_volume.setValue(70)
        self.slider_volume.setMaximumWidth(200)
        self.slider_volume.valueChanged.connect(self.set_volume)
        self.lbl_volume = QLabel("70%")
        self.lbl_volume.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.lbl_volume.setStyleSheet("color: #4527A0;")
        self.lbl_volume.setMinimumWidth(40)
        volume_layout.addWidget(volume_label)
        volume_layout.addWidget(self.slider_volume)
        volume_layout.addWidget(self.lbl_volume)
        volume_layout.addStretch()
        
        progress_layout.addLayout(button_layout)
        progress_layout.addWidget(self.slider_progress)
        progress_layout.addLayout(time_layout)
        progress_layout.addLayout(volume_layout)
        control_layout.addLayout(progress_layout)
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #EDE7F6; border-radius: 12px;
                border: 2px solid #D1C4E9; padding: 20px;
            }
        """)
        info_layout = QHBoxLayout()
        self.lbl_dataset_info = QLabel("Файлов в датасете: 0 | Текущий: 0/0")
        self.lbl_dataset_info.setFont(QFont("Segoe UI", 11, QFont.Medium))
        self.lbl_dataset_info.setStyleSheet("color: #4527A0;")
        self.lbl_dataset_info.setAlignment(Qt.AlignCenter)
        self.status_indicator = QLabel("● Остановлено")
        self.status_indicator.setFont(QFont("Segoe UI", 11, QFont.Medium))
        self.status_indicator.setStyleSheet("color: #F44336;")
        info_layout.addWidget(self.lbl_dataset_info)
        info_layout.addStretch()
        info_layout.addWidget(self.status_indicator)
        info_frame.setLayout(info_layout)
        main_layout.addWidget(info_frame)
        
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #EDE7F6; color: #5E35B1;
                font-weight: medium; border-top: 1px solid #D1C4E9;
            }
        """)
        self.status_bar.setFont(QFont("Segoe UI", 9))
        self.status_bar.showMessage("Готов к работе")
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_slider)
        self.timer.start(1000)
    
    def setup_media_player(self):
        self.media_player.stateChanged.connect(self.media_state_changed)
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.set_volume(70)
    
    def set_volume(self, value):
        self.media_player.setVolume(value)
        self.lbl_volume.setText(f"{value}%")
        logger.debug("Volume set to %d%%", value)
    
    def load_csv_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выберите CSV файл", "", "CSV Files (*.csv)"
        )
        if file_name:
            logger.info("User selected CSV file: %s", os.path.basename(file_name))
            self.audio_iterator = AudioIterator(annotation_file=file_name)
            if len(self.audio_iterator) > 0:
                self.btn_play.setEnabled(True)
                self.btn_next.setEnabled(True)
                self.btn_prev.setEnabled(True)
                self.slider_progress.setEnabled(True)
                self.update_dataset_info()
                self.next_track()
                self.status_bar.showMessage(f"Загружен CSV файл: {os.path.basename(file_name)}")
            else:
                logger.warning("Selected CSV file contains no audio tracks")
                self.status_bar.showMessage("CSV файл не содержит аудиофайлов")
    
    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Выберите папку с аудиофайлами"
        )
        if folder_path:
            logger.info("User selected folder: %s", os.path.basename(folder_path))
            self.audio_iterator = AudioIterator(dataset_path=folder_path)
            if len(self.audio_iterator) > 0:
                self.btn_play.setEnabled(True)
                self.btn_next.setEnabled(True)
                self.btn_prev.setEnabled(True)
                self.slider_progress.setEnabled(True)
                self.update_dataset_info()
                self.next_track()
                self.status_bar.showMessage(f"Загружена папка: {os.path.basename(folder_path)}")
            else:
                logger.warning("Selected folder contains no audio files")
                self.status_bar.showMessage("В папке не найдено аудиофайлов")
    
    def next_track(self):
        if self.audio_iterator and len(self.audio_iterator) > 0:
            self.stop_playback()
            next_file = self.audio_iterator.next()
            if next_file:
                logger.info("Switching to next track: %s", os.path.basename(next_file))
                if os.path.exists(next_file):
                    self.play_file(next_file)
                else:
                    logger.error("Next track file does not exist: %s",
                                 os.path.basename(next_file))
    
    def prev_track(self):
        if self.audio_iterator and len(self.audio_iterator) > 0:
            self.stop_playback()
            prev_file = self.audio_iterator.previous()
            if prev_file:
                logger.info("Switching to previous track: %s", os.path.basename(prev_file))
                if os.path.exists(prev_file):
                    self.play_file(prev_file)
                else:
                    logger.error("Previous track file does not exist: %s",
                                 os.path.basename(prev_file))
    
    def play_file(self, file_path):
        try:
            logger.info("Loading track for playback: %s", os.path.basename(file_path))
            url = QUrl.fromLocalFile(file_path)
            content = QMediaContent(url)
            self.media_player.setMedia(content)
            
            file_name = os.path.basename(file_path)
            track_name = os.path.splitext(file_name)[0]
            if '_' in track_name:
                parts = track_name.split('_')
                if len(parts) > 2:
                    track_name = ' '.join(parts[2:]).title()
                else:
                    track_name = ' '.join(parts).title()
            else:
                track_name = track_name.title()
            
            self.track_info_widget.update_info(track_name, file_name, "загрузка...")
            self.update_dataset_info()
            self.status_indicator.setText("● Готов")
            self.status_indicator.setStyleSheet("color: #4CAF50;")
            logger.debug("Track metadata updated: name='%s', file='%s'",
                         track_name, file_name)
        except Exception as e:
            logger.error("Failed to load track '%s': %s",
                         os.path.basename(file_path), e)
            self.status_bar.showMessage(f"Ошибка загрузки файла: {e}")
    
    def toggle_play(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.btn_play.setText("▶")
            self.btn_play.setStyleSheet("""
                QPushButton {
                    background-color: #5E35B1; color: white; border: none;
                    border-radius: 40px; padding: 10px 20px;
                    font-weight: bold; font-size: 16px;
                }
                QPushButton:hover { background-color: #4527A0; }
                QPushButton:pressed { background-color: #311B92; }
            """)
            self.is_playing = False
            self.status_indicator.setText("● Пауза")
            self.status_indicator.setStyleSheet("color: #FF9800;")
            logger.info("Playback paused")
        else:
            self.media_player.play()
            self.btn_play.setText("⏸")
            self.btn_play.setStyleSheet("""
                QPushButton {
                    background-color: #4527A0; color: white; border: none;
                    border-radius: 40px; padding: 10px 20px;
                    font-weight: bold; font-size: 16px;
                }
                QPushButton:hover { background-color: #5E35B1; }
                QPushButton:pressed { background-color: #311B92; }
            """)
            self.is_playing = True
            self.status_indicator.setText("● Играет")
            self.status_indicator.setStyleSheet("color: #2196F3;")
            logger.info("Playback started")
    
    def stop_playback(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.stop()
            self.btn_play.setText("▶")
            self.btn_play.setStyleSheet("""
                QPushButton {
                    background-color: #5E35B1; color: white; border: none;
                    border-radius: 40px; padding: 10px 20px;
                    font-weight: bold; font-size: 16px;
                }
                QPushButton:hover { background-color: #4527A0; }
                QPushButton:pressed { background-color: #311B92; }
            """)
            self.is_playing = False
            self.status_indicator.setText("● Стоп")
            self.status_indicator.setStyleSheet("color: #F44336;")
            logger.info("Playback stopped")
    
    def set_position(self, position):
        self.media_player.setPosition(position)
        logger.debug("Playback position set to %d ms", position)
    
    def update_slider(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.slider_progress.setValue(self.media_player.position())
    
    def media_state_changed(self, state):
        pass
    
    def position_changed(self, position):
        self.lbl_current_time.setText(self.format_time(position))
    
    def duration_changed(self, duration):
        self.slider_progress.setRange(0, duration)
        self.lbl_total_time.setText(self.format_time(duration))
        if self.audio_iterator and self.audio_iterator.current():
            file_name = os.path.basename(self.audio_iterator.current())
            track_name = os.path.splitext(file_name)[0]
            if '_' in track_name:
                parts = track_name.split('_')
                if len(parts) > 2:
                    track_name = ' '.join(parts[2:]).title()
                else:
                    track_name = ' '.join(parts).title()
            else:
                track_name = track_name.title()
            duration_str = self.format_time(duration)
            self.track_info_widget.update_info(track_name, file_name, duration_str)
            logger.debug("Track duration received: %s for file '%s'",
                         duration_str, file_name)
    
    def format_time(self, milliseconds):
        seconds = milliseconds // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        hours = minutes // 60
        minutes = minutes % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def update_dataset_info(self):
        if self.audio_iterator:
            total = len(self.audio_iterator)
            current = self.audio_iterator.current_index + 1
            self.lbl_dataset_info.setText(f"Файлов: {total} | Текущий: {current}/{total}")
    
    def closeEvent(self, event):
        logger.info("Application is closing")
        self.stop_playback()
        self.timer.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet("""
        QMainWindow { background-color: #FAF5FF; }
        QMenuBar {
            background-color: #EDE7F6; color: #5E35B1; font-weight: medium;
        }
        QMenuBar::item:selected { background-color: #D1C4E9; }
        QMenu { background-color: white; border: 1px solid #D1C4E9; }
        QMenu::item:selected { background-color: #EDE7F6; color: #5E35B1; }
        QToolTip {
            background-color: #F3E5F5; color: #4527A0;
            border: 1px solid #D1C4E9; padding: 5px;
            border-radius: 3px; font-size: 10px;
        }
        QScrollBar:vertical {
            border: none; background: #EDE7F6; width: 10px; margin: 0px;
        }
        QScrollBar::handle:vertical {
            background: #B39DDB; border-radius: 5px; min-height: 20px;
        }
        QScrollBar::handle:vertical:hover { background: #9575CD; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none; background: none;
        }
    """)
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
import sys
from main import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QTimer, QTime
from PyQt5 import QtWidgets
from PyQt5 import QtCore
import os


class Window(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.folderpath = None
        self.menu = None
        self.count = 0
        self.current_index = 0
        self.player = QMediaPlayer()
        self.time_vid = self.ui.label
        self.timer_l = QTimer()
        self.timer_l.timeout.connect(self.update_timer)
        self.timer_l.start(1000)
        self.ui.ope_folder.clicked.connect(self.select_folder)
        self.ui.left.clicked.connect(self.left_skip)
        self.ui.right.clicked.connect(self.right_skip)
        self.ui.pause.clicked.connect(self.pause_b)
        self.ui.loop.clicked.connect(self.repeat)
        self.volume = self.ui.volume
        self.volume.setMinimum(0)
        self.volume.setMaximum(100)
        self.volume.setValue(50)
        self.player.setVolume(50)
        self.volume.setTickInterval(10)
        self.volume.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.volume.valueChanged.connect(self.volume_ch)
        self.player.mediaStatusChanged.connect(self.media_status_changed)
        self.time_slider = self.ui.time_line
        self.time_slider.sliderMoved.connect(self.set_position)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_slider)
        self.timer.start(1000)

    def media_status_changed(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.right_skip()

    def select_folder(self):
        self.folderpath = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Выберите папку')
        QtWidgets.QMessageBox.information(
            self, 'Папка выбрана', self.folderpath)
        self.files = os.listdir(self.folderpath)
        self.mp3_files = [file for file in self.files if file.endswith('.mp3')]

    def play_current(self):
        url = QUrl.fromLocalFile(
            self.folderpath + '/' + self.mp3_files[self.current_index])
        content = QMediaContent(url)
        self.player.setMedia(content)
        self.player.play()

    def left_skip(self):
        self.current_index = (self.current_index - 1) % len(self.mp3_files)
        self.play_current()

    def right_skip(self):
        self.current_index = (self.current_index + 1) % len(self.mp3_files)
        self.play_current()

    def pause_b(self):
        if self.count == 0:
            self.count += 1
            self.play_current()
        else:
            if self.player.state() == QMediaPlayer.PlayingState:
                self.player.pause()
            else:
                self.player.play()

    def volume_ch(self):
        volume_ = self.volume.value()
        self.player.setVolume(volume_)

    def repeat(self):
        self.play_current()

    def update_timer(self):
        current_time = QTime(0, 0)
        current_time = current_time.addMSecs(self.player.position())
        self.time_vid.setText(current_time.toString("mm:ss"))

    def update_slider(self):
        if self.player.duration() > 0:
            pos = self.player.position()
            dur = self.player.duration()
            self.time_slider.setMaximum(dur)
            self.time_slider.setValue(pos)

    def set_position(self, position):
        self.player.setPosition(position)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
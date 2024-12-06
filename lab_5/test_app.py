import pytest
from main_window import Window
from PyQt5.QtWidgets import QApplication
from PyQt5.QtMultimedia import QMediaPlayer
from PyQt5.QtCore import QTimer
from unittest.mock import patch


@pytest.fixture(scope="module")
def app():
    """создаёт глобальный QApplication для тестов"""
    app = QApplication.instance() or QApplication([])
    return app


def test_initial_state(app):
    """проверяет, что окно создается с ожидаемыми начальными значениями"""
    window = Window()

    assert window.current_index == 0
    assert window.count == 0
    assert window.player.volume() == 50
    assert window.folderpath is None


def test_volume_change(app):
    """проверяет изменение громкости через слайдер"""
    window = Window()
    window.volume.setValue(75)
    assert window.player.volume() == 75


def test_select_folder(mocker, app):
    """тестирует выбор папки и фильтрацию файлов"""
    mocker.patch("PyQt5.QtWidgets.QFileDialog.getExistingDirectory",
                 return_value="/test/folder")
    mocker.patch("os.listdir", return_value=[
                 "file1.mp3", "file2.mp3", "image.jpg"])
    mock_message_box = mocker.patch(
        "PyQt5.QtWidgets.QMessageBox.information", return_value=None)
    window = Window()
    window.select_folder()
    assert window.folderpath == "/test/folder"
    assert window.mp3_files == ["file1.mp3", "file2.mp3"]
    mock_message_box.assert_called_once_with(
        window, 'Папка выбрана', "/test/folder")


def test_play_current(mocker, app):
    """проверяет воспроизведение текущего трека"""
    window = Window()
    window.folderpath = "/test/folder"
    window.mp3_files = ["song1.mp3", "song2.mp3"]
    mock_set_media = patch.object(window.player, "setMedia").start()
    mock_play = patch.object(window.player, "play").start()
    window.play_current()
    expected_url = "/test/folder/song1.mp3"
    mock_set_media.assert_called_once()
    mock_play.assert_called_once()


@pytest.mark.parametrize("initial_index, expected_index", [
    (0, 1),
    (1, 2),
    (2, 0),
])
def test_skip_buttons(app, initial_index, expected_index):
    """проверяет пропуск треков"""
    window = Window()
    window.folderpath = "/test/folder"
    window.mp3_files = ["song1.mp3", "song2.mp3", "song3.mp3"]
    window.current_index = initial_index
    window.right_skip()
    assert window.current_index == expected_index


def test_repeat(mocker, app):
    """тестирует кнопку повтора воспроизведения"""
    window = Window()
    mock_play_current = mocker.patch.object(window, "play_current")
    window.repeat()
    mock_play_current.assert_called_once()


def test_update_timer(mocker, app):
    """тестирует обновление таймера на метке времени"""
    window = Window()
    mocker.patch.object(window.player, "position", return_value=65000)
    mock_set_text = mocker.patch.object(window.ui.label, "setText")
    window.update_timer()
    mock_set_text.assert_called_once_with("01:05")


def test_pause_button(app):
    """тестирует кнопку паузы для различных состояний плеера"""
    window = Window()
    window.folderpath = "/test/folder"
    window.mp3_files = ["song1.mp3", "song2.mp3"]

    def on_state_changed(state):
        assert state == QMediaPlayer.PlayingState
        window.player.play()

    window.player.stateChanged.connect(on_state_changed)
    window.player.play()
    QTimer.singleShot(100, lambda: window.pause_b())

    def check_paused_state():
        assert window.player.state() == QMediaPlayer.PausedState

    QTimer.singleShot(200, check_paused_state)
    QTimer.singleShot(300, lambda: window.pause_b())

    def check_playing_state():
        assert window.player.state() == QMediaPlayer.PlayingState

    QTimer.singleShot(400, check_playing_state)


def test_media_status_changed(app):
    """тестирует поведение при окончании трека"""
    window = Window()
    window.mp3_files = ["song1.mp3", "song2.mp3"]
    window.folderpath = "/test/folder"
    window.current_index = 0
    window.media_status_changed(QMediaPlayer.EndOfMedia)
    assert window.current_index == 1
    window.media_status_changed(QMediaPlayer.LoadedMedia)
    assert window.current_index == 1

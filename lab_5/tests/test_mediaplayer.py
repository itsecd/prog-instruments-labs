import os
import sys
from unittest.mock import Mock, patch, PropertyMock

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from player import mediaplayer


@pytest.fixture
def media_player_instance():
    path_mock = Mock()
    song_time_mock = Mock()
    song_duration_mock = Mock()
    volume_mock = Mock()
    player = mediaplayer(path_mock, song_time_mock, song_duration_mock, volume_mock)
    return player


def test_play(media_player_instance):
    with patch.object(media_player_instance.player, "play") as mock_play:
        media_player_instance.play()
        mock_play.assert_called_once()  


def test_pause(media_player_instance):
    with patch.object(media_player_instance.player, "pause") as mock_pause:
        media_player_instance.pause()
        mock_pause.assert_called_once()  


def test_volume_set(media_player_instance):
    mock_volume = Mock()
    mock_volume.get.return_value = 0.5  
    media_player_instance.volume = mock_volume

    with patch.object(media_player_instance.player, "volume", new_callable=PropertyMock) as mock_player_volume:
        media_player_instance.volume_()
        mock_player_volume.assert_called_once_with(0.5)  


@pytest.mark.parametrize("time,expected_seek_called", [
    (10, True),  
    (9999, False),  
    (-10, False),  
])
def test_jump(media_player_instance, time, expected_seek_called):
    media_player_instance.player.source = Mock(duration=300)  
    with patch.object(media_player_instance.player, "seek") as mock_seek:
        media_player_instance.jump(time)
        assert mock_seek.called == expected_seek_called  


def test_play_song_error(media_player_instance):
    with patch("pyglet.media.load", side_effect=Exception("Test Error")):
        media_player_instance.path.get.return_value = "invalid/path"
        with patch("logging.error") as mock_log_error:
            media_player_instance.play_song()
            mock_log_error.assert_called_once()  


def test_now_(media_player_instance):
    with patch.object(media_player_instance.player, "time", new_callable=PropertyMock) as mock_time:
        mock_time.return_value = 123  
        result = media_player_instance.now_()
        assert result == "0:02:03" 


@pytest.mark.parametrize("time,current_time,expected_time", [
    (10, 20, 10),  
    (30, 10, 0),  
])
def test_rewind(media_player_instance, time, current_time, expected_time):
    with patch.object(media_player_instance.player, "time", new_callable=PropertyMock) as mock_time:
        mock_time.return_value = current_time
        with patch.object(media_player_instance.player, "seek") as mock_seek:
            media_player_instance.rewind()
            mock_seek.assert_called_once_with(expected_time)  
 
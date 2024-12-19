import pytest
from unittest.mock import patch
from io import StringIO

from Adventure import *

# Глобальная переменная для оружия
weapon = False


@pytest.fixture
def reset_weapon():
    global weapon
    weapon = False  # Сбрасываем оружие перед каждым тестом


@pytest.mark.parametrize("direction, expected_output", [
    ("left", "You see a dark shadowy figure appear"),
    ("right", "You see a wall of skeletons"),
    ("forward", "You hear strange voices")
])
def test_introScene(reset_weapon, direction, expected_output):
    with patch('builtins.input', return_value=direction), patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        introScene()
        output = mock_stdout.getvalue().strip()
        assert expected_output in output


@pytest.mark.parametrize("user_input, expected_output", [
    ("fight", "The goul-like creature has killed you."),
    ("flee", "You see a wall of skeletons as you walk into the room.")
])
def test_strangeCreature(reset_weapon, user_input, expected_output):
    with patch('builtins.input', side_effect=[user_input]), patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        if user_input == "fight":
            with pytest.raises(SystemExit):  # Проверка завершения программы
                strangeCreature()
        else:
            strangeCreature()
        output = mock_stdout.getvalue().strip()
        assert expected_output in output


def test_showSkeletons_left(reset_weapon):
    with patch('builtins.input', side_effect=["left"]), patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        showSkeletons()
        output = mock_stdout.getvalue().strip()
        assert "You find that this door opens into a wall." in output
        assert weapon is True  # Проверка, что оружие получено


def test_showSkeletons_backward(reset_weapon):
    with patch('builtins.input', side_effect=["backward"]), patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        showSkeletons()
        output = mock_stdout.getvalue().strip()
        assert "You are at a crossroads" in output


@pytest.mark.parametrize("user_input, expected_output", [
    ("right", "Multiple goul-like creatures start emerging"),
    ("left", "You made it! You've found an exit.")
])
def test_hauntedRoom(reset_weapon, user_input, expected_output):
    with patch('builtins.input', side_effect=[user_input]), patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        if user_input == "right":
            with pytest.raises(SystemExit):  # Проверка завершения программы
                hauntedRoom()
        else:
            hauntedRoom()
        output = mock_stdout.getvalue().strip()
        assert expected_output in output


def test_cameraScene_forward(reset_weapon):
    with patch('builtins.input', return_value="forward"), patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        cameraScene()
        output = mock_stdout.getvalue().strip()
        assert "You made it! You've found an exit." in output


def test_showShadowFigure_right(reset_weapon):
    with patch('builtins.input', return_value="right"), patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        showShadowFigure()
        output = mock_stdout.getvalue().strip()
        assert "You see a camera that has been dropped on the ground." in output

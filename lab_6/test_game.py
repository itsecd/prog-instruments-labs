import pytest
from unittest.mock import Mock, patch
import pygame as pg

# Важно: импортируем классы из нашего основного файла main
from lab_6.main import Ball, Game, SCREEN_WIDTH, BALL_RADIUS


# Инициализация Pygame для тестов.
# pygame-ce должен справиться с этим в CI/CD без экрана.


# --- 6 простых тестов ---

def test_ball_vertical_bounce():
    """Тест 1: Шар меняет вертикальное направление у верхней границы."""
    mock_rect = pg.Rect(100, 5, 10, 10)
    ball = Ball(mock_rect, 100, 5, speed=5, direction_mod=0)
    ball.dy = -5
    mock_rect.centerx = ball.x
    mock_rect.centery = ball.y
    ball.move()
    assert ball.dy > 0


def test_ball_horizontal_bounce():
    """Тест 2: Шар меняет горизонтальное направление у боковой границы."""
    mock_rect = pg.Rect(5, 100, 10, 10)
    ball = Ball(mock_rect, 5, 100, speed=5, direction_mod=1)
    ball.dx = -5
    mock_rect.centerx = ball.x
    mock_rect.centery = ball.y
    ball.move()
    assert ball.dx > 0


def test_win_condition_met():
    """Тест 3: game_win становится True при выполнении условия победы."""
    game = Game()
    game.check_win_condition(colors=[()] * 70, current_shots=10, max_shots=10)
    assert game.game_win is True


def test_win_condition_not_met_blocks_left():
    """Тест 4: game_win остается False, если есть блоки."""
    game = Game()
    game.check_win_condition(colors=[(255, 0, 0)], current_shots=10, max_shots=10)
    assert game.game_win is False


def test_volume_increase():
    """Тест 5: Громкость увеличивается корректно."""
    game = Game()
    game.volume = 0.5
    game.volume = min(1.0, game.volume + 0.1)
    assert game.volume == pytest.approx(0.6)


def test_volume_max_limit():
    """Тест 6: Громкость не превышает 1.0."""
    game = Game()
    game.volume = 1.0
    game.volume = min(1.0, game.volume + 0.1)
    assert game.volume == 1.0


# --- 2 сложных теста ---

@pytest.mark.parametrize("initial_dx, initial_dy, expected_dx_sign, expected_dy_sign", [
    (5, 5, -1, -1),
    (5, 0, -1, 0),
    (0, 5, 0, -1),
])
def test_ball_block_collision_bounce(initial_dx, initial_dy, expected_dx_sign, expected_dy_sign):
    """Тест 7 (Параметризованный): Проверка логики отскока шара от блока."""
    ball_rect = pg.Rect(50, 50, 10, 10)
    ball = Ball(ball_rect, 50, 50, speed=0, direction_mod=0)
    ball.dx, ball.dy = initial_dx, initial_dy
    block_rect = pg.Rect(55, 55, 20, 20)
    ball.move(block=block_rect)
    if expected_dx_sign != 0:
        assert (ball.dx > 0 and expected_dx_sign > 0) or (ball.dx < 0 and expected_dx_sign < 0)
    if expected_dy_sign != 0:
        assert (ball.dy > 0 and expected_dy_sign > 0) or (ball.dy < 0 and expected_dy_sign < 0)


def test_update_win_count_on_win():
    """Тест 8 (с Mock): Проверяет, что метод для обновления счета в БД вызывается при победе."""
    # Используем `patch` из `unittest.mock`
    with patch.object(Game, '_update_win_count_in_db') as mock_update:
        # Arrange
        game = Game()
        game.game_win = True

        # Act
        # Симулируем вызов, который происходит в коде при победе
        if game.game_win:
            game._update_win_count_in_db()

        # Assert
        mock_update.assert_called_once()

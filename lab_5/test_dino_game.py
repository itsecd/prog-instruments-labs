import pytest
import pygame
from unittest.mock import Mock, patch
from main import Dinosaur, SmallCactus, LargeCactus, Bird, Cloud, SCREEN_HEIGHT, SCREEN_WIDTH

@pytest.fixture(scope="module", autouse=True)
def no_display():
    with patch("pygame.display.set_mode"):
        yield

@pytest.fixture
def dino():
    """Фикстура для создания объекта Dinosaur."""
    return Dinosaur()

def test_dinosaur_initial_state(dino):
    """Тест начального состояния динозавра."""
    assert dino.dino_run is True
    assert dino.dino_jump is False
    assert dino.dino_duck is False
    assert dino.dino_rect.y == dino.Y_POS
    assert dino.step_index == 0

def test_dinosaur_starts_running():
    """Убедимся, что динозавр начинает игру в состоянии бега."""
    dino = Dinosaur()
    assert dino.dino_run is True
    assert dino.dino_jump is False
    assert dino.dino_duck is False

def test_dinosaur_jump(dino):
    """Тест прыжка динозавра."""
    dino.dino_jump = True
    dino.jump()
    assert dino.dino_rect.y < dino.Y_POS  
    assert dino.jump_vel < dino.JUMP_VEL  

def test_dinosaur_duck(dino):
    """Тест приседания динозавра."""
    dino.dino_duck = True
    dino.duck()
    assert dino.dino_rect.y == dino.Y_POS_DUCK  
    assert dino.image == dino.duck_img[0]  

def test_cloud_update():
    """Тест обновления облака."""
    cloud = Cloud()
    initial_x = cloud.x
    cloud.update()
    assert cloud.x < initial_x or (cloud.x == SCREEN_WIDTH + 800)  

def test_bird_animation():
    """Тест анимации птицы."""
    bird = Bird([Mock(), Mock()])
    bird.draw(Mock())
    assert bird.index == 1  

def test_dinosaur_reset_after_jump(dino):
    """Тест сброса состояния динозавра после завершения прыжка."""
    dino.dino_jump = True
    dino.jump_vel = -dino.JUMP_VEL - 1  
    dino.jump()
    assert dino.dino_jump is False 
    assert dino.jump_vel == dino.JUMP_VEL  

def test_small_cactus_position():
    """Тест начальной позиции маленького кактуса."""
    cactus = SmallCactus([Mock(), Mock(), Mock()])
    assert cactus.rect.y == 325  
    assert cactus.rect.x == SCREEN_WIDTH  

def test_large_cactus_position():
    """Тест начальной позиции большого кактуса."""
    cactus = LargeCactus([Mock(), Mock(), Mock()])
    assert cactus.rect.y == 300  
    assert cactus.rect.x == SCREEN_WIDTH  

@pytest.mark.parametrize("userInput, expected_state", [
    ({pygame.K_UP: True}, (False, False, True)),  # Прыжок
    ({pygame.K_UP: False, pygame.K_DOWN: False}, (False, True, False)),  # Бег
])
def test_dinosaur_update(userInput, expected_state, dino):
    """Тест обновления состояния динозавра."""
    with patch('pygame.key.get_pressed', return_value=userInput):
        dino.update(userInput)
    assert (dino.dino_duck, dino.dino_run, dino.dino_jump) == expected_state

@pytest.mark.parametrize("initial_x, expected_x", [
    (SCREEN_WIDTH + 100, SCREEN_WIDTH + 80),  # Движение облака
])
def test_cloud_update_position(initial_x, expected_x):
    """Проверка перемещения облака."""
    cloud = Cloud()
    cloud.x = initial_x
    cloud.update()
    if cloud.x <= 0:
        assert cloud.x > SCREEN_WIDTH  # Проверяем обнуление позиции
    else:
        assert cloud.x == expected_x
 
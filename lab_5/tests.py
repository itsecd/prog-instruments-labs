import pytest
import pygame
from objects import Road, Player, Nitro, Obstacle, HEIGHT


@pytest.fixture(scope="module", autouse=True)
def init_pygame():
    pygame.init()
    yield
    pygame.quit()


def test_road_initialization():
    road = Road()
    assert road.x == 30
    assert road.y1 == 0
    assert road.y2 == -512


def test_player_initialization():
    player = Player(100, 200, 1)
    assert player.rect.x == 100
    assert player.rect.y == 200
    assert player.image.get_size() == (48, 82)


def test_nitro_initialization():
    nitro = Nitro(100, 200)
    assert nitro.rect.x == 100
    assert nitro.rect.y == 200
    assert nitro.gas == 0


@pytest.mark.parametrize(
    "left, right, expected_x",
    [
        (True, False, 95),
        (False, True, 105),
        (False, False, 100),
    ]
)
def test_player_movement(left, right, expected_x):
    player = Player(100, 200, 1)
    player.update(left, right)
    assert player.rect.x == expected_x


@pytest.mark.parametrize(
    "nitro_on, expected_gas",
    [
        (True, -1),
        (False, 1),
    ]
)
def test_nitro_update(nitro_on, expected_gas):
    nitro = Nitro(100, 200)
    nitro.update(nitro_on)
    assert nitro.gas == expected_gas


def test_obstacle_initialization():
    obstacle = Obstacle(1)
    assert obstacle.rect.y == -100
    assert obstacle.rect.x in [50, 95, 142, 190]


@pytest.mark.parametrize(
    "speed, expected_y",
    [
        (5, -95),
        (0, -100)
    ]
)
def test_obstacle_update(speed, expected_y):
    obstacle = Obstacle(1)
    obstacle.update(speed)
    assert obstacle.rect.y == expected_y


def test_player_collision_with_obstacle():
    player = Player(100, 200, 1)
    obstacle = Obstacle(1)
    obstacle.rect.x = 100
    obstacle.rect.y = 200
    assert pygame.sprite.collide_mask(player, obstacle) is not None


def test_road_reset():
    road = Road()
    road.reset()
    assert road.x == 30
    assert road.y1 == 0
    assert road.y2 == -512


@pytest.mark.parametrize(
    "initial_x, move_left, move_right, expected_x",
    [
        (100, True, False, 95),
        (100, False, True, 105),
        (40, True, False, 40),
        (202, False, True, 202)
    ]
)
def test_player_movement_on_road(initial_x, move_left, move_right, expected_x):
    road = Road()
    player = Player(initial_x, 200, 1)

    player.update(move_left, move_right)

    assert player.rect.x == expected_x


@pytest.mark.parametrize(
    "initial_y, speed, expected_y",
    [
        (0, 5, 5),
        (0, 0, 0),
    ]
)
def test_road_update_and_player_position(initial_y, speed, expected_y):
    road = Road()
    player = Player(100, initial_y, 1)

    road.update(speed)

    player.rect.y += speed
    assert player.rect.y == expected_y

    assert road.y1 == initial_y + speed
    assert road.y2 == -HEIGHT + speed

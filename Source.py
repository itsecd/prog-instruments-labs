import pygame
import random
import sys
import time
from enum import Enum
from typing import List, Tuple, Optional

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class GameState(Enum):
    MENU = 0
    PLAYING = 1
    GAME_OVER = 2
    PAUSED = 3

class Colors:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 120, 255)
    DARK_GREEN = (0, 180, 0)
    GRAY = (100, 100, 100)
    YELLOW = (255, 255, 0)
    PURPLE = (180, 0, 255)

class Snake:
    def __init__(self, start_pos: Tuple[int, int], block_size: int = 20):
        self.body = [start_pos]
        self.direction = Direction.RIGHT
        self.next_direction = Direction.RIGHT
        self.block_size = block_size
        self.grow_pending = 3
        self.speed = 10
        self.last_move_time = 0
    
    def change_direction(self, new_direction: Direction):
        opposite_directions = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        
        if new_direction != opposite_directions.get(self.direction):
            self.next_direction = new_direction
    
    def move(self, current_time: float):
        if current_time - self.last_move_time < 1.0 / self.speed:
            return False
            
        self.direction = self.next_direction
        self.last_move_time = current_time
        
        head_x, head_y = self.body[ 0 ]
        dx, dy = self.direction.value
        
        new_head = (
            (head_x + dx * self.block_size) % 800,
            (head_y + dy * self.block_size) % 600
        )
        
        self.body.insert(0, new_head)
        
        if self.grow_pending > 0:
            self.grow_pending -= 1
        else:
            self.body.pop()
            
        return True
    
    def grow(self):
        self.grow_pending += 1
        if len(self.body) % 5 == 0:
            self.speed = min(self.speed + 0.5, 20)
    
    def check_self_collision(self) -> bool:
        return self.body[0] in self.body[1:]
    
    def get_head_position(self) -> Tuple[int, int]:
        return self.body[0]
    
    def draw(self, screen: pygame.Surface):
        for i, (x, y) in enumerate(self.body):
            color = Colors.DARK_GREEN if i > 0 else Colors.GREEN
            pygame.draw.rect(screen, color, (x, y, self.block_size - 2, self.block_size - 2))
            
            if i == 0:
                eye_size = 4
                eye_offset = 5
                
                if self.direction == Direction.RIGHT:
                    left_eye = (x + self.block_size - eye_offset, y + eye_offset)
                    right_eye = (x + self.block_size - eye_offset, y + self.block_size - eye_offset)
                elif self.direction == Direction.LEFT:
                    left_eye = (x + eye_offset, y + eye_offset)
                    right_eye = (x + eye_offset, y + self.block_size - eye_offset)
                elif self.direction == Direction.UP:
                    left_eye = (x + eye_offset, y + eye_offset)
                    right_eye = (x + self.block_size - eye_offset, y + eye_offset)
                else:
                    left_eye = (x + eye_offset, y + self.block_size - eye_offset)
                    right_eye = (x + self.block_size - eye_offset, y + self.block_size - eye_offset)
                
                pygame.draw.circle(screen, Colors.BLACK, left_eye, eye_size)
                pygame.draw.circle(screen, Colors.BLACK, right_eye, eye_size)

class Food:
    def __init__(self, block_size: int = 20):
        self.block_size = block_size
        self.position = (0, 0)
        self.spawn_time = 0
        self.lifetime = 8
        self.randomize_position()
    
    def randomize_position(self, snake_body: Optional[List[Tuple[int, int]]] = None):
        if snake_body is None:
            snake_body = []
            
        max_x = 800 // self.block_size - 1
        max_y = 600 // self.block_size - 1
        
        while True:
            new_pos = (
                random.randint(0, max_x) * self.block_size,
                random.randint(0, max_y) * self.block_size
            )
            if new_pos not in snake_body:
                self.position = new_pos
                self.spawn_time = time.time()
                break
    
    def should_despawning(self, current_time: float) -> bool:
        return current_time - self.spawn_time > self.lifetime
    
    def draw(self, screen: pygame.Surface, current_time: float):
        x, y = self.position
        
        time_left = self.lifetime - (current_time - self.spawn_time)
        if time_left < 2.0:
            if int(time_left * 3) % 2 == 0:
                return
        
        pygame.draw.rect(screen, Colors.RED, (x, y, self.block_size - 2, self.block_size - 2))
        
        stem_color = (139, 69, 19)
        leaf_color = Colors.DARK_GREEN
        
        pygame.draw.rect(screen, stem_color, (x + self.block_size // 2 - 1, y - 3, 2, 4))
        pygame.draw.ellipse(screen, leaf_color, (x + self.block_size // 2 + 1, y - 2, 4, 3))

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Змейка - Игра")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        self.block_size = 20
        self.snake = Snake((100, 100), self.block_size)
        self.food = Food(self.block_size)
        self.score = 0
        self.high_score = 0
        self.state = GameState.MENU
        self.last_food_check = 0
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if self.state == GameState.MENU:
                    if event.key == pygame.K_SPACE:
                        self.start_game()
                
                elif self.state == GameState.PLAYING:
                    if event.key == pygame.K_UP:
                        self.snake.change_direction(Direction.UP)
                    elif event.key == pygame.K_DOWN:
                        self.snake.change_direction(Direction.DOWN)
                    elif event.key == pygame.K_LEFT:
                        self.snake.change_direction(Direction.LEFT)
                    elif event.key == pygame.K_RIGHT:
                        self.snake.change_direction(Direction.RIGHT)
                    elif event.key == pygame.K_p:
                        self.state = GameState.PAUSED
                    elif event.key == pygame.K_ESCAPE:
                        self.state = GameState.MENU
                
                elif self.state == GameState.PAUSED:
                    if event.key == pygame.K_p:
                        self.state = GameState.PLAYING
                    elif event.key == pygame.K_ESCAPE:
                        self.state = GameState.MENU
                
                elif self.state == GameState.GAME_OVER:
                    if event.key == pygame.K_SPACE:
                        self.start_game()
                    elif event.key == pygame.K_ESCAPE:
                        self.state = GameState.MENU
        
        return True
    
    def start_game(self):
        self.snake = Snake((100, 100), self.block_size)
        self.food.randomize_position(self.snake.body)
        self.score = 0
        self.state = GameState.PLAYING
    
    def update(self):
        if self.state != GameState.PLAYING:
            return
            
        current_time = time.time()
        
        if self.snake.move(current_time):
            if self.snake.check_self_collision():
                self.game_over()
                return
            
            if self.snake.get_head_position() == self.food.position:
                self.snake.grow()
                self.score += 10
                self.food.randomize_position(self.snake.body)
            
            if current_time - self.last_food_check > 1.0:
                if self.food.should_despawning(current_time):
                    self.food.randomize_position(self.snake.body)
                self.last_food_check = current_time
    
    def game_over(self):
        self.high_score = max(self.high_score, self.score)
        self.state = GameState.GAME_OVER
    
    def draw_grid(self):
        for x in range(0, 800, self.block_size):
            pygame.draw.line(self.screen, Colors.GRAY, (x, 0), (x, 600), 1)
        for y in range(0, 600, self.block_size):
            pygame.draw.line(self.screen, Colors.GRAY, (0, y), (800, y), 1)
    
    def draw_menu(self):
        self.screen.fill(Colors.BLACK)
        
        title = self.font.render("ЗМЕЙКА", True, Colors.GREEN)
        start_text = self.font.render("Нажмите ПРОБЕЛ для начала игры", True, Colors.WHITE)
        controls_text = self.small_font.render("Управление: Стрелки | Пауза: P | Выход: ESC", True, Colors.GRAY)
        
        self.screen.blit(title, (400 - title.get_width() // 2, 200))
        self.screen.blit(start_text, (400 - start_text.get_width() // 2, 300))
        self.screen.blit(controls_text, (400 - controls_text.get_width() // 2, 350))
        
        demo_snake = [(300 + i * 20, 400) for i in range(5)]
        for i, pos in enumerate(demo_snake):
            color = Colors.DARK_GREEN if i > 0 else Colors.GREEN
            pygame.draw.rect(self.screen, color, (pos[0], pos[1], 18, 18))
    
    def draw_playing(self):
        self.screen.fill(Colors.BLACK)
        self.draw_grid()
        
        current_time = time.time()
        self.food.draw(self.screen, current_time)
        self.snake.draw(self.screen)
        
        score_text = self.font.render(f"Счет: {self.score}", True, Colors.WHITE)
        speed_text = self.small_font.render(f"Скорость: {self.snake.speed:.1f}", True, Colors.GRAY)
        high_score_text = self.small_font.render(f"Рекорд: {self.high_score}", True, Colors.YELLOW)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(speed_text, (10, 50))
        self.screen.blit(high_score_text, (10, 80))
        
        food_time_left = self.food.lifetime - (current_time - self.food.spawn_time)
        if food_time_left < 5:
            time_text = self.small_font.render(f"Еда исчезнет через: {food_time_left:.1f}с", True, Colors.RED)
            self.screen.blit(time_text, (600, 10))
    
    def draw_paused(self):
        self.draw_playing()
        
        overlay = pygame.Surface((800, 600), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))
        
        pause_text = self.font.render("ПАУЗА", True, Colors.YELLOW)
        continue_text = self.small_font.render("Нажмите P для продолжения", True, Colors.WHITE)
        
        self.screen.blit(pause_text, (400 - pause_text.get_width() // 2, 250))
        self.screen.blit(continue_text, (400 - continue_text.get_width() // 2, 300))
    
    def draw_game_over(self):
        self.screen.fill(Colors.BLACK)
        
        game_over_text = self.font.render("ИГРА ОКОНЧЕНА!", True, Colors.RED)
        score_text = self.font.render(f"Ваш счет: {self.score}", True, Colors.WHITE)
        high_score_text = self.font.render(f"Рекорд: {self.high_score}", True, Colors.YELLOW)
        restart_text = self.font.render("Нажмите ПРОБЕЛ для повторной игры", True, Colors.GREEN)
        menu_text = self.small_font.render("Нажмите ESC для выхода в меню", True, Colors.GRAY)
        
        self.screen.blit(game_over_text, (400 - game_over_text.get_width() // 2, 200))
        self.screen.blit(score_text, (400 - score_text.get_width() // 2, 250))
        self.screen.blit(high_score_text, (400 - high_score_text.get_width() // 2, 300))
        self.screen.blit(restart_text, (400 - restart_text.get_width() // 2, 350))
        self.screen.blit(menu_text, (400 - menu_text.get_width() // 2, 400))
    
    def draw(self):
        if self.state == GameState.MENU:
            self.draw_menu()
        elif self.state == GameState.PLAYING:
            self.draw_playing()
        elif self.state == GameState.PAUSED:
            self.draw_paused()
        elif self.state == GameState.GAME_OVER:
            self.draw_game_over()
        
        pygame.display.flip()
    
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()

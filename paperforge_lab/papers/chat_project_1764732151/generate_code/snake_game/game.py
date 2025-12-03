import pygame
from pygame.locals import *
from config import SCREEN_WIDTH, SCREEN_HEIGHT, BLACK, WHITE, GREEN, RED, FPS, BLOCK_SIZE, FONT_NAME, FONT_SIZE
from snake import Snake
from food import Food

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)
        self.snake = Snake()
        self.food = Food()
        self.score = 0
        self.game_over = False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    return False
                self.snake.handle_keys(event)
        return True

    def update(self):
        if not self.game_over:
            self.snake.update()
            self.check_collision()
            self.check_food_collision()

    def check_collision(self):
        head = self.snake.get_head_position()
        if head[0] < 0 or head[0] >= SCREEN_WIDTH or head[1] < 0 or head[1] >= SCREEN_HEIGHT:
            self.game_over = True
        for segment in self.snake.positions[1:]:
            if head == segment:
                self.game_over = True

    def check_food_collision(self):
        if self.snake.get_head_position() == self.food.position:
            self.snake.grow()
            self.food.randomize_position()
            self.score += 1

    def render(self):
        self.screen.fill(BLACK)
        self.snake.render(self.screen)
        self.food.render(self.screen)
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (5, 5))
        if self.game_over:
            game_over_text = self.font.render("Game Over! Press ESC to quit.", True, RED)
            self.screen.blit(game_over_text, (SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2))
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)

if __name__ == "__main__":
    game = Game()
    game.run()
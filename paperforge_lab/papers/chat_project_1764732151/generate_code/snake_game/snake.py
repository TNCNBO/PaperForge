import pygame
from config import SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE, GREEN, BLACK

class Snake:
    def __init__(self):
        self.length = 1
        self.positions = [(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)]
        self.direction = pygame.K_RIGHT  # Default direction
        self.color = GREEN
        self.score = 0

    def get_head_position(self):
        return self.positions[0]

    def update(self):
        head = self.get_head_position()
        x, y = head

        if self.direction == pygame.K_UP:
            y -= BLOCK_SIZE
        elif self.direction == pygame.K_DOWN:
            y += BLOCK_SIZE
        elif self.direction == pygame.K_LEFT:
            x -= BLOCK_SIZE
        elif self.direction == pygame.K_RIGHT:
            x += BLOCK_SIZE

        # Update positions
        self.positions.insert(0, (x, y))
        if len(self.positions) > self.length:
            self.positions.pop()

    def reset(self):
        self.length = 1
        self.positions = [(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)]
        self.direction = pygame.K_RIGHT
        self.score = 0

    def render(self, surface):
        for p in self.positions:
            pygame.draw.rect(surface, self.color, pygame.Rect(p[0], p[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(surface, BLACK, pygame.Rect(p[0], p[1], BLOCK_SIZE, BLOCK_SIZE), 1)

    def handle_keys(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and self.direction != pygame.K_DOWN:
                self.direction = pygame.K_UP
            elif event.key == pygame.K_DOWN and self.direction != pygame.K_UP:
                self.direction = pygame.K_DOWN
            elif event.key == pygame.K_LEFT and self.direction != pygame.K_RIGHT:
                self.direction = pygame.K_LEFT
            elif event.key == pygame.K_RIGHT and self.direction != pygame.K_LEFT:
                self.direction = pygame.K_RIGHT

    def grow(self):
        """增加蛇的长度"""
        self.length += 1
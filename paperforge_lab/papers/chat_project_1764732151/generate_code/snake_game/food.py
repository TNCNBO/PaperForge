import pygame
import random
from config import SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE, RED, BLACK

class Food:
    """
    Represents the food in the Snake game.
    """

    def __init__(self):
        """
        Initialize the food with a random position.
        """
        self.position = (0, 0)
        self.color = RED
        self.randomize_position()

    def randomize_position(self):
        """
        Randomly place the food within the screen boundaries.
        """
        self.position = (
            random.randint(0, (SCREEN_WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
            random.randint(0, (SCREEN_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        )

    def render(self, surface):
        """
        Draw the food on the given surface.
        """
        rect = pygame.Rect(
            (self.position[0], self.position[1]),
            (BLOCK_SIZE, BLOCK_SIZE)
        )
        pygame.draw.rect(surface, self.color, rect)
        pygame.draw.rect(surface, BLACK, rect, 1)  # Border for visibility
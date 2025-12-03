"""
Configuration file for the Tetris game.
Defines core game constants like grid dimensions, colors, and speed.
"""

# Grid dimensions
GRID_ROWS = 20
GRID_COLUMNS = 10

# Block size (in pixels)
BLOCK_SIZE = 30

# Colors (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# Block colors
BLOCK_COLORS = [CYAN, BLUE, ORANGE, YELLOW, GREEN, MAGENTA, RED]

# Game speed (milliseconds per frame)
GAME_SPEED = 500

# Screen dimensions
SCREEN_WIDTH = GRID_COLUMNS * BLOCK_SIZE
SCREEN_HEIGHT = GRID_ROWS * BLOCK_SIZE

# Font settings
FONT_NAME = "Arial"
FONT_SIZE = 24
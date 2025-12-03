import pygame
from config import GRID_ROWS, GRID_COLUMNS, GAME_SPEED
from block import Block

class GameLogic:
    def __init__(self):
        self.grid = [[0 for _ in range(GRID_COLUMNS)] for _ in range(GRID_ROWS)]
        self.current_block = self._get_random_block()
        self.next_block = self._get_random_block()
        self.score = 0
        self.game_over = False

    def _get_random_block(self):
        """Return a new random block."""
        return Block(shape_index=pygame.time.get_ticks() % 7, x=GRID_COLUMNS // 2 - 2, y=0)

    def move_block(self, direction):
        """Move the current block left, right, or down."""
        if direction == "left":
            self.current_block.x -= 1
            if self._check_collision():
                self.current_block.x += 1
        elif direction == "right":
            self.current_block.x += 1
            if self._check_collision():
                self.current_block.x -= 1
        elif direction == "down":
            self.current_block.y += 1
            if self._check_collision():
                self.current_block.y -= 1
                self._lock_block()

    def rotate_block(self, clockwise=True):
        """Rotate the current block clockwise or counter-clockwise."""
        if clockwise:
            self.current_block.rotate_clockwise()
        else:
            self.current_block.rotate_counterclockwise()
        if self._check_collision():
            if clockwise:
                self.current_block.rotate_counterclockwise()
            else:
                self.current_block.rotate_clockwise()

    def _check_collision(self):
        """Check if the current block collides with the grid boundaries or other blocks."""
        for y, row in enumerate(self.current_block.get_grid_positions()):
            for x, cell in enumerate(row):
                if cell:
                    if x < 0 or x >= GRID_COLUMNS or y >= GRID_ROWS or (y >= 0 and self.grid[y][x]):
                        return True
        return False

    def _lock_block(self):
        """Lock the current block into the grid and check for line clears."""
        for y, row in enumerate(self.current_block.get_grid_positions()):
            for x, cell in enumerate(row):
                if cell and y >= 0:
                    self.grid[y][x] = self.current_block.color
        self._clear_lines()
        self.current_block = self.next_block
        self.next_block = self._get_random_block()
        if self._check_collision():
            self.game_over = True

    def _clear_lines(self):
        """Clear completed lines and update the score."""
        lines_cleared = 0
        for y in range(GRID_ROWS):
            if all(self.grid[y]):
                lines_cleared += 1
                for y2 in range(y, 0, -1):
                    self.grid[y2] = self.grid[y2 - 1][:]
                self.grid[0] = [0] * GRID_COLUMNS
        if lines_cleared > 0:
            self.score += lines_cleared * 100

    def update(self):
        """Update the game state (e.g., move block down automatically)."""
        if not self.game_over:
            self.move_block("down")
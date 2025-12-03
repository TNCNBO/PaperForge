import pygame
from config import BLOCK_COLORS, BLOCK_SIZE, GRID_COLUMNS, GRID_ROWS

class Block:
    """Represents a Tetrimino block with shape, rotation, and position."""

    # Define the shapes of all Tetriminos
    SHAPES = [
        [[1, 1, 1, 1]],  # I-shape
        [[1, 0, 0], [1, 1, 1]],  # J-shape
        [[0, 0, 1], [1, 1, 1]],  # L-shape
        [[1, 1], [1, 1]],  # O-shape
        [[0, 1, 1], [1, 1, 0]],  # S-shape
        [[0, 1, 0], [1, 1, 1]],  # T-shape
        [[1, 1, 0], [0, 1, 1]]   # Z-shape
    ]

    def __init__(self, shape_index, x=GRID_COLUMNS // 2 - 1, y=0):
        """Initialize a block with a specific shape and starting position.

        Args:
            shape_index (int): Index of the shape in SHAPES.
            x (int): Initial x-coordinate on the grid.
            y (int): Initial y-coordinate on the grid.
        """
        self.shape = self.SHAPES[shape_index]
        self.color = BLOCK_COLORS[shape_index]
        self.x = x
        self.y = y
        self.rotation = 0

    def rotate_clockwise(self):
        """Rotate the block 90 degrees clockwise."""
        self.shape = [list(row) for row in zip(*self.shape[::-1])]

    def rotate_counterclockwise(self):
        """Rotate the block 90 degrees counterclockwise."""
        self.shape = [list(row) for row in zip(*self.shape)][::-1]

    def get_rotated_shape(self, clockwise=True):
        """Return a rotated version of the block's shape without modifying it.

        Args:
            clockwise (bool): If True, rotate clockwise; else counterclockwise.

        Returns:
            list: Rotated shape.
        """
        if clockwise:
            return [list(row) for row in zip(*self.shape[::-1])]
        else:
            return [list(row) for row in zip(*self.shape)][::-1]

    def get_grid_positions(self):
        """Return the positions of all blocks in the current shape relative to the grid.

        Returns:
            list: List of (x, y) tuples representing block positions.
        """
        positions = []
        for i, row in enumerate(self.shape):
            for j, cell in enumerate(row):
                if cell:
                    positions.append((self.x + j, self.y + i))
        return positions
import pygame
from config import GRID_ROWS, GRID_COLUMNS, BLOCK_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, BLACK, WHITE, BLOCK_COLORS, FONT_NAME, FONT_SIZE
from block import Block

class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris")
        self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

    def draw_grid(self, grid):
        """Draw the game grid."""
        self.screen.fill(BLACK)
        for y in range(GRID_ROWS):
            for x in range(GRID_COLUMNS):
                if grid[y][x]:
                    pygame.draw.rect(
                        self.screen,
                        BLOCK_COLORS[grid[y][x] - 1],
                        (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
                        0
                    )
                pygame.draw.rect(
                    self.screen,
                    WHITE,
                    (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
                    1
                )

    def draw_block(self, block):
        """Draw the current block."""
        if block:
            for y, row in enumerate(block.shape):
                for x, cell in enumerate(row):
                    if cell:
                        pygame.draw.rect(
                            self.screen,
                            BLOCK_COLORS[block.shape_index],
                            (
                                (block.x + x) * BLOCK_SIZE,
                                (block.y + y) * BLOCK_SIZE,
                                BLOCK_SIZE,
                                BLOCK_SIZE
                            ),
                            0
                        )

    def draw_score(self, score):
        """Draw the current score."""
        score_text = self.font.render(f"Score: {score}", True, WHITE)
        self.screen.blit(score_text, (GRID_COLUMNS * BLOCK_SIZE + 20, 20))

    def draw_game_over(self):
        """Draw the game over message."""
        game_over_text = self.font.render("GAME OVER", True, WHITE)
        self.screen.blit(
            game_over_text,
            (
                SCREEN_WIDTH // 2 - game_over_text.get_width() // 2,
                SCREEN_HEIGHT // 2 - game_over_text.get_height() // 2
            )
        )

    def update_display(self):
        """Update the display."""
        pygame.display.flip()
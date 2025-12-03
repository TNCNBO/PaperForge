import pygame
from pygame.locals import *
import sys
from game_logic import GameLogic
from render import Renderer
from config import SCREEN_WIDTH, SCREEN_HEIGHT, GAME_SPEED

def main():
    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption('Tetris')
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    # Initialize game components
    game_logic = GameLogic()
    renderer = Renderer()

    # Game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_LEFT:
                    game_logic.move_block('left')
                elif event.key == K_RIGHT:
                    game_logic.move_block('right')
                elif event.key == K_DOWN:
                    game_logic.move_block('down')
                elif event.key == K_UP:
                    game_logic.rotate_block()
                elif event.key == K_SPACE:
                    # Hard drop
                    while game_logic.move_block('down'):
                        pass
                elif event.key == K_p:
                    # Pause game
                    pass

        # Update game state
        game_logic.update()

        # Render
        screen.fill((0, 0, 0))  # Clear screen
        renderer.draw_grid(game_logic.grid)
        renderer.draw_block(game_logic.current_block)
        renderer.draw_score(game_logic.score)
        if game_logic.game_over:
            renderer.draw_game_over()
        renderer.update_display()

        # Cap the frame rate
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
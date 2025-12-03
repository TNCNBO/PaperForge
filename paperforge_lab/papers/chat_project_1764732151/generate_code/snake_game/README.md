# Snake Game

A classic Snake game implemented in Python using Pygame for graphical rendering.

## Features
- Snake movement controlled by arrow keys.
- Food spawning randomly on the screen.
- Snake growth upon eating food.
- Collision detection with walls and self.
- Score tracking and display.
- Game over screen.

## Installation
1. Ensure you have Python installed (recommended version 3.8 or higher).
2. Clone this repository or download the source code.
3. Navigate to the project directory (`snake_game`).
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Game
1. From the project directory, run:
   ```bash
   python main.py
   ```
2. Use the arrow keys to control the snake.
3. Press `ESC` or close the window to quit the game.

## Project Structure
```
snake_game/
├── main.py                 # Entry point for the game
├── game.py                 # Core game logic and loop
├── snake.py                # Snake class implementation
├── food.py                 # Food class implementation
├── config.py               # Configuration settings (e.g., screen size, colors)
├── assets/                 # Folder for game assets (optional)
│   ├── sounds/             # Sound effects
│   └── fonts/              # Font files
├── requirements.txt        # List of dependencies
└── README.md               # Documentation and instructions
```

## Dependencies
- `pygame==2.5.2`

## Optional Features
- Sound effects (requires `pygame.mixer`).
- Custom fonts (place `.ttf` files in `assets/fonts/`).

## License
This project is open-source and available under the MIT License.

# Tetris Game

A classic Tetris game implemented in Python using Pygame. This project includes all core gameplay features such as falling blocks, rotation, scoring, and game over detection.

## Features

- **Classic Tetris Gameplay**: Includes all 7 Tetrimino shapes with rotation capabilities.
- **Keyboard Controls**: Move blocks left/right, rotate clockwise/counter-clockwise, and hard drop.
- **Scoring System**: Points awarded for line clears and combos.
- **Game States**: Start screen, gameplay, pause, and game over states.
- **Sound Effects**: Optional sound effects for block movements and line clears.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd tetris_game
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Game**:
   ```bash
   python main.py
   ```

## Project Structure

- `main.py`: Entry point for the game.
- `game_logic.py`: Core game mechanics (block movement, collision, scoring).
- `block.py`: Block class (shape definitions and rotations).
- `render.py`: Rendering logic (displaying the game board).
- `config.py`: Game settings (colors, grid size, speed).
- `assets/`: Folder for game assets (fonts, sounds).

## Dependencies

- **Required**: `pygame>=2.0`
- **Optional**: `pygame-mixer` (for sound effects and music)

## Controls

- **Left Arrow**: Move block left.
- **Right Arrow**: Move block right.
- **Down Arrow**: Move block down faster.
- **Up Arrow**: Rotate block clockwise.
- **Space**: Hard drop (instantly drop block to bottom).
- **P**: Pause/resume game.

## Contributing

Feel free to fork the repository and submit pull requests for improvements or bug fixes.

## License

This project is open-source and available under the MIT License.
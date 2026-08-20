# Adaptive Tic-Tac-Toe RL

A graphical Tic-Tac-Toe game in which a human player competes against a Q-learning agent. The agent adjusts its difficulty based on the player's recent performance and improves its Q-values over multiple episodes.

## Requirements

- Python 3.8 or newer
- Pygame
- NumPy

## Installation

Create and activate a virtual environment if desired, then install the dependencies:

```bash
python -m pip install pygame numpy
```

## Run

From this directory, run:

```bash
python ttt.py
```

A Pygame window will open at 1200x700 pixels.

## How to Play

- You play as `X`.
- The RL agent plays as `O`.
- Click an empty cell to make a move.
- Press `R` to restart the current game.
- Press `Esc` or close the window to quit.
- After a game ends, the next episode starts automatically after two seconds.

## Adaptive Difficulty

Difficulty is calculated from the player's results in the most recent 10 completed games:

| Player win rate | Difficulty |
| --- | --- |
| Below 25% | Very Easy |
| 25% to below 45% | Easy |
| 45% to below 65% | Normal |
| 65% to below 80% | Hard |
| 80% or higher | Expert |

The first three games remain at Very Easy while enough results are collected. Higher levels reduce random exploration, rely more strongly on learned Q-values, and add tactical behavior such as winning immediately, blocking the player, and preferring the center or corners.

## Q-Learning Details

- State: the 3x3 board encoded as a nine-character string.
- Actions: the nine board positions.
- Learning rate (`alpha`): `0.35`.
- Discount factor (`gamma`): `0.90`.
- Exploration starts at `1.0`, decays after each episode, and has a minimum of `0.05`.
- Rewards: RL win `+10`, draw `+3`, player win `-10`.

The Q-table is preserved between games during the same run. It is stored in memory and is not saved to disk when the program exits.

## On-Screen Information

The right-hand panel displays the current difficulty, episode statistics, exploration rate, number of learned states, current board state, the latest RL decision, Q-values for available moves, and recent Q-table updates. A final training summary is printed in the terminal when the game closes.

## Project File

- `adaptive_ttt.py` - main Pygame game and Q-learning implementation

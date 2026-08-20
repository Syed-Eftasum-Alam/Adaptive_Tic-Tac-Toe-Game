import pygame
import random
import numpy as np
from collections import defaultdict

# ============================================================
# ADAPTIVE TIC-TAC-TOE USING Q-LEARNING
# ============================================================
# Player = X
# RL Agent = O
#
# IMPORTANT:
# This version automatically plays MANY episodes.
# The RL agent becomes harder according to the player's skill.
#
# Difficulty increases through:
#   1. Less random exploration
#   2. Stronger use of learned Q-values
#   3. Tactical move selection at higher difficulty
#   4. Difficulty level displayed live
#
# The Q-table is NOT reset between games.
#
# Install:
#     pip install pygame numpy
#
# Run:
#     python adaptive_tictactoe_rl_adaptive.py
# ============================================================

pygame.init()

WIDTH = 1200
HEIGHT = 700

BOARD_SIZE = 540
CELL_SIZE = BOARD_SIZE // 3
BOARD_X = 30
BOARD_Y = 80

PANEL_X = 610
PANEL_WIDTH = 560

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Adaptive Tic-Tac-Toe - RL Dynamic Difficulty")

clock = pygame.time.Clock()

# ------------------------------------------------------------
# Fonts
# ------------------------------------------------------------
font = pygame.font.SysFont("arial", 20)
small_font = pygame.font.SysFont("arial", 16)
medium_font = pygame.font.SysFont("arial", 25)
large_font = pygame.font.SysFont("arial", 36)

# ------------------------------------------------------------
# Colors
# ------------------------------------------------------------
WHITE = (245, 245, 245)
BLACK = (25, 25, 25)
GRAY = (120, 120, 120)
LIGHT_GRAY = (230, 230, 230)
BLUE = (40, 100, 220)
RED = (215, 60, 60)
GREEN = (45, 160, 90)
PURPLE = (125, 70, 180)
ORANGE = (220, 140, 35)

# ============================================================
# Q-LEARNING
# ============================================================

Q = defaultdict(lambda: np.zeros(9, dtype=float))

ALPHA = 0.35
GAMMA = 0.90

# Exploration decreases over time.
epsilon = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.965

# ============================================================
# GAME VARIABLES
# ============================================================

board = [" "] * 9

current_turn = "X"

game_over = False
winner = None

episode = 1

player_wins = 0
agent_wins = 0
draws = 0

# Recent results are used to estimate player skill.
recent_results = []

# Number of completed games used for skill estimation.
SKILL_WINDOW = 10

# Difficulty:
# 1 = Very Easy
# 2 = Easy
# 3 = Normal
# 4 = Hard
# 5 = Expert
difficulty = 1

difficulty_name = "Very Easy"

# RL information
previous_state = None
previous_action = None

last_state = "---------"
last_action = "-"
last_reward = 0

q_before = 0.0
q_after = 0.0

update_history = []

message = "You are X. Make your first move."

# Time before automatically starting the next game.
game_end_timer = 0
AUTO_RESTART_DELAY = 2.0


# ============================================================
# BOARD FUNCTIONS
# ============================================================

def board_to_state():
    return "".join(
        "." if c == " " else c
        for c in board
    )


def available_moves():
    return [
        i for i in range(9)
        if board[i] == " "
    ]


def check_winner():
    winning_lines = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6)
    ]

    for a, b, c in winning_lines:

        if (
            board[a] != " "
            and board[a] == board[b]
            and board[b] == board[c]
        ):
            return board[a]

    if not available_moves():
        return "Draw"

    return None


# ============================================================
# DIFFICULTY SYSTEM
# ============================================================

def calculate_difficulty():
    """
    Dynamically determine difficulty from recent player results.

    The agent becomes harder when the player repeatedly performs well.

    Player win rate over recent games:
        >= 80% -> Expert
        >= 65% -> Hard
        >= 45% -> Normal
        >= 25% -> Easy
        < 25%  -> Very Easy

    The difficulty also considers the number of games played.
    """

    if len(recent_results) < 3:
        return 1

    window = recent_results[-SKILL_WINDOW:]

    player_win_rate = window.count("X") / len(window)

    if len(window) >= 3 and player_win_rate >= 0.80:
        return 5

    if len(window) >= 3 and player_win_rate >= 0.65:
        return 4

    if len(window) >= 3 and player_win_rate >= 0.45:
        return 3

    if len(window) >= 3 and player_win_rate >= 0.25:
        return 2

    return 1


def update_difficulty():
    global difficulty
    global difficulty_name

    difficulty = calculate_difficulty()

    names = {
        1: "Very Easy",
        2: "Easy",
        3: "Normal",
        4: "Hard",
        5: "Expert"
    }

    difficulty_name = names[difficulty]


# ============================================================
# TIC-TAC-TOE TACTICAL HELPERS
# ============================================================

def find_winning_move(symbol):
    """
    Return a move that immediately wins for the symbol.
    """

    for move in available_moves():

        board[move] = symbol

        result = check_winner()

        board[move] = " "

        if result == symbol:
            return move

    return None


def find_blocking_move():
    """
    Block the player's immediate winning move.
    """

    return find_winning_move("X")


def strategic_move():
    """
    A small rule-based tactical layer used at higher difficulty.

    This is not replacing Q-learning.
    It is used to make the increasing difficulty visible
    and meaningful in a small Tic-Tac-Toe environment.
    """

    moves = available_moves()

    # 1. Win immediately if possible.
    winning = find_winning_move("O")

    if winning is not None:
        return winning

    # 2. Block the player.
    blocking = find_blocking_move()

    if blocking is not None:
        return blocking

    # 3. Prefer center.
    if 4 in moves:
        return 4

    # 4. Prefer corners.
    corners = [
        m for m in [0, 2, 6, 8]
        if m in moves
    ]

    if corners:
        return random.choice(corners)

    return random.choice(moves)


# ============================================================
# RL ACTION SELECTION
# ============================================================

def choose_rl_action(state):
    """
    Difficulty-dependent epsilon-greedy selection.

    Very Easy:
        Mostly random.

    Easy:
        Some learned Q-values.

    Normal:
        Mostly learned Q-values.

    Hard:
        Strong learned Q-values + tactical decisions.

    Expert:
        Very strong Q-values + tactical decisions.
    """

    moves = available_moves()

    if not moves:
        return None

    q_values = Q[state]

    # Effective exploration becomes smaller as difficulty rises.
    if difficulty == 1:
        effective_epsilon = max(0.65, epsilon)

    elif difficulty == 2:
        effective_epsilon = max(0.40, epsilon * 0.75)

    elif difficulty == 3:
        effective_epsilon = max(0.20, epsilon * 0.45)

    elif difficulty == 4:
        effective_epsilon = max(0.08, epsilon * 0.20)

    else:
        effective_epsilon = 0.02

    # At high difficulty, tactical behavior is used.
    if difficulty >= 4:

        # 70% tactical/learned behavior.
        if random.random() < 0.70:
            tactical = strategic_move()

            if tactical is not None:
                return tactical

    # Exploration.
    if random.random() < effective_epsilon:
        return random.choice(moves)

    # Learned Q-value selection.
    best_value = max(
        q_values[m]
        for m in moves
    )

    best_moves = [
        m for m in moves
        if q_values[m] == best_value
    ]

    return random.choice(best_moves)


# ============================================================
# Q-LEARNING UPDATE
# ============================================================

def update_q(previous_state, action, reward, next_state):

    global q_before
    global q_after

    old_value = Q[previous_state][action]

    next_moves = available_moves()

    if next_moves:
        best_next = max(
            Q[next_state][m]
            for m in next_moves
        )
    else:
        best_next = 0.0

    new_value = old_value + ALPHA * (
        reward +
        GAMMA * best_next -
        old_value
    )

    Q[previous_state][action] = new_value

    q_before = old_value
    q_after = new_value

    update_history.insert(
        0,
        (
            previous_state,
            action + 1,
            reward,
            old_value,
            new_value
        )
    )

    del update_history[8:]


# ============================================================
# REWARD
# ============================================================

def reward_for_result(result):

    if result == "O":
        return 10

    if result == "Draw":
        return 3

    if result == "X":
        return -10

    return 0


# ============================================================
# RESET GAME
# ============================================================

def reset_game():

    global board
    global current_turn
    global game_over
    global winner
    global previous_state
    global previous_action
    global last_state
    global last_action
    global last_reward
    global q_before
    global q_after
    global message
    global game_end_timer

    board = [" "] * 9

    current_turn = "X"

    game_over = False
    winner = None

    previous_state = None
    previous_action = None

    last_state = "---------"
    last_action = "-"
    last_reward = 0

    q_before = 0.0
    q_after = 0.0

    game_end_timer = 0

    message = (
        f"Episode {episode}: "
        f"You are X. Make your move."
    )


# ============================================================
# FINISH GAME
# ============================================================

def finish_game(result):

    global game_over
    global winner
    global player_wins
    global agent_wins
    global draws
    global last_reward
    global episode
    global epsilon
    global game_end_timer
    global message
    global previous_state
    global previous_action

    game_over = True
    winner = result

    reward = reward_for_result(result)

    last_reward = reward

    # Final Q update.
    if (
        previous_state is not None
        and previous_action is not None
    ):

        next_state = board_to_state()

        update_q(
            previous_state,
            previous_action,
            reward,
            next_state
        )

    # Record result.
    recent_results.append(result)

    if result == "X":

        player_wins += 1
        message = "YOU WIN! Difficulty will adapt."

    elif result == "O":

        agent_wins += 1
        message = "RL AGENT WINS! Learning updated."

    else:

        draws += 1
        message = "DRAW! Learning updated."

    # Update adaptive difficulty.
    update_difficulty()

    # Reduce exploration after every episode.
    epsilon = max(
        EPSILON_MIN,
        epsilon * EPSILON_DECAY
    )

    episode += 1

    game_end_timer = pygame.time.get_ticks()


# ============================================================
# RL MOVE
# ============================================================

def make_rl_move():

    global current_turn
    global previous_state
    global previous_action
    global last_state
    global last_action
    global message

    state = board_to_state()

    action = choose_rl_action(state)

    previous_state = state
    previous_action = action

    last_state = state
    last_action = str(action + 1)

    board[action] = "O"

    result = check_winner()

    if result is not None:

        finish_game(result)

    else:

        current_turn = "X"

        message = (
            f"Difficulty: {difficulty_name} | "
            "Your turn."
        )


# ============================================================
# PLAYER MOVE
# ============================================================

def player_move(position):

    global current_turn
    global last_state
    global message

    if game_over:
        return

    if current_turn != "X":
        return

    if board[position] != " ":
        return

    last_state = board_to_state()

    board[position] = "X"

    result = check_winner()

    if result is not None:

        finish_game(result)

        return

    current_turn = "O"

    message = "RL agent is thinking..."

    # Short delay would be more realistic, but immediate action
    # keeps the game responsive.
    make_rl_move()


# ============================================================
# DRAW BOARD
# ============================================================

def draw_board():

    pygame.draw.rect(
        screen,
        WHITE,
        (
            BOARD_X,
            BOARD_Y,
            BOARD_SIZE,
            BOARD_SIZE
        )
    )

    for i in range(1, 3):

        pygame.draw.line(
            screen,
            BLACK,
            (
                BOARD_X + i * CELL_SIZE,
                BOARD_Y
            ),
            (
                BOARD_X + i * CELL_SIZE,
                BOARD_Y + BOARD_SIZE
            ),
            5
        )

        pygame.draw.line(
            screen,
            BLACK,
            (
                BOARD_X,
                BOARD_Y + i * CELL_SIZE
            ),
            (
                BOARD_X + BOARD_SIZE,
                BOARD_Y + i * CELL_SIZE
            ),
            5
        )

    for i, value in enumerate(board):

        row = i // 3
        col = i % 3

        cx = (
            BOARD_X +
            col * CELL_SIZE +
            CELL_SIZE // 2
        )

        cy = (
            BOARD_Y +
            row * CELL_SIZE +
            CELL_SIZE // 2
        )

        if value == "X":

            offset = 55

            pygame.draw.line(
                screen,
                BLUE,
                (
                    cx - offset,
                    cy - offset
                ),
                (
                    cx + offset,
                    cy + offset
                ),
                10
            )

            pygame.draw.line(
                screen,
                BLUE,
                (
                    cx + offset,
                    cy - offset
                ),
                (
                    cx - offset,
                    cy + offset
                ),
                10
            )

        elif value == "O":

            pygame.draw.circle(
                screen,
                RED,
                (
                    cx,
                    cy
                ),
                58,
                10
            )


# ============================================================
# DRAW PANEL
# ============================================================

def draw_panel():

    pygame.draw.rect(
        screen,
        LIGHT_GRAY,
        (
            PANEL_X,
            20,
            PANEL_WIDTH,
            HEIGHT - 40
        ),
        border_radius=12
    )

    title = medium_font.render(
        "RL Learning & Adaptive Difficulty",
        True,
        BLACK
    )

    screen.blit(
        title,
        (
            PANEL_X + 20,
            35
        )
    )

    y = 75

    # --------------------------------------------------------
    # Difficulty
    # --------------------------------------------------------

    difficulty_text = medium_font.render(
        f"DIFFICULTY: {difficulty}/5 - {difficulty_name}",
        True,
        RED if difficulty >= 4 else GREEN
    )

    screen.blit(
        difficulty_text,
        (
            PANEL_X + 20,
            y
        )
    )

    y += 35

    # Difficulty progress bar.
    bar_x = PANEL_X + 20
    bar_y = y
    bar_width = 400
    bar_height = 18

    pygame.draw.rect(
        screen,
        WHITE,
        (
            bar_x,
            bar_y,
            bar_width,
            bar_height
        )
    )

    pygame.draw.rect(
        screen,
        RED,
        (
            bar_x,
            bar_y,
            int(
                bar_width *
                difficulty /
                5
            ),
            bar_height
        )
    )

    y += 32

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = [
        f"Episode: {episode}",
        f"Player wins: {player_wins}",
        f"RL wins: {agent_wins}",
        f"Draws: {draws}",
        f"Exploration (epsilon): {epsilon:.3f}",
        f"Learned states: {len(Q)}",
    ]

    for text in stats:

        rendered = font.render(
            text,
            True,
            BLACK
        )

        screen.blit(
            rendered,
            (
                PANEL_X + 20,
                y
            )
        )

        y += 25

    # --------------------------------------------------------
    # Recent skill estimate
    # --------------------------------------------------------

    y += 5

    if recent_results:

        window = recent_results[-SKILL_WINDOW:]

        player_rate = (
            window.count("X") /
            len(window)
        )

    else:

        player_rate = 0

    skill_text = (
        f"Recent player win rate: "
        f"{player_rate * 100:.0f}%"
    )

    rendered = font.render(
        skill_text,
        True,
        PURPLE
    )

    screen.blit(
        rendered,
        (
            PANEL_X + 20,
            y
        )
    )

    y += 30

    # --------------------------------------------------------
    # Current state
    # --------------------------------------------------------

    pygame.draw.line(
        screen,
        GRAY,
        (
            PANEL_X + 20,
            y
        ),
        (
            PANEL_X + PANEL_WIDTH - 20,
            y
        ),
        2
    )

    y += 10

    rendered = font.render(
        "Current RL State:",
        True,
        BLACK
    )

    screen.blit(
        rendered,
        (
            PANEL_X + 20,
            y
        )
    )

    y += 25

    rendered = medium_font.render(
        board_to_state(),
        True,
        PURPLE
    )

    screen.blit(
        rendered,
        (
            PANEL_X + 20,
            y
        )
    )

    y += 35

    # --------------------------------------------------------
    # Last RL decision
    # --------------------------------------------------------

    decision = [
        f"RL selected position: {last_action}",
        f"Reward: {last_reward:+}",
        f"Q before: {q_before:+.3f}",
        f"Q after:  {q_after:+.3f}",
    ]

    for text in decision:

        rendered = small_font.render(
            text,
            True,
            BLACK
        )

        screen.blit(
            rendered,
            (
                PANEL_X + 20,
                y
            )
        )

        y += 21

    # --------------------------------------------------------
    # Q-value moveset
    # --------------------------------------------------------

    y += 3

    rendered = font.render(
        "LIVE Q-VALUE MOVESET",
        True,
        BLACK
    )

    screen.blit(
        rendered,
        (
            PANEL_X + 20,
            y
        )
    )

    y += 25

    state = board_to_state()

    q_values = Q[state]

    # 3x3 Q-value representation.
    for row in range(3):

        line = ""

        for col in range(3):

            pos = row * 3 + col

            if board[pos] != " ":

                line += (
                    f" {board[pos]:^6} "
                )

            else:

                line += (
                    f" {pos + 1}:{q_values[pos]:+5.2f} "
                )

        rendered = small_font.render(
            line,
            True,
            BLACK
        )

        screen.blit(
            rendered,
            (
                PANEL_X + 20,
                y
            )
        )

        y += 22

    # --------------------------------------------------------
    # Q update history
    # --------------------------------------------------------

    y += 5

    rendered = font.render(
        "Recent Q-table Updates",
        True,
        BLACK
    )

    screen.blit(
        rendered,
        (
            PANEL_X + 20,
            y
        )
    )

    y += 24

    for state, action, reward, before, after in update_history:

        text = (
            f"A{action} | R{reward:+} | "
            f"{before:+.2f} -> {after:+.2f}"
        )

        rendered = small_font.render(
            text,
            True,
            BLACK
        )

        screen.blit(
            rendered,
            (
                PANEL_X + 20,
                y
            )
        )

        y += 18


# ============================================================
# DRAW MESSAGE
# ============================================================

def draw_message():

    rendered = font.render(
        message,
        True,
        GREEN if not game_over else RED
    )

    screen.blit(
        rendered,
        (
            BOARD_X,
            BOARD_Y + BOARD_SIZE + 20
        )
    )

    instructions = small_font.render(
        "Click a cell to play X | R = restart | ESC = quit",
        True,
        GRAY
    )

    screen.blit(
        instructions,
        (
            BOARD_X,
            BOARD_Y + BOARD_SIZE + 50
        )
    )


# ============================================================
# MAIN LOOP
# ============================================================

running = True

while running:

    clock.tick(60)

    # --------------------------------------------------------
    # Automatic next episode
    # --------------------------------------------------------

    if game_over:

        elapsed = (
            pygame.time.get_ticks() -
            game_end_timer
        ) / 1000

        if elapsed >= AUTO_RESTART_DELAY:

            reset_game()

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:
                running = False

            if event.key == pygame.K_r:
                reset_game()

        if event.type == pygame.MOUSEBUTTONDOWN:

            if event.button == 1:

                mouse_x, mouse_y = event.pos

                if (
                    BOARD_X <= mouse_x <=
                    BOARD_X + BOARD_SIZE
                    and
                    BOARD_Y <= mouse_y <=
                    BOARD_Y + BOARD_SIZE
                ):

                    col = (
                        mouse_x - BOARD_X
                    ) // CELL_SIZE

                    row = (
                        mouse_y - BOARD_Y
                    ) // CELL_SIZE

                    position = (
                        row * 3 + col
                    )

                    player_move(position)

    # --------------------------------------------------------
    # Draw
    # --------------------------------------------------------

    screen.fill(WHITE)

    title = large_font.render(
        "Adaptive Tic-Tac-Toe",
        True,
        BLACK
    )

    screen.blit(
        title,
        (
            BOARD_X,
            25
        )
    )

    draw_board()
    draw_message()
    draw_panel()

    # Game over overlay
    if game_over:

        overlay = pygame.Surface(
            (
                BOARD_SIZE,
                BOARD_SIZE
            ),
            pygame.SRCALPHA
        )

        overlay.fill(
            (
                255,
                255,
                255,
                180
            )
        )

        screen.blit(
            overlay,
            (
                BOARD_X,
                BOARD_Y
            )
        )

        if winner == "X":
            result_text = "YOU WIN"

        elif winner == "O":
            result_text = "RL WINS"

        else:
            result_text = "DRAW"

        result_rendered = large_font.render(
            result_text,
            True,
            BLACK
        )

        screen.blit(
            result_rendered,
            (
                BOARD_X +
                BOARD_SIZE // 2 -
                result_rendered.get_width() // 2,
                BOARD_Y +
                BOARD_SIZE // 2 -
                result_rendered.get_height() // 2
            )
        )

        next_text = small_font.render(
            "Next episode starting...",
            True,
            BLACK
        )

        screen.blit(
            next_text,
            (
                BOARD_X +
                BOARD_SIZE // 2 -
                next_text.get_width() // 2,
                BOARD_Y +
                BOARD_SIZE // 2 + 35
            )
        )

    pygame.display.flip()


pygame.quit()

print("\n============================================")
print("Adaptive Tic-Tac-Toe RL Training Summary")
print("============================================")
print("Episodes completed:", episode - 1)
print("Player wins:", player_wins)
print("RL wins:", agent_wins)
print("Draws:", draws)
print("Final difficulty:", difficulty, difficulty_name)
print("Learned states:", len(Q))
print("Final epsilon:", round(epsilon, 4))

if recent_results:
    recent = recent_results[-SKILL_WINDOW:]
    print(
        "Recent player win rate:",
        round(
            recent.count("X") / len(recent) * 100,
            2
        ),
        "%"
    )

print("\nQ-table was preserved across all episodes.")

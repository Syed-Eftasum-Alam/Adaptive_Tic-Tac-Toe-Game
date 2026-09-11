import pygame
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import os
from collections import defaultdict, deque

# ============================================================
# Adaptive 4x4 Tic-Tac-Toe using Q-Learning
# Rule: 3 consecutive marks are enough to win
# Session: exactly 25 episodes, then a detailed analytics dashboard
# Player: X
# RL Agent: O
# ============================================================

pygame.init()

# ---------------------- Window / Layout ----------------------
WIDTH, HEIGHT = 1400, 900
BOARD_SIZE = 640
GRID_SIZE = 4
CELL_SIZE = BOARD_SIZE // GRID_SIZE

BOARD_X = 45
BOARD_Y = 125

PANEL_X = 735
PANEL_Y = 80
PANEL_W = 620
PANEL_H = 780

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Adaptive 4x4 Tic-Tac-Toe - Q Learning Analytics")
clock = pygame.time.Clock()

# ---------------------- Fonts ----------------------
FONT_TITLE = pygame.font.SysFont("arial", 38, bold=True)
FONT_BIG = pygame.font.SysFont("arial", 31, bold=True)
FONT_MED = pygame.font.SysFont("arial", 23, bold=True)
FONT = pygame.font.SysFont("arial", 19)
FONT_SMALL = pygame.font.SysFont("consolas", 16)
FONT_TINY = pygame.font.SysFont("consolas", 14)

# ---------------------- Colors ----------------------
BG = (24, 27, 34)
PANEL_BG = (34, 38, 47)
CARD_BG = (42, 47, 58)
GRID_COLOR = (210, 214, 222)
X_COLOR = (80, 180, 255)
O_COLOR = (255, 110, 135)
TEXT = (242, 244, 248)
MUTED = (166, 174, 188)
GREEN = (88, 220, 145)
YELLOW = (245, 206, 88)
RED = (245, 100, 100)
BLUE = (90, 170, 255)
PURPLE = (188, 130, 255)
CYAN = (80, 220, 220)
BAR_BG = (70, 76, 90)
BORDER = (72, 80, 96)
OVERLAY = (10, 12, 16)

# ---------------------- RL Settings ----------------------
PLAYER = "X"
AGENT = "O"
EMPTY = "."

ALPHA = 0.30
GAMMA = 0.90
EPSILON_START = 1.00
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.95

TOTAL_EPISODES = 25

Q = defaultdict(lambda: np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32))

# ---------------------- User Login / Persistence ----------------------
# The file is saved beside this Python script. It is plain-text JSON,
# so usernames, performance statistics and learned Q-values are readable.
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users_data.txt")

users_data = {}
current_user = None
current_user_key = None
app_mode = "login"
evaluation_images_saved = False
login_input = ""
login_message = "Enter your username to login, or create a new account."
login_message_color = MUTED

# Persistent values for the currently logged-in user.
recent_performance_results = []
cumulative_games = 0
cumulative_player_wins = 0
cumulative_agent_wins = 0
cumulative_draws = 0
cumulative_q_updates = 0


def load_users_data():
    global users_data
    if not os.path.exists(DATA_FILE):
        users_data = {}
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            users_data = json.loads(raw) if raw else {}
            if not isinstance(users_data, dict):
                users_data = {}
    except (OSError, json.JSONDecodeError):
        users_data = {}


def write_users_data():
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(users_data, f, indent=2)
    except OSError as exc:
        print("Could not save user data:", exc)


def default_profile(display_name):
    return {
        "username": display_name,
        "player_skill": 50,
        "difficulty_level": 1,
        "difficulty_name": "Very Easy",
        "epsilon": EPSILON_START,
        "recent_results": [],
        "cumulative_games": 0,
        "cumulative_player_wins": 0,
        "cumulative_agent_wins": 0,
        "cumulative_draws": 0,
        "cumulative_q_updates": 0,
        "q_table": {},
        "last_q_value": 0.0
    }


def serialize_q_table():
    saved = {}
    for state, values in Q.items():
        # Store only states that actually contain learned/non-zero information.
        if np.any(values != 0):
            saved[state] = [float(v) for v in values]
    return saved


def restore_q_table(saved_q):
    global Q
    Q = defaultdict(lambda: np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32))

    if not isinstance(saved_q, dict):
        return

    for state, values in saved_q.items():
        try:
            arr = np.array(values, dtype=np.float32)
            if len(arr) == GRID_SIZE * GRID_SIZE:
                Q[state] = arr
        except (TypeError, ValueError):
            pass


def save_current_user():
    if current_user_key is None:
        return

    profile = users_data.setdefault(current_user_key, default_profile(current_user or current_user_key))
    profile.update({
        "username": current_user,
        "player_skill": int(player_skill),
        "difficulty_level": int(difficulty_level),
        "difficulty_name": difficulty_name,
        "epsilon": float(epsilon),
        "recent_results": list(recent_performance_results[-10:]),
        "cumulative_games": int(cumulative_games),
        "cumulative_player_wins": int(cumulative_player_wins),
        "cumulative_agent_wins": int(cumulative_agent_wins),
        "cumulative_draws": int(cumulative_draws),
        "cumulative_q_updates": int(cumulative_q_updates),
        "q_table": serialize_q_table(),
        "last_q_value": float(last_action_info.get("q_after", 0.0))
    })
    write_users_data()


def login_existing_user(username):
    global current_user, current_user_key, app_mode
    global player_skill, difficulty_level, difficulty_name, epsilon
    global recent_performance_results
    global cumulative_games, cumulative_player_wins, cumulative_agent_wins
    global cumulative_draws, cumulative_q_updates

    key = username.strip().lower()
    if not key or key not in users_data:
        return False

    profile = users_data[key]
    current_user_key = key
    current_user = profile.get("username", username.strip())

    player_skill = int(profile.get("player_skill", 50))
    difficulty_level = int(profile.get("difficulty_level", 1))
    difficulty_name = profile.get("difficulty_name", "Very Easy")
    epsilon = float(profile.get("epsilon", EPSILON_START))

    recent_performance_results = list(profile.get("recent_results", []))[-10:]
    cumulative_games = int(profile.get("cumulative_games", 0))
    cumulative_player_wins = int(profile.get("cumulative_player_wins", 0))
    cumulative_agent_wins = int(profile.get("cumulative_agent_wins", 0))
    cumulative_draws = int(profile.get("cumulative_draws", 0))
    cumulative_q_updates = int(profile.get("cumulative_q_updates", 0))

    restore_q_table(profile.get("q_table", {}))
    start_new_session(preserve_progress=True)
    app_mode = "game"
    return True


def create_new_user(username):
    global current_user, current_user_key, app_mode
    global player_skill, difficulty_level, difficulty_name, epsilon
    global recent_performance_results
    global cumulative_games, cumulative_player_wins, cumulative_agent_wins
    global cumulative_draws, cumulative_q_updates, Q

    clean = username.strip()
    key = clean.lower()
    if not clean or key in users_data:
        return False

    users_data[key] = default_profile(clean)
    write_users_data()

    current_user = clean
    current_user_key = key
    player_skill = 50
    difficulty_level = 1
    difficulty_name = "Very Easy"
    epsilon = EPSILON_START
    recent_performance_results = []
    cumulative_games = 0
    cumulative_player_wins = 0
    cumulative_agent_wins = 0
    cumulative_draws = 0
    cumulative_q_updates = 0
    Q = defaultdict(lambda: np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32))

    start_new_session(preserve_progress=True)
    save_current_user()
    app_mode = "game"
    return True


def logout_user():
    global current_user, current_user_key, app_mode
    global login_input, login_message, login_message_color

    save_current_user()
    current_user = None
    current_user_key = None
    login_input = ""
    login_message = "Enter your username to login, or create a new account."
    login_message_color = MUTED
    app_mode = "login"


load_users_data()

# ---------------------- Winning Lines ----------------------
WINNING_LINES = []

# Horizontal
for r in range(4):
    for c in range(2):
        WINNING_LINES.append([r * 4 + c, r * 4 + c + 1, r * 4 + c + 2])

# Vertical
for c in range(4):
    for r in range(2):
        WINNING_LINES.append([r * 4 + c, (r + 1) * 4 + c, (r + 2) * 4 + c])

# Diagonal down-right
for r in range(2):
    for c in range(2):
        WINNING_LINES.append([
            r * 4 + c,
            (r + 1) * 4 + c + 1,
            (r + 2) * 4 + c + 2
        ])

# Diagonal down-left
for r in range(2):
    for c in range(2, 4):
        WINNING_LINES.append([
            r * 4 + c,
            (r + 1) * 4 + c - 1,
            (r + 2) * 4 + c - 2
        ])

# ---------------------- Game State ----------------------
board = [EMPTY] * 16

epsilon = EPSILON_START
session_results = []
player_wins = 0
agent_wins = 0
draws = 0

player_skill = 50
difficulty_name = "Very Easy"
difficulty_level = 1

session_finished = False
game_over = False
next_game_time = 0
winning_line = None

last_agent_state = None
last_agent_action = None

last_action_info = {
    "state": "",
    "action": "-",
    "reward": 0,
    "q_before": 0.0,
    "q_after": 0.0,
    "delta": 0.0
}

recent_updates = deque(maxlen=10)

# ---------------------- Analytics History ----------------------
episode_history = []
current_episode_q_changes = []
total_q_updates = 0
previous_episode_skill = 50

# ---------------------- Utility Functions ----------------------

def board_to_state(b):
    return "".join(b)


def available_actions(b):
    return [i for i, v in enumerate(b) if v == EMPTY]


def check_winner(b, mark):
    for line in WINNING_LINES:
        if all(b[i] == mark for i in line):
            return True, line
    return False, None


def is_draw(b):
    player_win, _ = check_winner(b, PLAYER)
    agent_win, _ = check_winner(b, AGENT)
    return EMPTY not in b and not player_win and not agent_win


def reset_board():
    global board, game_over, last_agent_state, last_agent_action
    global winning_line, current_episode_q_changes

    board = [EMPTY] * 16
    game_over = False
    last_agent_state = None
    last_agent_action = None
    winning_line = None
    current_episode_q_changes = []


def estimate_player_skill():
    global player_skill

    # Skill is based on the persistent recent history, not only this session.
    if not recent_performance_results:
        player_skill = 50
        return

    score = 50
    for result in recent_performance_results[-10:]:
        if result == "P":
            score += 10
        elif result == "D":
            score += 3
        elif result == "A":
            score -= 8

    player_skill = max(0, min(100, score))


def update_difficulty():
    global difficulty_name, difficulty_level

    # Brand-new users get a short beginner period. Returning users keep
    # the level saved in their profile and continue from that point.
    if cumulative_games < 3:
        difficulty_name = "Very Easy"
        difficulty_level = 1
        return

    estimate_player_skill()

    if player_skill >= 80:
        difficulty_name = "Expert"
        difficulty_level = 5
    elif player_skill >= 65:
        difficulty_name = "Hard"
        difficulty_level = 4
    elif player_skill >= 45:
        difficulty_name = "Normal"
        difficulty_level = 3
    elif player_skill >= 25:
        difficulty_name = "Easy"
        difficulty_level = 2
    else:
        difficulty_name = "Very Easy"
        difficulty_level = 1

def player_win_rate():
    if not session_results:
        return 0.0
    return 100.0 * player_wins / len(session_results)


def effective_epsilon():
    factors = {
        1: 1.35,
        2: 1.10,
        3: 0.80,
        4: 0.50,
        5: 0.25
    }
    return max(EPSILON_MIN, min(1.0, epsilon * factors[difficulty_level]))


def tactical_move(b):
    actions = available_actions(b)
    if not actions:
        return None

    # 1. Immediate winning move.
    for a in actions:
        temp = b.copy()
        temp[a] = AGENT
        win, _ = check_winner(temp, AGENT)
        if win:
            return a

    # 2. Block player's immediate winning move.
    for a in actions:
        temp = b.copy()
        temp[a] = PLAYER
        win, _ = check_winner(temp, PLAYER)
        if win:
            return a

    # 3. Tactical scoring.
    scores = {}
    for a in actions:
        score = 0.0
        temp = b.copy()
        temp[a] = AGENT

        for line in WINNING_LINES:
            if a not in line:
                continue

            values = [temp[i] for i in line]
            agent_count = values.count(AGENT)
            player_count = values.count(PLAYER)
            empty_count = values.count(EMPTY)

            if player_count == 0:
                if agent_count == 2 and empty_count == 1:
                    score += 10
                elif agent_count == 1 and empty_count == 2:
                    score += 3

        if a in [5, 6, 9, 10]:
            score += 2.5

        if a in [1, 2, 4, 7, 8, 11, 13, 14]:
            score += 1.0

        scores[a] = score

    max_score = max(scores.values())
    best = [a for a, s in scores.items() if s == max_score]
    return random.choice(best)


def choose_agent_action():
    state = board_to_state(board)
    actions = available_actions(board)

    if not actions:
        return None

    eps = effective_epsilon()

    tactical_probability = {
        1: 0.05,
        2: 0.15,
        3: 0.35,
        4: 0.70,
        5: 0.92
    }[difficulty_level]

    if random.random() < tactical_probability:
        move = tactical_move(board)
        if move is not None:
            return move

    # Epsilon-greedy policy.
    if random.random() < eps:
        return random.choice(actions)

    q_values = Q[state]
    legal_q = [(a, q_values[a]) for a in actions]
    max_q = max(v for _, v in legal_q)
    best_actions = [a for a, v in legal_q if v == max_q]
    return random.choice(best_actions)


def q_update(state, action, reward, next_state, terminal):
    global total_q_updates, cumulative_q_updates

    q_before = float(Q[state][action])

    if terminal:
        target = reward
    else:
        legal_next = [i for i, ch in enumerate(next_state) if ch == EMPTY]

        if legal_next:
            next_max = max(float(Q[next_state][a]) for a in legal_next)
        else:
            next_max = 0.0

        target = reward + GAMMA * next_max

    Q[state][action] += ALPHA * (target - Q[state][action])
    q_after = float(Q[state][action])
    delta = q_after - q_before

    current_episode_q_changes.append(abs(delta))
    total_q_updates += 1
    cumulative_q_updates += 1

    recent_updates.appendleft(
        f"a={action + 1:02d} r={reward:+.1f} "
        f"Q:{q_before:+.2f}->{q_after:+.2f} dQ={delta:+.2f}"
    )

    return q_before, q_after


def make_agent_move():
    global last_agent_state, last_agent_action, last_action_info

    if game_over or session_finished:
        return

    action = choose_agent_action()
    if action is None:
        return

    state = board_to_state(board)
    q_before = float(Q[state][action])

    board[action] = AGENT

    last_agent_state = state
    last_agent_action = action

    agent_win, _ = check_winner(board, AGENT)

    if agent_win:
        q_before, q_after = q_update(
            state, action, 10.0, board_to_state(board), True
        )
        last_action_info = {
            "state": state,
            "action": action + 1,
            "reward": 10,
            "q_before": q_before,
            "q_after": q_after,
            "delta": q_after - q_before
        }
        finish_game("A")
        return

    if is_draw(board):
        q_before, q_after = q_update(
            state, action, 3.0, board_to_state(board), True
        )
        last_action_info = {
            "state": state,
            "action": action + 1,
            "reward": 3,
            "q_before": q_before,
            "q_after": q_after,
            "delta": q_after - q_before
        }
        finish_game("D")
        return

    last_action_info = {
        "state": state,
        "action": action + 1,
        "reward": 0,
        "q_before": q_before,
        "q_after": q_before,
        "delta": 0.0
    }


def player_move(index):
    global last_action_info

    if game_over or session_finished:
        return

    if index < 0 or index >= 16 or board[index] != EMPTY:
        return

    board[index] = PLAYER

    player_win, _ = check_winner(board, PLAYER)

    if player_win:
        if last_agent_state is not None and last_agent_action is not None:
            q_before, q_after = q_update(
                last_agent_state,
                last_agent_action,
                -10.0,
                board_to_state(board),
                True
            )

            last_action_info = {
                "state": last_agent_state,
                "action": last_agent_action + 1,
                "reward": -10,
                "q_before": q_before,
                "q_after": q_after,
                "delta": q_after - q_before
            }

        finish_game("P")
        return

    if is_draw(board):
        if last_agent_state is not None and last_agent_action is not None:
            q_before, q_after = q_update(
                last_agent_state,
                last_agent_action,
                3.0,
                board_to_state(board),
                True
            )

            last_action_info = {
                "state": last_agent_state,
                "action": last_agent_action + 1,
                "reward": 3,
                "q_before": q_before,
                "q_after": q_after,
                "delta": q_after - q_before
            }

        finish_game("D")
        return

    # Normal non-terminal Q update after player's response.
    if last_agent_state is not None and last_agent_action is not None:
        q_before, q_after = q_update(
            last_agent_state,
            last_agent_action,
            0.0,
            board_to_state(board),
            False
        )

        last_action_info = {
            "state": last_agent_state,
            "action": last_agent_action + 1,
            "reward": 0,
            "q_before": q_before,
            "q_after": q_after,
            "delta": q_after - q_before
        }

    make_agent_move()


def finish_game(result):
    global game_over, player_wins, agent_wins, draws, winning_line
    global epsilon, session_finished, next_game_time
    global previous_episode_skill
    global cumulative_games, cumulative_player_wins
    global cumulative_agent_wins, cumulative_draws

    game_over = True
    session_results.append(result)
    recent_performance_results.append(result)
    del recent_performance_results[:-10]

    cumulative_games += 1

    if result == "P":
        _, winning_line = check_winner(board, PLAYER)
        player_wins += 1
        cumulative_player_wins += 1
    elif result == "A":
        _, winning_line = check_winner(board, AGENT)
        agent_wins += 1
        cumulative_agent_wins += 1
    else:
        winning_line = None
        draws += 1
        cumulative_draws += 1

    old_skill = player_skill

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    estimate_player_skill()
    update_difficulty()

    avg_q_change = (
        float(np.mean(current_episode_q_changes))
        if current_episode_q_changes else 0.0
    )
    max_q_change = (
        float(np.max(current_episode_q_changes))
        if current_episode_q_changes else 0.0
    )

    episode_history.append({
        "episode": len(session_results),
        "result": result,
        "skill": player_skill,
        "skill_change": player_skill - old_skill,
        "difficulty": difficulty_level,
        "difficulty_name": difficulty_name,
        "epsilon": effective_epsilon(),
        "avg_q_change": avg_q_change,
        "max_q_change": max_q_change,
        "q_updates": len(current_episode_q_changes),
        "learned_states": len(Q),
        "player_win_rate": player_win_rate()
    })

    previous_episode_skill = player_skill

    # Autosave after every episode so progress survives closing the program.
    save_current_user()

    if len(session_results) >= TOTAL_EPISODES:
        session_finished = True
    else:
        next_game_time = pygame.time.get_ticks() + 1200

def start_new_session(preserve_progress=True):
    global session_results, player_wins, agent_wins, draws
    global epsilon, session_finished, player_skill
    global difficulty_name, difficulty_level
    global last_action_info, episode_history
    global total_q_updates, previous_episode_skill
    global evaluation_images_saved

    session_results = []
    player_wins = 0
    agent_wins = 0
    draws = 0
    session_finished = False
    evaluation_images_saved = False

    episode_history = []
    total_q_updates = 0
    recent_updates.clear()

    if not preserve_progress:
        epsilon = EPSILON_START
        player_skill = 50
        difficulty_name = "Very Easy"
        difficulty_level = 1

    previous_episode_skill = player_skill

    last_action_info = {
        "state": "",
        "action": "-",
        "reward": 0,
        "q_before": 0.0,
        "q_after": 0.0,
        "delta": 0.0
    }

    reset_board()



# ---------------------- Drawing Helpers ----------------------

def draw_text(text, x, y, font=FONT, color=TEXT):
    surf = font.render(str(text), True, color)
    screen.blit(surf, (x, y))


def draw_centered_text(text, center_x, y, font=FONT, color=TEXT):
    surf = font.render(str(text), True, color)
    screen.blit(surf, (center_x - surf.get_width() // 2, y))


def draw_card(rect, title=None):
    pygame.draw.rect(screen, CARD_BG, rect, border_radius=12)
    pygame.draw.rect(screen, BORDER, rect, 1, border_radius=12)

    if title:
        draw_text(title, rect.x + 14, rect.y + 10, FONT_MED)
        pygame.draw.line(
            screen,
            BORDER,
            (rect.x + 12, rect.y + 44),
            (rect.right - 12, rect.y + 44),
            1
        )


def draw_metric_card(rect, title, value, value_color=TEXT):
    pygame.draw.rect(screen, CARD_BG, rect, border_radius=11)
    pygame.draw.rect(screen, BORDER, rect, 1, border_radius=11)
    draw_text(title, rect.x + 12, rect.y + 9, FONT_TINY, MUTED)
    draw_text(value, rect.x + 12, rect.y + 29, FONT_MED, value_color)


def draw_board():
    pygame.draw.rect(
        screen,
        (30, 33, 40),
        (BOARD_X, BOARD_Y, BOARD_SIZE, BOARD_SIZE),
        border_radius=8
    )

    for i in range(1, GRID_SIZE):
        x = BOARD_X + i * CELL_SIZE
        y = BOARD_Y + i * CELL_SIZE

        pygame.draw.line(
            screen,
            GRID_COLOR,
            (x, BOARD_Y),
            (x, BOARD_Y + BOARD_SIZE),
            4
        )
        pygame.draw.line(
            screen,
            GRID_COLOR,
            (BOARD_X, y),
            (BOARD_X + BOARD_SIZE, y),
            4
        )

    for i, value in enumerate(board):
        r = i // GRID_SIZE
        c = i % GRID_SIZE

        cx = BOARD_X + c * CELL_SIZE + CELL_SIZE // 2
        cy = BOARD_Y + r * CELL_SIZE + CELL_SIZE // 2

        if value == PLAYER:
            pad = 42
            pygame.draw.line(
                screen, X_COLOR,
                (cx - pad, cy - pad),
                (cx + pad, cy + pad),
                10
            )
            pygame.draw.line(
                screen, X_COLOR,
                (cx + pad, cy - pad),
                (cx - pad, cy + pad),
                10
            )

        elif value == AGENT:
            pygame.draw.circle(screen, O_COLOR, (cx, cy), 47, 10)

        draw_text(
            str(i + 1),
            BOARD_X + c * CELL_SIZE + 9,
            BOARD_Y + r * CELL_SIZE + 7,
            FONT_TINY,
            MUTED
        )

    if winning_line:
        points = []
        for index in winning_line:
            row = index // GRID_SIZE
            col = index % GRID_SIZE
            points.append((
                BOARD_X + col * CELL_SIZE + CELL_SIZE // 2,
                BOARD_Y + row * CELL_SIZE + CELL_SIZE // 2
            ))

        line_color = GREEN if board[winning_line[0]] == PLAYER else O_COLOR
        pygame.draw.line(screen, line_color, points[0], points[-1], 8)


def draw_difficulty_section(rect):
    draw_card(rect, "Adaptive Difficulty")

    draw_text(
        f"{difficulty_name}  ({difficulty_level}/5)",
        rect.x + 16,
        rect.y + 58,
        FONT_MED
    )

    bar_x = rect.x + 16
    bar_y = rect.y + 94
    bar_w = rect.w - 32

    pygame.draw.rect(
        screen, BAR_BG,
        (bar_x, bar_y, bar_w, 20),
        border_radius=10
    )

    fill_w = int(bar_w * difficulty_level / 5)

    if difficulty_level <= 2:
        bar_color = GREEN
    elif difficulty_level == 3:
        bar_color = YELLOW
    else:
        bar_color = RED

    pygame.draw.rect(
        screen, bar_color,
        (bar_x, bar_y, fill_w, 20),
        border_radius=10
    )

    draw_text(
        f"Skill: {player_skill}/100",
        rect.x + 16,
        rect.y + 125,
        FONT_SMALL,
        TEXT
    )
    draw_text(
        f"Effective epsilon: {effective_epsilon():.3f}",
        rect.x + 205,
        rect.y + 125,
        FONT_SMALL,
        TEXT
    )


def draw_q_moveset(rect):
    draw_card(rect, "Best Current Q-values")

    state = board_to_state(board)
    q_values = Q[state]
    actions = available_actions(board)

    if not actions:
        draw_text("No legal moves.", rect.x + 14, rect.y + 58, FONT_SMALL, MUTED)
        return

    items = [(a, float(q_values[a])) for a in actions]
    items.sort(key=lambda x: x[1], reverse=True)

    y = rect.y + 58
    for a, qv in items[:6]:
        r = a // 4 + 1
        c = a % 4 + 1
        draw_text(
            f"Cell {a+1:02d}  r{r}c{c}   Q={qv:+.3f}",
            rect.x + 14,
            y,
            FONT_SMALL,
            TEXT
        )
        y += 22


def draw_recent_updates(rect):
    draw_card(rect, "Recent Q Updates")

    y = rect.y + 58
    if not recent_updates:
        draw_text("No Q updates yet.", rect.x + 14, y, FONT_SMALL, MUTED)
        return

    for item in list(recent_updates)[:5]:
        draw_text(item, rect.x + 14, y, FONT_TINY, MUTED)
        y += 20


def draw_panel():
    pygame.draw.rect(
        screen,
        PANEL_BG,
        (PANEL_X, PANEL_Y, PANEL_W, PANEL_H),
        border_radius=14
    )

    current_episode = min(len(session_results) + 1, TOTAL_EPISODES)
    if session_finished:
        current_episode = TOTAL_EPISODES

    draw_text(
        f"{current_user}  |  Episode {current_episode} / {TOTAL_EPISODES}",
        PANEL_X + 20,
        PANEL_Y + 15,
        FONT_BIG
    )

    # Compact metric cards.
    card_y = PANEL_Y + 62
    gap = 10
    card_w = 138
    card_h = 58

    draw_metric_card(
        pygame.Rect(PANEL_X + 20, card_y, card_w, card_h),
        "PLAYER WINS",
        str(player_wins),
        X_COLOR
    )
    draw_metric_card(
        pygame.Rect(PANEL_X + 20 + card_w + gap, card_y, card_w, card_h),
        "RL WINS",
        str(agent_wins),
        O_COLOR
    )
    draw_metric_card(
        pygame.Rect(PANEL_X + 20 + 2 * (card_w + gap), card_y, card_w, card_h),
        "DRAWS",
        str(draws),
        YELLOW
    )
    draw_metric_card(
        pygame.Rect(PANEL_X + 20 + 3 * (card_w + gap), card_y, 132, card_h),
        "LEARNED STATES",
        str(len(Q)),
        CYAN
    )

    diff_rect = pygame.Rect(PANEL_X + 20, PANEL_Y + 132, PANEL_W - 40, 165)
    draw_difficulty_section(diff_rect)

    # Last action section.
    action_rect = pygame.Rect(PANEL_X + 20, PANEL_Y + 307, PANEL_W - 40, 105)
    draw_card(action_rect, "Last RL Action")

    draw_text(
        f"Cell: {last_action_info['action']}    "
        f"Reward: {last_action_info['reward']:+}",
        action_rect.x + 14,
        action_rect.y + 55,
        FONT_SMALL
    )
    draw_text(
        f"Q: {last_action_info['q_before']:+.3f} -> "
        f"{last_action_info['q_after']:+.3f}    "
        f"dQ={last_action_info['delta']:+.3f}",
        action_rect.x + 14,
        action_rect.y + 78,
        FONT_SMALL,
        MUTED
    )

    q_rect = pygame.Rect(PANEL_X + 20, PANEL_Y + 422, 280, 205)
    updates_rect = pygame.Rect(PANEL_X + 310, PANEL_Y + 422, 290, 205)

    draw_q_moveset(q_rect)
    draw_recent_updates(updates_rect)

    footer_rect = pygame.Rect(PANEL_X + 20, PANEL_Y + 637, PANEL_W - 40, 122)
    draw_card(footer_rect, "Session Progress")

    draw_text(
        f"Player win rate: {player_win_rate():.1f}%",
        footer_rect.x + 14,
        footer_rect.y + 57,
        FONT_SMALL
    )
    draw_text(
        f"Q updates (session): {total_q_updates}",
        footer_rect.x + 220,
        footer_rect.y + 57,
        FONT_SMALL
    )

    results = "".join(session_results[-18:]) or "-"
    draw_text(
        f"Recent outcomes: {results}   |   All-time games: {cumulative_games}",
        footer_rect.x + 14,
        footer_rect.y + 83,
        FONT_SMALL,
        MUTED
    )


def draw_top_text():
    draw_text("Adaptive 4x4 Tic-Tac-Toe", BOARD_X, 30, FONT_TITLE)
    draw_text(
        "Three consecutive marks win. You are X; the RL agent is O.",
        BOARD_X,
        78,
        FONT
    )

    draw_text(
        "Click an empty cell to play.",
        BOARD_X,
        785,
        FONT_MED
    )
    draw_text(
        "The agent automatically adapts its difficulty to your recent performance.",
        BOARD_X,
        820,
        FONT_SMALL,
        MUTED
    )
    draw_text(
        "ESC = Quit",
        BOARD_X,
        847,
        FONT_SMALL,
        MUTED
    )


def draw_episode_result_overlay():
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((*OVERLAY, 110))
    screen.blit(overlay, (0, 0))

    card = pygame.Rect(355, 285, 690, 275)
    pygame.draw.rect(screen, PANEL_BG, card, border_radius=20)

    result = session_results[-1]

    if result == "P":
        title = "YOU WIN!"
        subtitle = "Your success increases the estimated expertise."
        color = GREEN
    elif result == "A":
        title = "RL AGENT WINS"
        subtitle = "The model learns from the transition and adapts."
        color = O_COLOR
    else:
        title = "DRAW"
        subtitle = "A balanced episode also contributes to skill estimation."
        color = YELLOW

    pygame.draw.rect(screen, color, card, 3, border_radius=20)

    draw_centered_text(title, WIDTH // 2, 320, FONT_TITLE, color)
    draw_centered_text(subtitle, WIDTH // 2, 378, FONT)
    draw_centered_text(
        f"Episode {len(session_results)} of {TOTAL_EPISODES} complete",
        WIDTH // 2,
        425,
        FONT_MED,
        MUTED
    )

    if episode_history:
        h = episode_history[-1]
        change = h["skill_change"]
        sign = "+" if change >= 0 else ""
        draw_centered_text(
            f"Expertise: {h['skill']}/100 ({sign}{change})  |  "
            f"Difficulty: {h['difficulty_name']}  |  "
            f"Avg |dQ|: {h['avg_q_change']:.3f}",
            WIDTH // 2,
            470,
            FONT_SMALL,
            TEXT
        )

    draw_centered_text(
        "Next episode starting...",
        WIDTH // 2,
        515,
        FONT_SMALL,
        MUTED
    )


# ---------------------- Login Screen ----------------------

def draw_login_screen():
    screen.fill(BG)

    draw_centered_text("Adaptive Tic-Tac-Toe", WIDTH // 2, 120, FONT_TITLE, TEXT)
    draw_centered_text(
        "Q-learning profiles with persistent difficulty and performance",
        WIDTH // 2, 170, FONT, MUTED
    )

    card = pygame.Rect(WIDTH // 2 - 330, 250, 660, 390)
    pygame.draw.rect(screen, PANEL_BG, card, border_radius=18)
    pygame.draw.rect(screen, BORDER, card, 2, border_radius=18)

    draw_text("USERNAME", card.x + 55, card.y + 55, FONT_SMALL, MUTED)

    input_rect = pygame.Rect(card.x + 55, card.y + 90, card.w - 110, 62)
    pygame.draw.rect(screen, CARD_BG, input_rect, border_radius=10)
    pygame.draw.rect(screen, BLUE, input_rect, 2, border_radius=10)

    display_input = login_input if login_input else "Type username..."
    display_color = TEXT if login_input else MUTED
    draw_text(display_input, input_rect.x + 18, input_rect.y + 18, FONT_MED, display_color)

    login_btn = pygame.Rect(card.x + 55, card.y + 180, 260, 62)
    create_btn = pygame.Rect(card.x + 345, card.y + 180, 260, 62)

    pygame.draw.rect(screen, BLUE, login_btn, border_radius=10)
    pygame.draw.rect(screen, GREEN, create_btn, border_radius=10)

    draw_centered_text("LOGIN", login_btn.centerx, login_btn.y + 18, FONT_MED, TEXT)
    draw_centered_text("CREATE ACCOUNT", create_btn.centerx, create_btn.y + 18, FONT_MED, BG)

    draw_centered_text(
        login_message,
        WIDTH // 2,
        card.y + 275,
        FONT_SMALL,
        login_message_color
    )

    draw_centered_text(
        "No password is required. Your learning progress is saved automatically.",
        WIDTH // 2, card.y + 320, FONT_SMALL, MUTED
    )
    draw_centered_text(
        "Enter = Login    |    Ctrl+Enter = Create Account",
        WIDTH // 2, card.y + 350, FONT_TINY, MUTED
    )

    return input_rect, login_btn, create_btn


def process_login_attempt(create=False):
    global login_message, login_message_color, login_input

    username = login_input.strip()
    if not username:
        login_message = "Please enter a username."
        login_message_color = RED
        return

    key = username.lower()

    if create:
        if key in users_data:
            login_message = "That username already exists. Click LOGIN instead."
            login_message_color = YELLOW
            return

        if create_new_user(username):
            login_message = "Account created."
            login_message_color = GREEN
    else:
        if key not in users_data:
            login_message = "New user detected. Click CREATE ACCOUNT to register this username."
            login_message_color = YELLOW
            return

        if login_existing_user(username):
            login_message = "Login successful."
            login_message_color = GREEN


# ---------------------- Summary Charts ----------------------

def draw_line_chart(rect, title, values, min_value=None, max_value=None,
                    line_color=BLUE, value_suffix=""):
    draw_card(rect, title)

    if not values:
        draw_text("No data.", rect.x + 16, rect.y + 62, FONT_SMALL, MUTED)
        return

    plot_left = rect.x + 48
    plot_right = rect.right - 18
    plot_top = rect.y + 58
    plot_bottom = rect.bottom - 34

    if min_value is None:
        min_value = min(values)
    if max_value is None:
        max_value = max(values)

    if max_value == min_value:
        max_value += 1

    # Axes/grid.
    for i in range(5):
        y = int(plot_top + i * (plot_bottom - plot_top) / 4)
        pygame.draw.line(
            screen,
            BORDER,
            (plot_left, y),
            (plot_right, y),
            1
        )

    pygame.draw.line(
        screen, MUTED,
        (plot_left, plot_top),
        (plot_left, plot_bottom),
        1
    )
    pygame.draw.line(
        screen, MUTED,
        (plot_left, plot_bottom),
        (plot_right, plot_bottom),
        1
    )

    def map_y(v):
        ratio = (v - min_value) / (max_value - min_value)
        return int(plot_bottom - ratio * (plot_bottom - plot_top))

    points = []
    n = len(values)

    for i, v in enumerate(values):
        if n == 1:
            x = (plot_left + plot_right) // 2
        else:
            x = int(plot_left + i * (plot_right - plot_left) / (n - 1))
        y = map_y(v)
        points.append((x, y))

    if len(points) > 1:
        pygame.draw.lines(screen, line_color, False, points, 3)

    for x, y in points:
        pygame.draw.circle(screen, line_color, (x, y), 3)

    # Axis labels.
    draw_text(
        f"{max_value:.1f}{value_suffix}",
        rect.x + 8,
        plot_top - 5,
        FONT_TINY,
        MUTED
    )
    draw_text(
        f"{min_value:.1f}{value_suffix}",
        rect.x + 8,
        plot_bottom - 8,
        FONT_TINY,
        MUTED
    )
    draw_text("1", plot_left - 3, plot_bottom + 8, FONT_TINY, MUTED)
    draw_text(
        str(len(values)),
        plot_right - 12,
        plot_bottom + 8,
        FONT_TINY,
        MUTED
    )



def draw_dynamic_difficulty_chart(rect, history):
    """
    Draw difficulty directly from the main game's episode_history.

    Markers:
        Green triangle  = difficulty increased
        Red triangle    = difficulty decreased
        Gray circle     = difficulty unchanged

    This uses the real difficulty value stored after every episode.
    """
    draw_card(rect, "Dynamic Difficulty Change")

    if not history:
        draw_text(
            "No difficulty history yet.",
            rect.x + 16,
            rect.y + 62,
            FONT_SMALL,
            MUTED
        )
        return

    difficulties = [int(h.get("difficulty", 1)) for h in history]

    # Calculate episode-to-episode difficulty change.
    changes = [0]
    for i in range(1, len(difficulties)):
        changes.append(difficulties[i] - difficulties[i - 1])

    plot_left = rect.x + 76
    plot_right = rect.right - 18
    plot_top = rect.y + 55
    plot_bottom = rect.bottom - 43

    # Horizontal lines for the five discrete difficulty levels.
    level_names = {
        1: "Very Easy",
        2: "Easy",
        3: "Normal",
        4: "Hard",
        5: "Expert"
    }

    def map_y(level):
        ratio = (level - 1) / 4
        return int(plot_bottom - ratio * (plot_bottom - plot_top))

    for level in range(1, 6):
        y = map_y(level)
        pygame.draw.line(
            screen,
            BORDER,
            (plot_left, y),
            (plot_right, y),
            1
        )
        draw_text(
            f"{level}",
            rect.x + 16,
            y - 7,
            FONT_TINY,
            MUTED
        )
        draw_text(
            level_names[level],
            rect.x + 31,
            y - 7,
            FONT_TINY,
            MUTED
        )

    # Axes.
    pygame.draw.line(
        screen, MUTED,
        (plot_left, plot_top),
        (plot_left, plot_bottom),
        1
    )
    pygame.draw.line(
        screen, MUTED,
        (plot_left, plot_bottom),
        (plot_right, plot_bottom),
        1
    )

    n = len(difficulties)
    points = []

    for i, level in enumerate(difficulties):
        if n == 1:
            x = (plot_left + plot_right) // 2
        else:
            x = int(
                plot_left
                + i * (plot_right - plot_left) / (n - 1)
            )

        y = map_y(level)
        points.append((x, y))

    # Draw a step-like connection to emphasize discrete difficulty levels.
    if len(points) > 1:
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            middle_x = (x1 + x2) // 2

            pygame.draw.line(
                screen,
                PURPLE,
                (x1, y1),
                (middle_x, y1),
                3
            )
            pygame.draw.line(
                screen,
                PURPLE,
                (middle_x, y1),
                (middle_x, y2),
                3
            )
            pygame.draw.line(
                screen,
                PURPLE,
                (middle_x, y2),
                (x2, y2),
                3
            )

    # Mark each episode according to difficulty movement.
    for i, ((x, y), change) in enumerate(zip(points, changes)):
        if i == 0:
            pygame.draw.circle(screen, PURPLE, (x, y), 5)
            continue

        if change > 0:
            marker_color = GREEN

            # Up triangle.
            pygame.draw.polygon(
                screen,
                marker_color,
                [
                    (x, y - 7),
                    (x - 6, y + 5),
                    (x + 6, y + 5)
                ]
            )

            draw_text(
                f"+{change}",
                x - 7,
                y - 24,
                FONT_TINY,
                GREEN
            )

        elif change < 0:
            marker_color = RED

            # Down triangle.
            pygame.draw.polygon(
                screen,
                marker_color,
                [
                    (x, y + 7),
                    (x - 6, y - 5),
                    (x + 6, y - 5)
                ]
            )

            draw_text(
                str(change),
                x - 7,
                y + 10,
                FONT_TINY,
                RED
            )

        else:
            pygame.draw.circle(
                screen,
                MUTED,
                (x, y),
                4
            )

    # Episode labels.
    draw_text(
        "1",
        plot_left - 3,
        plot_bottom + 9,
        FONT_TINY,
        MUTED
    )
    draw_text(
        str(n),
        plot_right - 12,
        plot_bottom + 9,
        FONT_TINY,
        MUTED
    )

    # Compact legend.
    legend_y = rect.bottom - 21

    pygame.draw.polygon(
        screen,
        GREEN,
        [
            (rect.x + 105, legend_y - 5),
            (rect.x + 99, legend_y + 5),
            (rect.x + 111, legend_y + 5)
        ]
    )
    draw_text(
        "Increased",
        rect.x + 117,
        legend_y - 7,
        FONT_TINY,
        MUTED
    )

    pygame.draw.polygon(
        screen,
        RED,
        [
            (rect.x + 205, legend_y + 5),
            (rect.x + 199, legend_y - 5),
            (rect.x + 211, legend_y - 5)
        ]
    )
    draw_text(
        "Decreased",
        rect.x + 217,
        legend_y - 7,
        FONT_TINY,
        MUTED
    )

    pygame.draw.circle(
        screen,
        MUTED,
        (rect.x + 318, legend_y),
        4
    )
    draw_text(
        "Unchanged",
        rect.x + 330,
        legend_y - 7,
        FONT_TINY,
        MUTED
    )

def draw_outcome_strip(rect):
    draw_card(rect, "Episode Outcomes")

    if not session_results:
        return

    x0 = rect.x + 18
    y0 = rect.y + 60
    usable_w = rect.w - 36
    cell_gap = 3
    cell_w = max(12, int((usable_w - cell_gap * 24) / 25))

    for i, result in enumerate(session_results):
        x = x0 + i * (cell_w + cell_gap)

        if result == "P":
            color = X_COLOR
        elif result == "A":
            color = O_COLOR
        else:
            color = YELLOW

        pygame.draw.rect(
            screen,
            color,
            (x, y0, cell_w, 36),
            border_radius=5
        )
        draw_centered_text(
            str(i + 1),
            x + cell_w // 2,
            y0 + 43,
            FONT_TINY,
            MUTED
        )

    draw_text(
        "Blue=P win   Pink=Agent win   Yellow=Draw",
        rect.x + 18,
        rect.bottom - 23,
        FONT_TINY,
        MUTED
    )



EVALUATION_OUTPUT_DIR = "evaluation_output"


def sanitize_export_name(value):
    cleaned = "".join(
        ch if ch.isalnum() or ch in "-_" else "_"
        for ch in str(value)
    )
    return cleaned or "user"


def next_export_session_number(username):
    """
    Returns the next unused export number for this user's PNG files.
    Existing screenshots are never overwritten.
    """
    os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)

    safe_name = sanitize_export_name(username)
    number = 1

    while True:
        summary_file = os.path.join(
            EVALUATION_OUTPUT_DIR,
            f"{safe_name}_session_{number}_summary.png"
        )
        difficulty_file = os.path.join(
            EVALUATION_OUTPUT_DIR,
            f"{safe_name}_session_{number}_difficulty.png"
        )

        if (
            not os.path.exists(summary_file)
            and not os.path.exists(difficulty_file)
        ):
            return number

        number += 1


def save_evaluation_images():
    """
    Saves:
      1. Full Pygame summary dashboard as PNG.
      2. Dynamic difficulty graph generated with matplotlib/pyplot.

    The difficulty PNG is NOT a screenshot/crop. It is a real pyplot graph
    generated directly from episode_history.
    """
    if not current_user:
        return None, None

    os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)

    safe_name = sanitize_export_name(current_user)
    session_no = next_export_session_number(current_user)

    summary_path = os.path.join(
        EVALUATION_OUTPUT_DIR,
        f"{safe_name}_session_{session_no}_summary.png"
    )
    difficulty_path = os.path.join(
        EVALUATION_OUTPUT_DIR,
        f"{safe_name}_session_{session_no}_difficulty.png"
    )

    # Keep saving the complete in-game summary dashboard.
    pygame.image.save(screen, summary_path)

    # ---------------------------------------------------------
    # MATPLOTLIB / PYPLOT DYNAMIC DIFFICULTY GRAPH
    # ---------------------------------------------------------
    if episode_history:
        episodes = [
            int(h.get("episode", i + 1))
            for i, h in enumerate(episode_history)
        ]

        difficulties = [
            int(h.get("difficulty", 1))
            for h in episode_history
        ]

        skills = [
            float(h.get("skill", 0))
            for h in episode_history
        ]

        # Difficulty change from the previous episode.
        changes = [0]
        for i in range(1, len(difficulties)):
            changes.append(difficulties[i] - difficulties[i - 1])

        fig, ax = plt.subplots(figsize=(12, 6.5))

        # Discrete difficulty should look like steps rather than a smooth curve.
        ax.step(
            episodes,
            difficulties,
            where="mid",
            linewidth=2.5,
            label="Difficulty Level"
        )

        # Plot all episode points.
        ax.scatter(
            episodes,
            difficulties,
            s=45,
            zorder=3,
            label="Episode Difficulty"
        )

        # Mark increases/decreases separately.
        increase_eps = [
            ep for ep, change in zip(episodes, changes)
            if change > 0
        ]
        increase_levels = [
            level for level, change in zip(difficulties, changes)
            if change > 0
        ]

        decrease_eps = [
            ep for ep, change in zip(episodes, changes)
            if change < 0
        ]
        decrease_levels = [
            level for level, change in zip(difficulties, changes)
            if change < 0
        ]

        if increase_eps:
            ax.scatter(
                increase_eps,
                increase_levels,
                marker="^",
                s=130,
                zorder=4,
                label="Difficulty Increased"
            )

        if decrease_eps:
            ax.scatter(
                decrease_eps,
                decrease_levels,
                marker="v",
                s=130,
                zorder=4,
                label="Difficulty Decreased"
            )

        # Label actual difficulty transitions.
        for ep, level, change in zip(episodes, difficulties, changes):
            if change > 0:
                ax.annotate(
                    f"+{change}",
                    (ep, level),
                    xytext=(0, 13),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9
                )
            elif change < 0:
                ax.annotate(
                    str(change),
                    (ep, level),
                    xytext=(0, -18),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9
                )

        difficulty_labels = [
            "Very Easy",
            "Easy",
            "Normal",
            "Hard",
            "Expert"
        ]

        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(
            [
                "1 - Very Easy",
                "2 - Easy",
                "3 - Normal",
                "4 - Hard",
                "5 - Expert"
            ]
        )

        ax.set_ylim(0.5, 5.5)
        ax.set_xlim(
            max(0.5, min(episodes) - 0.5),
            max(episodes) + 0.5
        )

        # For 25 episodes, every episode can be shown clearly.
        ax.set_xticks(episodes)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Difficulty Level")
        ax.set_title(
            f"Dynamic Difficulty Adaptation - {current_user} "
            f"(Session {session_no})"
        )

        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        # Add a compact expertise note beneath each transition where useful.
        # This helps connect the player's estimated skill to difficulty changes.
        for ep, level, skill, change in zip(
            episodes, difficulties, skills, changes
        ):
            if change != 0:
                ax.annotate(
                    f"Skill={skill:.0f}",
                    (ep, level),
                    xytext=(8, 0),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.75
                )

        fig.tight_layout()
        fig.savefig(
            difficulty_path,
            dpi=200,
            bbox_inches="tight"
        )
        plt.close(fig)

    else:
        # Still create a pyplot image if there is unexpectedly no history.
        fig, ax = plt.subplots(figsize=(12, 6.5))
        ax.text(
            0.5,
            0.5,
            "No episode difficulty history available.",
            ha="center",
            va="center",
            transform=ax.transAxes
        )
        ax.set_title(
            f"Dynamic Difficulty Adaptation - {current_user}"
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Difficulty Level")
        fig.tight_layout()
        fig.savefig(
            difficulty_path,
            dpi=200,
            bbox_inches="tight"
        )
        plt.close(fig)

    print("\nEvaluation images automatically saved:")
    print(f"  Full summary       : {summary_path}")
    print(f"  Pyplot difficulty  : {difficulty_path}")

    return summary_path, difficulty_path


def draw_summary_overlay():
    # Full-screen dashboard.
    screen.fill(BG)

    draw_text(f"25-Episode Learning Summary - {current_user}", 45, 22, FONT_TITLE)
    draw_text(
        f"Persistent profile | All-time games: {cumulative_games} | All-time Q updates: {cumulative_q_updates}",
        47,
        67,
        FONT,
        MUTED
    )

    total = max(1, len(session_results))
    p_rate = 100 * player_wins / total
    a_rate = 100 * agent_wins / total
    d_rate = 100 * draws / total

    # Top metric cards.
    mx = 45
    my = 105
    mw = 205
    mh = 72
    mg = 12

    draw_metric_card(
        pygame.Rect(mx, my, mw, mh),
        "PLAYER RECORD",
        f"{player_wins} wins ({p_rate:.1f}%)",
        X_COLOR
    )
    draw_metric_card(
        pygame.Rect(mx + (mw + mg), my, mw, mh),
        "RL AGENT RECORD",
        f"{agent_wins} wins ({a_rate:.1f}%)",
        O_COLOR
    )
    draw_metric_card(
        pygame.Rect(mx + 2 * (mw + mg), my, mw, mh),
        "DRAWS",
        f"{draws} ({d_rate:.1f}%)",
        YELLOW
    )
    draw_metric_card(
        pygame.Rect(mx + 3 * (mw + mg), my, mw, mh),
        "FINAL EXPERTISE",
        f"{player_skill}/100",
        GREEN
    )
    draw_metric_card(
        pygame.Rect(mx + 4 * (mw + mg), my, mw, mh),
        "FINAL DIFFICULTY",
        f"{difficulty_name} ({difficulty_level}/5)",
        PURPLE
    )
    draw_metric_card(
        pygame.Rect(mx + 5 * (mw + mg), my, 205, mh),
        "Q-TABLE",
        f"{len(Q)} states",
        CYAN
    )

    # Prepare histories.
    skills = [h["skill"] for h in episode_history]
    q_changes = [h["avg_q_change"] for h in episode_history]
    difficulties = [h["difficulty"] for h in episode_history]
    epsilons = [h["epsilon"] for h in episode_history]

    # Charts.
    chart_w = 425
    chart_h = 245
    gap = 18
    chart_y = 198

    draw_line_chart(
        pygame.Rect(45, chart_y, chart_w, chart_h),
        "Player Expertise by Episode",
        skills,
        min_value=0,
        max_value=100,
        line_color=GREEN,
        value_suffix=""
    )

    q_max = max(q_changes) if q_changes else 1.0
    q_axis_max = max(1.0, q_max * 1.15)

    draw_line_chart(
        pygame.Rect(45 + chart_w + gap, chart_y, chart_w, chart_h),
        "Average |Q-value Update|",
        q_changes,
        min_value=0,
        max_value=q_axis_max,
        line_color=CYAN
    )

    draw_dynamic_difficulty_chart(
        pygame.Rect(45 + 2 * (chart_w + gap), chart_y, chart_w, chart_h),
        episode_history
    )

    # Second row: epsilon + detailed episode table + outcomes.
    second_y = 462

    draw_line_chart(
        pygame.Rect(45, second_y, chart_w, 215),
        "Exploration Rate (Epsilon)",
        epsilons,
        min_value=0,
        max_value=1,
        line_color=YELLOW
    )

    table_rect = pygame.Rect(45 + chart_w + gap, second_y, 868, 215)
    draw_card(table_rect, "Episode Learning Details")

    headers = ["Ep", "Result", "Skill", "dSkill", "Diff", "Avg|dQ|", "Q upd", "States"]
    x_positions = [
        table_rect.x + 14,
        table_rect.x + 60,
        table_rect.x + 125,
        table_rect.x + 200,
        table_rect.x + 275,
        table_rect.x + 350,
        table_rect.x + 465,
        table_rect.x + 545
    ]

    for htxt, x in zip(headers, x_positions):
        draw_text(htxt, x, table_rect.y + 52, FONT_TINY, MUTED)

    # Show last 7 episodes, which fits without overlap.
    y = table_rect.y + 76
    for h in episode_history[-7:]:
        result_name = {"P": "P", "A": "A", "D": "D"}[h["result"]]
        skill_delta = h["skill_change"]
        values = [
            f"{h['episode']:02d}",
            result_name,
            str(h["skill"]),
            f"{skill_delta:+d}",
            str(h["difficulty"]),
            f"{h['avg_q_change']:.3f}",
            str(h["q_updates"]),
            str(h["learned_states"])
        ]
        for value, x in zip(values, x_positions):
            draw_text(value, x, y, FONT_TINY, TEXT)
        y += 18

    draw_text(
        f"Total Q updates: {total_q_updates}   |   "
        f"Final effective epsilon: {effective_epsilon():.3f}",
        table_rect.x + 620,
        table_rect.y + 52,
        FONT_TINY,
        MUTED
    )

    outcome_rect = pygame.Rect(45, 698, 1313, 125)
    draw_outcome_strip(outcome_rect)

    draw_centered_text(
        "R = New 25-episode session   |   L = Logout   |   Q-table and difficulty are saved",
        WIDTH // 2,
        845,
        FONT_SMALL,
        GREEN
    )
    draw_text("ESC = Quit", 45, 850, FONT_SMALL, MUTED)


def print_terminal_summary():
    if not session_results:
        return

    total = len(session_results)

    print("\n" + "=" * 72)
    print("25 EPISODE SESSION SUMMARY")
    print("=" * 72)
    print(f"Player wins       : {player_wins}")
    print(f"RL agent wins     : {agent_wins}")
    print(f"Draws             : {draws}")
    print(f"Player win rate   : {100 * player_wins / total:.2f}%")
    print(f"RL win rate       : {100 * agent_wins / total:.2f}%")
    print(f"Draw rate         : {100 * draws / total:.2f}%")
    print(f"Final expertise   : {player_skill}/100")
    print(f"Final difficulty  : {difficulty_name} ({difficulty_level}/5)")
    print(f"Learned states    : {len(Q)}")
    print(f"Total Q updates   : {total_q_updates}")
    print(f"Final epsilon     : {effective_epsilon():.4f}")
    print("Sequence          :", " ".join(session_results))

    print("\nPER-EPISODE ANALYTICS")
    print("-" * 72)
    print("Ep  R  Skill dSkill Diff  Avg|dQ|  Max|dQ|  Qupd  States")
    for h in episode_history:
        print(
            f"{h['episode']:02d}  {h['result']}  "
            f"{h['skill']:>3}   {h['skill_change']:+3d}    "
            f"{h['difficulty']}    "
            f"{h['avg_q_change']:.4f}   "
            f"{h['max_q_change']:.4f}   "
            f"{h['q_updates']:>3}   "
            f"{h['learned_states']:>4}"
        )

    print("=" * 72)


# ---------------------- Main Loop ----------------------
running = True
reset_board()

while running:
    clock.tick(60)

    # Login UI rectangles are recreated each frame for click handling.
    login_input_rect = None
    login_button_rect = None
    create_button_rect = None

    if app_mode == "login":
        login_input_rect, login_button_rect, create_button_rect = draw_login_screen()
        pygame.display.flip()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            save_current_user()
            running = False

        elif app_mode == "login":
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_BACKSPACE:
                    login_input = login_input[:-1]
                elif event.key == pygame.K_RETURN:
                    ctrl_pressed = bool(event.mod & pygame.KMOD_CTRL)
                    process_login_attempt(create=ctrl_pressed)
                else:
                    if event.unicode and event.unicode.isprintable() and len(login_input) < 24:
                        # Keep usernames simple and safe for a text-based profile file.
                        if event.unicode.isalnum() or event.unicode in "_-.":
                            login_input += event.unicode

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if login_button_rect and login_button_rect.collidepoint(event.pos):
                    process_login_attempt(create=False)
                elif create_button_rect and create_button_rect.collidepoint(event.pos):
                    process_login_attempt(create=True)

        elif app_mode == "game":
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    save_current_user()
                    running = False

                elif event.key == pygame.K_l:
                    logout_user()

                elif event.key == pygame.K_r and session_finished:
                    # Start another 25 episodes without resetting learned progress.
                    start_new_session(preserve_progress=True)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and not session_finished and not game_over:
                    mx, my = event.pos

                    if (
                        BOARD_X <= mx < BOARD_X + BOARD_SIZE
                        and BOARD_Y <= my < BOARD_Y + BOARD_SIZE
                    ):
                        col = (mx - BOARD_X) // CELL_SIZE
                        row = (my - BOARD_Y) // CELL_SIZE
                        index = row * GRID_SIZE + col
                        player_move(index)

    if app_mode == "game":
        if game_over and not session_finished:
            if pygame.time.get_ticks() >= next_game_time:
                reset_board()

        screen.fill(BG)

        if session_finished:
            draw_summary_overlay()

            # Save the dashboard and difficulty graph once per completed session.
            if not evaluation_images_saved:
                pygame.display.flip()
                save_evaluation_images()
                evaluation_images_saved = True
        else:
            draw_top_text()
            draw_board()
            draw_panel()

            # User can logout at any time with L.
            draw_text(f"Logged in: {current_user}   |   L = Logout", 990, 35, FONT_SMALL, MUTED)

            if game_over and session_results:
                draw_episode_result_overlay()

        pygame.display.flip()

save_current_user()
print_terminal_summary()
pygame.quit()

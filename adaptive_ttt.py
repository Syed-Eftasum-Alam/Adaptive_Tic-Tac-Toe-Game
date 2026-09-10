import pygame
import numpy as np
import random
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

    if not session_results:
        player_skill = 50
        return

    score = 50

    # Adapt using the most recent 10 episodes.
    for result in session_results[-10:]:
        if result == "P":
            score += 10
        elif result == "D":
            score += 3
        elif result == "A":
            score -= 8

    player_skill = max(0, min(100, score))


def update_difficulty():
    global difficulty_name, difficulty_level

    # First 3 completed episodes remain Very Easy.
    if len(session_results) < 3:
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
    global total_q_updates

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

    game_over = True
    session_results.append(result)

    if result == "P":
        _, winning_line = check_winner(board, PLAYER)
        player_wins += 1
    elif result == "A":
        _, winning_line = check_winner(board, AGENT)
        agent_wins += 1
    else:
        winning_line = None
        draws += 1

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

    if len(session_results) >= TOTAL_EPISODES:
        session_finished = True
    else:
        next_game_time = pygame.time.get_ticks() + 1200


def start_new_session():
    global session_results, player_wins, agent_wins, draws
    global epsilon, session_finished, player_skill
    global difficulty_name, difficulty_level
    global last_action_info, episode_history
    global total_q_updates, previous_episode_skill

    session_results = []
    player_wins = 0
    agent_wins = 0
    draws = 0

    epsilon = EPSILON_START
    player_skill = 50
    previous_episode_skill = 50

    difficulty_name = "Very Easy"
    difficulty_level = 1
    session_finished = False

    episode_history = []
    total_q_updates = 0
    recent_updates.clear()

    last_action_info = {
        "state": "",
        "action": "-",
        "reward": 0,
        "q_before": 0.0,
        "q_after": 0.0,
        "delta": 0.0
    }

    # Q-table is intentionally preserved so the agent keeps learning
    # between 25-episode sessions.
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
        f"Episode {current_episode} / {TOTAL_EPISODES}",
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
        f"Total Q updates: {total_q_updates}",
        footer_rect.x + 220,
        footer_rect.y + 57,
        FONT_SMALL
    )

    results = "".join(session_results[-18:]) or "-"
    draw_text(
        f"Recent outcomes: {results}",
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


def draw_summary_overlay():
    # Full-screen dashboard.
    screen.fill(BG)

    draw_text("25-Episode Learning Summary", 45, 22, FONT_TITLE)
    draw_text(
        "Adaptive difficulty, expertise progression and Q-learning behavior",
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

    draw_line_chart(
        pygame.Rect(45 + 2 * (chart_w + gap), chart_y, chart_w, chart_h),
        "Difficulty Level by Episode",
        difficulties,
        min_value=1,
        max_value=5,
        line_color=PURPLE
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
        "Press R to start a new 25-episode session (Q-table learning is preserved)",
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

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_r and session_finished:
                start_new_session()

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

    if game_over and not session_finished:
        if pygame.time.get_ticks() >= next_game_time:
            reset_board()

    screen.fill(BG)

    if session_finished:
        draw_summary_overlay()
    else:
        draw_top_text()
        draw_board()
        draw_panel()

        if game_over and session_results:
            draw_episode_result_overlay()

    pygame.display.flip()

print_terminal_summary()
pygame.quit()

# ============================================================
# ablation_engine.py
#
# Automated experimental engine for:
# Adaptive Game Difficulty Using Reinforcement Learning
#
# Based on the logic used in adaptive_ttt.py
#
# Board:
#   4 x 4
# Winning condition:
#   3 consecutive marks
#
# Agent:
#   O
#
# Opponent:
#   X
# ============================================================

import random
import numpy as np

from collections import defaultdict

from metrics_logger import ExperimentLogger


# ============================================================
# Environment configuration
# ============================================================

GRID_SIZE = 4
BOARD_CELLS = GRID_SIZE * GRID_SIZE

PLAYER = "X"
AGENT = "O"
EMPTY = "."

# ============================================================
# RL hyperparameters
# ============================================================

ALPHA = 0.30
GAMMA = 0.90

EPSILON_START = 1.00
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.95

# ============================================================
# Rewards
# ============================================================

REWARD_AGENT_WIN = 10.0
REWARD_PLAYER_WIN = -10.0
REWARD_DRAW = 3.0
REWARD_STEP = 0.0


# ============================================================
# Winning lines
# ============================================================

WINNING_LINES = []

# Horizontal
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE - 2):
        WINNING_LINES.append([
            r * GRID_SIZE + c,
            r * GRID_SIZE + c + 1,
            r * GRID_SIZE + c + 2
        ])

# Vertical
for c in range(GRID_SIZE):
    for r in range(GRID_SIZE - 2):
        WINNING_LINES.append([
            r * GRID_SIZE + c,
            (r + 1) * GRID_SIZE + c,
            (r + 2) * GRID_SIZE + c
        ])

# Diagonal down-right
for r in range(GRID_SIZE - 2):
    for c in range(GRID_SIZE - 2):
        WINNING_LINES.append([
            r * GRID_SIZE + c,
            (r + 1) * GRID_SIZE + c + 1,
            (r + 2) * GRID_SIZE + c + 2
        ])

# Diagonal down-left
for r in range(GRID_SIZE - 2):
    for c in range(2, GRID_SIZE):
        WINNING_LINES.append([
            r * GRID_SIZE + c,
            (r + 1) * GRID_SIZE + c - 1,
            (r + 2) * GRID_SIZE + c - 2
        ])


# ============================================================
# Utility functions
# ============================================================

def board_to_state(board):
    return "".join(board)


def available_actions(board):
    return [
        i for i, value in enumerate(board)
        if value == EMPTY
    ]


def check_winner(board, mark):

    for line in WINNING_LINES:

        if all(board[i] == mark for i in line):
            return True, line

    return False, None


def is_draw(board):

    player_win, _ = check_winner(board, PLAYER)
    agent_win, _ = check_winner(board, AGENT)

    return (
        EMPTY not in board
        and not player_win
        and not agent_win
    )


# ============================================================
# Tactical policy
# ============================================================

def tactical_move(board):

    actions = available_actions(board)

    if not actions:
        return None

    # --------------------------------------------------------
    # 1. Immediate winning move
    # --------------------------------------------------------

    for action in actions:

        temp = board.copy()
        temp[action] = AGENT

        win, _ = check_winner(temp, AGENT)

        if win:
            return action

    # --------------------------------------------------------
    # 2. Block player's immediate winning move
    # --------------------------------------------------------

    for action in actions:

        temp = board.copy()
        temp[action] = PLAYER

        win, _ = check_winner(temp, PLAYER)

        if win:
            return action

    # --------------------------------------------------------
    # 3. Tactical scoring
    # --------------------------------------------------------

    scores = {}

    for action in actions:

        score = 0.0

        temp = board.copy()
        temp[action] = AGENT

        for line in WINNING_LINES:

            if action not in line:
                continue

            values = [
                temp[index]
                for index in line
            ]

            agent_count = values.count(AGENT)
            player_count = values.count(PLAYER)
            empty_count = values.count(EMPTY)

            if player_count == 0:

                if agent_count == 2 and empty_count == 1:
                    score += 10.0

                elif agent_count == 1 and empty_count == 2:
                    score += 3.0

        # Central cells
        if action in [5, 6, 9, 10]:
            score += 2.5

        # Other strategically useful cells
        if action in [1, 2, 4, 7, 8, 11, 13, 14]:
            score += 1.0

        scores[action] = score

    max_score = max(scores.values())

    best_actions = [
        action
        for action, score in scores.items()
        if score == max_score
    ]

    return random.choice(best_actions)


# ============================================================
# Automated opponents
# ============================================================

def random_opponent_move(board):

    actions = available_actions(board)

    if not actions:
        return None

    return random.choice(actions)


def heuristic_opponent_move(board):

    actions = available_actions(board)

    if not actions:
        return None

    # --------------------------------------------------------
    # Opponent wins if possible
    # --------------------------------------------------------

    for action in actions:

        temp = board.copy()
        temp[action] = PLAYER

        win, _ = check_winner(temp, PLAYER)

        if win:
            return action

    # --------------------------------------------------------
    # Block agent
    # --------------------------------------------------------

    for action in actions:

        temp = board.copy()
        temp[action] = AGENT

        win, _ = check_winner(temp, AGENT)

        if win:
            return action

    # --------------------------------------------------------
    # Otherwise choose randomly
    # --------------------------------------------------------

    return random.choice(actions)


def strong_opponent_move(board):

    # Strong opponent uses the same tactical logic,
    # but from the opponent's perspective.

    actions = available_actions(board)

    if not actions:
        return None

    # Winning move
    for action in actions:

        temp = board.copy()
        temp[action] = PLAYER

        win, _ = check_winner(temp, PLAYER)

        if win:
            return action

    # Block agent
    for action in actions:

        temp = board.copy()
        temp[action] = AGENT

        win, _ = check_winner(temp, AGENT)

        if win:
            return action

    # Prefer center
    center_actions = [
        a for a in actions
        if a in [5, 6, 9, 10]
    ]

    if center_actions:
        return random.choice(center_actions)

    return random.choice(actions)


# ============================================================
# Q-learning Agent
# ============================================================

class RLAgent:

    def __init__(
        self,
        use_q_learning=True,
        use_tactical=True,
        use_adaptive=True,
        use_epsilon=True
    ):

        self.use_q_learning = use_q_learning
        self.use_tactical = use_tactical
        self.use_adaptive = use_adaptive
        self.use_epsilon = use_epsilon

        self.Q = defaultdict(
            lambda: np.zeros(
                BOARD_CELLS,
                dtype=np.float32
            )
        )

        self.epsilon = EPSILON_START

        self.player_skill = 50

        self.difficulty_level = 1
        self.difficulty_name = "Very Easy"

        self.total_q_updates = 0

        # Metrics for current episode
        self.current_q_changes = []

        self.tactical_moves = 0
        self.epsilon_moves = 0
        self.q_policy_moves = 0

    # --------------------------------------------------------
    # Reset episode statistics
    # --------------------------------------------------------

    def reset_episode_metrics(self):

        self.current_q_changes = []

        self.tactical_moves = 0
        self.epsilon_moves = 0
        self.q_policy_moves = 0

    # --------------------------------------------------------
    # Effective epsilon
    # --------------------------------------------------------

    def effective_epsilon(self):

        factors = {
            1: 1.35,
            2: 1.10,
            3: 0.80,
            4: 0.50,
            5: 0.25
        }

        factor = factors[self.difficulty_level]

        return max(
            EPSILON_MIN,
            min(
                1.0,
                self.epsilon * factor
            )
        )

    # --------------------------------------------------------
    # Player skill estimation
    # --------------------------------------------------------

    def estimate_player_skill(self, recent_results):

        if not recent_results:

            self.player_skill = 50

            return

        score = 50

        for result in recent_results[-10:]:

            if result == "P":
                score += 10

            elif result == "D":
                score += 3

            elif result == "A":
                score -= 8

        self.player_skill = max(
            0,
            min(100, score)
        )

    # --------------------------------------------------------
    # Adaptive difficulty
    # --------------------------------------------------------

    def update_difficulty(self, cumulative_games, recent_results):

        if not self.use_adaptive:

            self.difficulty_level = 3
            self.difficulty_name = "Normal"

            return

        # Same beginner-period idea as the original code
        if cumulative_games < 3:

            self.difficulty_level = 1
            self.difficulty_name = "Very Easy"

            return

        self.estimate_player_skill(recent_results)

        if self.player_skill >= 80:

            self.difficulty_level = 5
            self.difficulty_name = "Expert"

        elif self.player_skill >= 65:

            self.difficulty_level = 4
            self.difficulty_name = "Hard"

        elif self.player_skill >= 45:

            self.difficulty_level = 3
            self.difficulty_name = "Normal"

        elif self.player_skill >= 25:

            self.difficulty_level = 2
            self.difficulty_name = "Easy"

        else:

            self.difficulty_level = 1
            self.difficulty_name = "Very Easy"

    # --------------------------------------------------------
    # Choose action
    # --------------------------------------------------------

    def choose_action(self, board):

        actions = available_actions(board)

        if not actions:
            return None, "none"

        # ----------------------------------------------------
        # Tactical policy
        # ----------------------------------------------------

        if self.use_tactical:

            tactical_probability = {
                1: 0.05,
                2: 0.15,
                3: 0.35,
                4: 0.70,
                5: 0.92
            }[self.difficulty_level]

            if random.random() < tactical_probability:

                action = tactical_move(board)

                if action is not None:

                    self.tactical_moves += 1

                    return action, "tactical"

        # ----------------------------------------------------
        # Epsilon-greedy
        # ----------------------------------------------------

        if self.use_epsilon:

            eps = self.effective_epsilon()

            if random.random() < eps:

                self.epsilon_moves += 1

                return random.choice(actions), "epsilon"

        # ----------------------------------------------------
        # Q policy
        # ----------------------------------------------------

        if self.use_q_learning:

            state = board_to_state(board)

            q_values = self.Q[state]

            legal_q = [
                (action, float(q_values[action]))
                for action in actions
            ]

            max_q = max(
                value
                for _, value in legal_q
            )

            best_actions = [
                action
                for action, value in legal_q
                if value == max_q
            ]

            self.q_policy_moves += 1

            return random.choice(best_actions), "q"

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        return random.choice(actions), "random"

    # --------------------------------------------------------
    # Q update
    # --------------------------------------------------------

    def q_update(
        self,
        state,
        action,
        reward,
        next_state,
        terminal
    ):

        if not self.use_q_learning:

            return 0.0, 0.0, 0.0

        q_before = float(
            self.Q[state][action]
        )

        if terminal:

            target = reward

        else:

            legal_next = [
                i
                for i, value in enumerate(next_state)
                if value == EMPTY
            ]

            if legal_next:

                next_max = max(
                    float(self.Q[next_state][a])
                    for a in legal_next
                )

            else:

                next_max = 0.0

            target = (
                reward
                + GAMMA * next_max
            )

        self.Q[state][action] += (
            ALPHA
            * (
                target
                - self.Q[state][action]
            )
        )

        q_after = float(
            self.Q[state][action]
        )

        delta = q_after - q_before

        self.current_q_changes.append(
            abs(delta)
        )

        self.total_q_updates += 1

        return q_before, q_after, delta

    # --------------------------------------------------------
    # Decay epsilon
    # --------------------------------------------------------

    def decay_epsilon(self):

        if not self.use_epsilon:
            return

        self.epsilon = max(
            EPSILON_MIN,
            self.epsilon * EPSILON_DECAY
        )

    # --------------------------------------------------------
    # Learned states
    # --------------------------------------------------------

    def learned_states(self):

        return len(self.Q)


# ============================================================
# One complete game
# ============================================================

def play_game(
    agent,
    opponent_type="random"
):

    board = [EMPTY] * BOARD_CELLS

    last_agent_state = None
    last_agent_action = None

    move_count = 0

    opponent_move_function = {
        "random": random_opponent_move,
        "heuristic": heuristic_opponent_move,
        "strong": strong_opponent_move
    }[opponent_type]

    while True:

        # ====================================================
        # Opponent / Player move
        # ====================================================

        player_action = opponent_move_function(board)

        if player_action is None:
            return "D", move_count

        board[player_action] = PLAYER

        move_count += 1

        player_win, _ = check_winner(
            board,
            PLAYER
        )

        if player_win:

            if (
                last_agent_state is not None
                and last_agent_action is not None
            ):

                agent.q_update(
                    last_agent_state,
                    last_agent_action,
                    REWARD_PLAYER_WIN,
                    board_to_state(board),
                    True
                )

            return "P", move_count

        if is_draw(board):

            if (
                last_agent_state is not None
                and last_agent_action is not None
            ):

                agent.q_update(
                    last_agent_state,
                    last_agent_action,
                    REWARD_DRAW,
                    board_to_state(board),
                    True
                )

            return "D", move_count

        # ====================================================
        # Agent move
        # ====================================================

        state = board_to_state(board)

        action, action_type = agent.choose_action(board)

        if action is None:
            return "D", move_count

        # ----------------------------------------------------
        # Update previous agent action after opponent response
        # ----------------------------------------------------

        if (
            last_agent_state is not None
            and last_agent_action is not None
        ):

            agent.q_update(
                last_agent_state,
                last_agent_action,
                REWARD_STEP,
                state,
                False
            )

        # ----------------------------------------------------
        # Make agent move
        # ----------------------------------------------------

        board[action] = AGENT

        move_count += 1

        last_agent_state = state
        last_agent_action = action

        agent_win, _ = check_winner(
            board,
            AGENT
        )

        if agent_win:

            agent.q_update(
                state,
                action,
                REWARD_AGENT_WIN,
                board_to_state(board),
                True
            )

            return "A", move_count

        if is_draw(board):

            agent.q_update(
                state,
                action,
                REWARD_DRAW,
                board_to_state(board),
                True
            )

            return "D", move_count


# ============================================================
# Experiment runner
# ============================================================

def run_experiment(
    model_name,
    seeds,
    episodes_per_seed=100,
    opponent_type="random",
    use_q_learning=True,
    use_tactical=True,
    use_adaptive=True,
    use_epsilon=True,
    output_file="ablation_results.csv"
):

    print()
    print("=" * 80)
    print(f"EXPERIMENT: {model_name}")
    print("=" * 80)

    print(f"Seeds              : {seeds}")
    print(f"Episodes / seed    : {episodes_per_seed}")
    print(f"Opponent           : {opponent_type}")
    print(f"Q-learning         : {use_q_learning}")
    print(f"Tactical policy    : {use_tactical}")
    print(f"Adaptive difficulty: {use_adaptive}")
    print(f"Epsilon-greedy     : {use_epsilon}")
    print("=" * 80)

    logger = ExperimentLogger(output_file)

    for seed in seeds:

        print()
        print(
            f"Starting seed {seed} "
            f"for {model_name}"
        )

        random.seed(seed)
        np.random.seed(seed)

        agent = RLAgent(
            use_q_learning=use_q_learning,
            use_tactical=use_tactical,
            use_adaptive=use_adaptive,
            use_epsilon=use_epsilon
        )

        recent_results = []

        cumulative_agent_wins = 0
        cumulative_player_wins = 0
        cumulative_draws = 0

        for episode in range(
            1,
            episodes_per_seed + 1
        ):

            agent.reset_episode_metrics()

            old_skill = agent.player_skill

            result, move_count = play_game(
                agent,
                opponent_type
            )

            # ------------------------------------------------
            # Outcome counters
            # ------------------------------------------------

            if result == "A":

                cumulative_agent_wins += 1

            elif result == "P":

                cumulative_player_wins += 1

            else:

                cumulative_draws += 1

            recent_results.append(result)

            # ------------------------------------------------
            # Update skill/difficulty
            # ------------------------------------------------

            agent.decay_epsilon()

            agent.estimate_player_skill(
                recent_results
            )

            agent.update_difficulty(
                episode,
                recent_results
            )

            # ------------------------------------------------
            # Reward
            # ------------------------------------------------

            if result == "A":

                reward = REWARD_AGENT_WIN

            elif result == "P":

                reward = REWARD_PLAYER_WIN

            else:

                reward = REWARD_DRAW

            # ------------------------------------------------
            # Q metrics
            # ------------------------------------------------

            if agent.current_q_changes:

                avg_q_change = float(
                    np.mean(
                        agent.current_q_changes
                    )
                )

                max_q_change = float(
                    np.max(
                        agent.current_q_changes
                    )
                )

            else:

                avg_q_change = 0.0
                max_q_change = 0.0

            # ------------------------------------------------
            # Win rates
            # ------------------------------------------------

            total_games = episode

            agent_win_rate = (
                100.0
                * cumulative_agent_wins
                / total_games
            )

            player_win_rate = (
                100.0
                * cumulative_player_wins
                / total_games
            )

            draw_rate = (
                100.0
                * cumulative_draws
                / total_games
            )

            # ------------------------------------------------
            # Log all metrics
            # ------------------------------------------------

            logger.record_episode(
                model=model_name,
                seed=seed,
                episode=episode,
                result=result,
                reward=reward,
                move_count=move_count,

                agent_win_rate=agent_win_rate,
                player_win_rate=player_win_rate,
                draw_rate=draw_rate,

                skill=agent.player_skill,
                skill_change=(
                    agent.player_skill
                    - old_skill
                ),

                difficulty=agent.difficulty_level,
                difficulty_name=agent.difficulty_name,

                epsilon=agent.epsilon,
                effective_epsilon=(
                    agent.effective_epsilon()
                    if use_epsilon
                    else 0.0
                ),

                avg_q_change=avg_q_change,
                max_q_change=max_q_change,

                q_updates=agent.total_q_updates,
                episode_q_updates=len(
                    agent.current_q_changes
                ),

                learned_states=agent.learned_states(),

                tactical_moves=agent.tactical_moves,
                epsilon_moves=agent.epsilon_moves,
                q_policy_moves=agent.q_policy_moves,

                use_q_learning=use_q_learning,
                use_tactical=use_tactical,
                use_adaptive=use_adaptive,
                use_epsilon=use_epsilon
            )

            # ------------------------------------------------
            # Progress display
            # ------------------------------------------------

            if (
                episode == 1
                or episode % 25 == 0
                or episode == episodes_per_seed
            ):

                print(
                    f"Seed {seed} | "
                    f"Episode {episode:4d}/{episodes_per_seed} | "
                    f"Result={result} | "
                    f"WinRate={agent_win_rate:6.2f}% | "
                    f"Skill={agent.player_skill:3d} | "
                    f"Difficulty={agent.difficulty_level} | "
                    f"States={agent.learned_states()}"
                )

    print()
    print(
        f"Finished experiment: {model_name}"
    )
    print()
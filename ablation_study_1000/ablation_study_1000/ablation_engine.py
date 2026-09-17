from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

BOARD_SIZE = 4
WIN_LENGTH = 3
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE

# ============================================================
# Q-LEARNING HYPERPARAMETERS
# ============================================================

ALPHA = 0.30
GAMMA = 0.90

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.95

# ============================================================
# REWARDS
# ============================================================

REWARD_WIN = 10.0
REWARD_DRAW = 3.0
REWARD_LOSS = -10.0
REWARD_STEP = 0.0

# ============================================================
# DYNAMIC DIFFICULTY
# ============================================================

TACTICAL_PROB = {
    1: 0.05,   # Very Easy
    2: 0.15,   # Easy
    3: 0.35,   # Normal
    4: 0.70,   # Hard
    5: 0.92,   # Expert
}

EPSILON_FACTOR = {
    1: 1.35,
    2: 1.10,
    3: 0.80,
    4: 0.50,
    5: 0.25,
}

DIFFICULTY_NAMES = {
    1: "Very Easy",
    2: "Easy",
    3: "Normal",
    4: "Hard",
    5: "Expert",
}

# Same strategic cells used by the project.
CENTER_CELLS = [5, 6, 9, 10]

STRATEGIC_CELLS = [
    1, 2, 4, 7,
    8, 11, 13, 14
]

State = Tuple[int, ...]


# ============================================================
# BOARD / GAME FUNCTIONS
# ============================================================

def all_windows() -> List[Tuple[int, int, int]]:
    """
    Generate every possible 3-cell winning line
    on a 4x4 board.

    Includes:
        - horizontal
        - vertical
        - diagonal down-right
        - diagonal down-left
    """

    windows = []

    # ----------------------------
    # Horizontal
    # ----------------------------

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE - WIN_LENGTH + 1):

            window = tuple(
                row * BOARD_SIZE + col + k
                for k in range(WIN_LENGTH)
            )

            windows.append(window)

    # ----------------------------
    # Vertical
    # ----------------------------

    for col in range(BOARD_SIZE):
        for row in range(BOARD_SIZE - WIN_LENGTH + 1):

            window = tuple(
                (row + k) * BOARD_SIZE + col
                for k in range(WIN_LENGTH)
            )

            windows.append(window)

    # ----------------------------
    # Diagonal down-right
    # ----------------------------

    for row in range(BOARD_SIZE - WIN_LENGTH + 1):
        for col in range(BOARD_SIZE - WIN_LENGTH + 1):

            window = tuple(
                (row + k) * BOARD_SIZE + (col + k)
                for k in range(WIN_LENGTH)
            )

            windows.append(window)

    # ----------------------------
    # Diagonal down-left
    # ----------------------------

    for row in range(BOARD_SIZE - WIN_LENGTH + 1):
        for col in range(WIN_LENGTH - 1, BOARD_SIZE):

            window = tuple(
                (row + k) * BOARD_SIZE + (col - k)
                for k in range(WIN_LENGTH)
            )

            windows.append(window)

    return windows


WIN_WINDOWS = all_windows()


def legal_actions(board: State) -> List[int]:
    """
    Return all empty board positions.
    """

    return [
        index
        for index, value in enumerate(board)
        if value == 0
    ]


def apply_action(
    board: State,
    action: int,
    player: int
) -> State:

    new_board = list(board)

    new_board[action] = player

    return tuple(new_board)


def has_winner(
    board: State,
    player: int
) -> bool:

    return any(
        all(board[index] == player for index in window)
        for window in WIN_WINDOWS
    )


def is_draw(board: State) -> bool:

    return (
        all(value != 0 for value in board)
        and not has_winner(board, 1)
        and not has_winner(board, -1)
    )


def terminal_result(board: State) -> Optional[str]:

    if has_winner(board, -1):
        return "agent_win"

    if has_winner(board, 1):
        return "player_win"

    if is_draw(board):
        return "draw"

    return None


# ============================================================
# TACTICAL FUNCTIONS
# ============================================================

def immediate_winning_actions(
    board: State,
    player: int
) -> List[int]:

    actions = []

    for action in legal_actions(board):

        next_board = apply_action(
            board,
            action,
            player
        )

        if has_winner(next_board, player):
            actions.append(action)

    return actions


def tactical_scores(
    board: State
) -> Dict[int, float]:

    scores = {
        action: 0.0
        for action in legal_actions(board)
    }

    # --------------------------------------------------------
    # Agent tactical opportunities
    # --------------------------------------------------------

    for window in WIN_WINDOWS:

        values = [
            board[index]
            for index in window
        ]

        # Agent has two O's and one empty cell.
        if values.count(-1) == 2 and values.count(0) == 1:

            empty_cell = window[
                values.index(0)
            ]

            if empty_cell in scores:
                scores[empty_cell] += 10.0

        # Agent has one O and two empty cells.
        elif values.count(-1) == 1 and values.count(0) == 2:

            for index in window:

                if (
                    board[index] == 0
                    and index in scores
                ):
                    scores[index] += 3.0

    # --------------------------------------------------------
    # Center preference
    # --------------------------------------------------------

    for cell in CENTER_CELLS:

        if cell in scores:
            scores[cell] += 2.5

    # --------------------------------------------------------
    # Strategic positions
    # --------------------------------------------------------

    for cell in STRATEGIC_CELLS:

        if cell in scores:
            scores[cell] += 1.0

    return scores


# ============================================================
# AGENT CONFIGURATION
# ============================================================

@dataclass
class AgentConfig:

    use_q: bool = True
    use_tactical: bool = True
    use_adaptive: bool = True
    use_epsilon: bool = True


# ============================================================
# RL AGENT
# ============================================================

class RLAgent:

    def __init__(
        self,
        seed: int,
        config: AgentConfig
    ):

        self.rng = random.Random(seed)

        self.config = config

        # Q-table:
        # state -> 16 action values
        self.q_table: Dict[
            State,
            List[float]
        ] = defaultdict(
            lambda: [0.0] * NUM_ACTIONS
        )

        self.alpha = ALPHA
        self.gamma = GAMMA

        self.epsilon = EPSILON_START

        # Initial player skill.
        self.skill = 50.0

        # Initial difficulty.
        self.difficulty = 1

        self.games_played = 0

        self.recent_outcomes = deque(
            maxlen=10
        )

        self.total_q_updates = 0

        self.learned_states = set()

    # --------------------------------------------------------
    # Difficulty
    # --------------------------------------------------------

    def get_difficulty(self) -> int:

        # No Adaptive version:
        # difficulty remains Normal.
        if not self.config.use_adaptive:
            return 3

        # Original project behavior.
        if self.games_played < 3:
            return 1

        if self.skill >= 80:
            return 5

        if self.skill >= 65:
            return 4

        if self.skill >= 45:
            return 3

        if self.skill >= 25:
            return 2

        return 1

    # --------------------------------------------------------
    # Effective epsilon
    # --------------------------------------------------------

    def effective_epsilon(
        self,
        epsilon_override: Optional[float] = None
    ) -> float:

        base_epsilon = (
            self.epsilon
            if epsilon_override is None
            else epsilon_override
        )

        if not self.config.use_adaptive:
            factor = EPSILON_FACTOR[3]

        else:
            factor = EPSILON_FACTOR[
                self.difficulty
            ]

        effective = base_epsilon * factor

        return max(
            EPSILON_MIN,
            min(1.0, effective)
        )

    # --------------------------------------------------------
    # Tactical probability
    # --------------------------------------------------------

    def tactical_probability(self) -> float:

        if not self.config.use_tactical:
            return 0.0

        return TACTICAL_PROB[
            self.difficulty
        ]

    # --------------------------------------------------------
    # Skill update
    # --------------------------------------------------------

    def update_skill(
        self,
        result: str
    ) -> None:

        if not self.config.use_adaptive:

            self.difficulty = 3

            return

        # Player perspective:
        #
        # Player win  -> +10
        # Draw        -> +3
        # Agent win   -> -8

        skill_change = {
            "player_win": 10,
            "draw": 3,
            "agent_win": -8,
        }[result]

        self.recent_outcomes.append(
            result
        )

        self.skill += skill_change

        self.skill = max(
            0.0,
            min(100.0, self.skill)
        )

        self.games_played += 1

        self.difficulty = self.get_difficulty()

    # --------------------------------------------------------
    # Epsilon decay
    # --------------------------------------------------------

    def decay_epsilon(self) -> None:

        if not self.config.use_epsilon:
            return

        self.epsilon = max(
            EPSILON_MIN,
            self.epsilon * EPSILON_DECAY
        )

    # --------------------------------------------------------
    # Action selection
    # --------------------------------------------------------

    def choose_action(
        self,
        board: State,
        evaluation: bool = False
    ) -> Tuple[int, str, float]:

        legal = legal_actions(board)

        if not legal:
            raise ValueError(
                "No legal actions available."
            )

        # During evaluation:
        #
        # exploration is disabled.
        #
        # Q-table remains frozen.
        if evaluation:

            effective_epsilon = 0.0

        else:

            effective_epsilon = (
                self.effective_epsilon()
            )

        # ====================================================
        # Tactical policy
        # ====================================================

        if self.config.use_tactical:

            # 1. Immediate winning move.
            winning_moves = (
                immediate_winning_actions(
                    board,
                    -1
                )
            )

            if winning_moves:

                return (
                    self.rng.choice(
                        winning_moves
                    ),
                    "tactical",
                    effective_epsilon
                )

            # 2. Immediate block.
            blocking_moves = (
                immediate_winning_actions(
                    board,
                    1
                )
            )

            if blocking_moves:

                return (
                    self.rng.choice(
                        blocking_moves
                    ),
                    "tactical",
                    effective_epsilon
                )

            # 3. Tactical scoring.
            if (
                self.rng.random()
                < self.tactical_probability()
            ):

                scores = tactical_scores(
                    board
                )

                if scores:

                    best_score = max(
                        scores.values()
                    )

                    best_actions = [
                        action
                        for action, score
                        in scores.items()
                        if score == best_score
                    ]

                    if best_score > 0:

                        return (
                            self.rng.choice(
                                best_actions
                            ),
                            "tactical",
                            effective_epsilon
                        )

        # ====================================================
        # Epsilon exploration
        # ====================================================

        if (
            self.config.use_epsilon
            and not evaluation
            and self.rng.random()
            < effective_epsilon
        ):

            return (
                self.rng.choice(legal),
                "epsilon",
                effective_epsilon
            )

        # ====================================================
        # Q-learning exploitation
        # ====================================================

        if self.config.use_q:

            q_values = self.q_table[
                board
            ]

            best_value = max(
                q_values[action]
                for action in legal
            )

            best_actions = [
                action
                for action in legal
                if q_values[action]
                == best_value
            ]

            return (
                self.rng.choice(
                    best_actions
                ),
                "q_policy",
                effective_epsilon
            )

        # ====================================================
        # No-Q baseline fallback
        # ====================================================

        scores = tactical_scores(
            board
        )

        if scores:

            best_score = max(
                scores.values()
            )

            if best_score > 0:

                best_actions = [
                    action
                    for action, score
                    in scores.items()
                    if score == best_score
                ]

                return (
                    self.rng.choice(
                        best_actions
                    ),
                    "tactical_fallback",
                    effective_epsilon
                )

        return (
            self.rng.choice(legal),
            "random_fallback",
            effective_epsilon
        )

    # --------------------------------------------------------
    # Q-learning update
    # --------------------------------------------------------

    def update_q(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        terminal: bool,
        enabled: bool = True
    ) -> float:

        # Evaluation or No-Q:
        # do not modify Q-table.
        if not (
            self.config.use_q
            and enabled
        ):
            return 0.0

        old_value = self.q_table[
            state
        ][action]

        if terminal:

            target = reward

        else:

            next_legal = legal_actions(
                next_state
            )

            if next_legal:

                max_next_q = max(
                    self.q_table[
                        next_state
                    ][a]
                    for a in next_legal
                )

                target = (
                    reward
                    + self.gamma
                    * max_next_q
                )

            else:

                target = reward

        new_value = (
            old_value
            + self.alpha
            * (
                target
                - old_value
            )
        )

        delta = (
            new_value
            - old_value
        )

        self.q_table[
            state
        ][action] = new_value

        self.total_q_updates += 1

        self.learned_states.add(
            state
        )

        return delta


# ============================================================
# OPPONENT POLICIES
# ============================================================

def choose_opponent_action(
    board: State,
    opponent_name: str,
    rng: random.Random
) -> int:

    legal = legal_actions(board)

    # --------------------------------------------------------
    # Random opponent
    # --------------------------------------------------------

    if opponent_name == "random":

        return rng.choice(
            legal
        )

    # --------------------------------------------------------
    # Heuristic opponent
    # --------------------------------------------------------

    if opponent_name in (
        "heuristic",
        "strong"
    ):

        # Immediate win.
        winning_moves = (
            immediate_winning_actions(
                board,
                1
            )
        )

        if winning_moves:

            return rng.choice(
                winning_moves
            )

        # Immediate block.
        blocking_moves = (
            immediate_winning_actions(
                board,
                -1
            )
        )

        if blocking_moves:

            return rng.choice(
                blocking_moves
            )

        # ----------------------------------------------------
        # Strong Tactical Opponent
        # ----------------------------------------------------

        if opponent_name == "strong":

            center_moves = [
                action
                for action in CENTER_CELLS
                if action in legal
            ]

            if center_moves:

                return rng.choice(
                    center_moves
                )

    # Default random move.
    return rng.choice(
        legal
    )


# ============================================================
# PLAY ONE GAME
# ============================================================

def play_game(
    agent: RLAgent,
    opponent_name: str,
    episode: int,
    phase: str,
    seed: int,
    q_updates_enabled: bool,
    rng: random.Random
) -> Dict:

    # Empty 4x4 board.
    board: State = tuple(
        [0] * NUM_ACTIONS
    )

    move_count = 0

    tactical_moves = 0
    epsilon_moves = 0
    q_policy_moves = 0

    q_deltas = []

    # X = opponent/player
    # O = RL agent
    #
    # The RL agent is the second player.
    current_player = 1

    while True:

        result = terminal_result(
            board
        )

        if result:
            break

        # ====================================================
        # Player / opponent X
        # ====================================================

        if current_player == 1:

            action = choose_opponent_action(
                board,
                opponent_name,
                rng
            )

            board = apply_action(
                board,
                action,
                1
            )

            move_count += 1

            result = terminal_result(
                board
            )

            if result:
                break

            current_player = -1

        # ====================================================
        # RL agent O
        # ====================================================

        else:

            state = board

            action, policy, effective_eps = (
                agent.choose_action(
                    board,
                    evaluation=(
                        phase == "eval"
                    )
                )
            )

            board = apply_action(
                board,
                action,
                -1
            )

            move_count += 1

            # Record action-selection behavior.
            if policy in (
                "tactical",
                "tactical_fallback"
            ):

                tactical_moves += 1

            elif policy == "epsilon":

                epsilon_moves += 1

            elif policy == "q_policy":

                q_policy_moves += 1

            result = terminal_result(
                board
            )

            # ------------------------------------------------
            # Terminal transition
            # ------------------------------------------------

            if result:

                reward = {
                    "agent_win": REWARD_WIN,
                    "player_win": REWARD_LOSS,
                    "draw": REWARD_DRAW,
                }[result]

                delta = agent.update_q(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=board,
                    terminal=True,
                    enabled=q_updates_enabled
                )

                if delta != 0.0:
                    q_deltas.append(
                        abs(delta)
                    )

                break

            # ------------------------------------------------
            # Non-terminal transition
            # ------------------------------------------------

            delta = agent.update_q(
                state=state,
                action=action,
                reward=REWARD_STEP,
                next_state=board,
                terminal=False,
                enabled=q_updates_enabled
            )

            if delta != 0.0:
                q_deltas.append(
                    abs(delta)
                )

            current_player = 1

    # ========================================================
    # DDA UPDATE
    #
    # DDA remains active during evaluation.
    # Q-learning does NOT.
    # ========================================================

    if agent.config.use_adaptive:

        agent.update_skill(
            result
        )

    # Epsilon decays only during training.
    if phase == "train":

        agent.decay_epsilon()

    # During evaluation epsilon is explicitly treated
    # as disabled, even though the stored training epsilon
    # remains available for reporting.
    if phase == "eval":

        effective_epsilon_for_log = 0.0

    else:

        effective_epsilon_for_log = (
            agent.effective_epsilon()
        )

    reward = {
        "agent_win": REWARD_WIN,
        "player_win": REWARD_LOSS,
        "draw": REWARD_DRAW,
    }[result]

    return {

        "seed": seed,

        "episode": episode,

        "phase": phase,

        "opponent": opponent_name,

        "result": result,

        "reward": reward,

        "move_count": move_count,

        "skill": agent.skill,

        "difficulty": agent.difficulty,

        "difficulty_name":
            DIFFICULTY_NAMES[
                agent.difficulty
            ],

        "epsilon": agent.epsilon,

        "effective_epsilon":
            effective_epsilon_for_log,

        "q_updates_enabled":
            int(q_updates_enabled),

        "q_updates":
            agent.total_q_updates,

        "learned_states":
            len(agent.learned_states),

        "avg_abs_q_change":
            (
                sum(q_deltas)
                / len(q_deltas)
                if q_deltas
                else 0.0
            ),

        "max_abs_q_change":
            (
                max(q_deltas)
                if q_deltas
                else 0.0
            ),

        "tactical_moves":
            tactical_moves,

        "epsilon_moves":
            epsilon_moves,

        "q_policy_moves":
            q_policy_moves,
    }


# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

MODEL_CONFIGS = {

    "Full": AgentConfig(
        use_q=True,
        use_tactical=True,
        use_adaptive=True,
        use_epsilon=True
    ),

    "No Tactical": AgentConfig(
        use_q=True,
        use_tactical=False,
        use_adaptive=True,
        use_epsilon=True
    ),

    "No Adaptive": AgentConfig(
        use_q=True,
        use_tactical=True,
        use_adaptive=False,
        use_epsilon=True
    ),

    "No Q": AgentConfig(
        use_q=False,
        use_tactical=True,
        use_adaptive=True,
        use_epsilon=False
    ),
}


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(
    model_name: str,
    opponent_name: str,
    seeds: List[int],
    train_episodes: int,
    eval_episodes: int,
    output_file: str
) -> None:

    # Import here to avoid circular import problems.
    from metrics_logger import MetricsLogger

    logger = MetricsLogger(
        output_file
    )

    total_episodes = (
        train_episodes
        + eval_episodes
    )

    print()
    print("=" * 75)
    print(
        f"MODEL     : {model_name}"
    )
    print(
        f"OPPONENT  : {opponent_name}"
    )
    print(
        f"TRAINING  : {train_episodes}"
    )
    print(
        f"EVALUATION: {eval_episodes}"
    )
    print(
        f"SEEDS     : {seeds}"
    )
    print(
        f"TOTAL     : "
        f"{len(seeds) * total_episodes}"
        f" games"
    )
    print("=" * 75)

    for seed in seeds:

        config = MODEL_CONFIGS[
            model_name
        ]

        # Fresh model for every seed.
        agent = RLAgent(
            seed=seed,
            config=config
        )

        # Separate opponent RNG.
        rng = random.Random(
            seed + 100000
        )

        # ====================================================
        # TRAINING
        # ====================================================

        print()
        print(
            f"Seed {seed}: "
            f"starting training..."
        )

        for episode in range(
            1,
            train_episodes + 1
        ):

            row = play_game(
                agent=agent,
                opponent_name=opponent_name,
                episode=episode,
                phase="train",
                seed=seed,
                q_updates_enabled=config.use_q,
                rng=rng
            )

            row["model"] = model_name

            logger.log(
                row
            )

            if (
                episode % 100 == 0
                or episode == train_episodes
            ):

                print(
                    f"  Training "
                    f"{episode:4d}/"
                    f"{train_episodes} | "
                    f"skill="
                    f"{agent.skill:5.1f} | "
                    f"difficulty="
                    f"{agent.difficulty} | "
                    f"epsilon="
                    f"{agent.epsilon:.4f}"
                )

        # ====================================================
        # FREEZE Q-TABLE
        # ====================================================

        q_updates_before_eval = (
            agent.total_q_updates
        )

        learned_states_before_eval = (
            len(agent.learned_states)
        )

        print()
        print(
            f"Seed {seed}: "
            f"training finished."
        )

        print(
            "  Q-table frozen for evaluation."
        )

        print(
            f"  Q updates before evaluation: "
            f"{q_updates_before_eval}"
        )

        print(
            f"  Learned states before evaluation: "
            f"{learned_states_before_eval}"
        )

        # ====================================================
        # EVALUATION
        # ====================================================

        print(
            f"  Starting frozen evaluation..."
        )

        for eval_index in range(
            1,
            eval_episodes + 1
        ):

            episode = (
                train_episodes
                + eval_index
            )

            row = play_game(
                agent=agent,
                opponent_name=opponent_name,
                episode=episode,
                phase="eval",
                seed=seed,
                q_updates_enabled=False,
                rng=rng
            )

            row["model"] = model_name

            # Ensure the logged Q update count represents
            # the frozen Q-table.
            row["q_updates"] = (
                q_updates_before_eval
            )

            row["learned_states"] = (
                learned_states_before_eval
            )

            logger.log(
                row
            )

        print(
            f"  Evaluation complete."
        )

        print(
            f"  Q updates after evaluation: "
            f"{agent.total_q_updates}"
        )

    logger.close()

    print()
    print(
        f"Finished: {output_file}"
    )
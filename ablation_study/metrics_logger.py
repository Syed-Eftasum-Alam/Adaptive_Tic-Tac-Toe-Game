# ============================================================
# metrics_logger.py
#
# Records all metrics from the ablation experiments.
# ============================================================

import csv
import os


FIELDS = [

    # Experiment identification
    "model",
    "seed",
    "episode",

    # Game result
    "result",
    "reward",
    "move_count",

    # Performance
    "agent_win_rate",
    "player_win_rate",
    "draw_rate",

    # Player adaptation
    "skill",
    "skill_change",
    "difficulty",
    "difficulty_name",

    # Exploration
    "epsilon",
    "effective_epsilon",

    # Q-learning convergence
    "avg_q_change",
    "max_q_change",

    # Q-table learning
    "q_updates",
    "episode_q_updates",
    "learned_states",

    # Action selection
    "tactical_moves",
    "epsilon_moves",
    "q_policy_moves",

    # Configuration
    "use_q_learning",
    "use_tactical",
    "use_adaptive",
    "use_epsilon"
]


class ExperimentLogger:

    def __init__(
        self,
        filename="ablation_results.csv"
    ):

        self.filename = filename

        self.initialize_file()

    # --------------------------------------------------------
    # Create CSV file if it doesn't exist
    # --------------------------------------------------------

    def initialize_file(self):

        if os.path.exists(self.filename):
            return

        with open(
            self.filename,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=FIELDS
            )

            writer.writeheader()

    # --------------------------------------------------------
    # Record one episode
    # --------------------------------------------------------

    def record_episode(
        self,
        model,
        seed,
        episode,
        result,
        reward,
        move_count,

        agent_win_rate,
        player_win_rate,
        draw_rate,

        skill,
        skill_change,

        difficulty,
        difficulty_name,

        epsilon,
        effective_epsilon,

        avg_q_change,
        max_q_change,

        q_updates,
        episode_q_updates,
        learned_states,

        tactical_moves,
        epsilon_moves,
        q_policy_moves,

        use_q_learning,
        use_tactical,
        use_adaptive,
        use_epsilon
    ):

        row = {

            "model": model,
            "seed": seed,
            "episode": episode,

            "result": result,
            "reward": reward,
            "move_count": move_count,

            "agent_win_rate": agent_win_rate,
            "player_win_rate": player_win_rate,
            "draw_rate": draw_rate,

            "skill": skill,
            "skill_change": skill_change,

            "difficulty": difficulty,
            "difficulty_name": difficulty_name,

            "epsilon": epsilon,
            "effective_epsilon": effective_epsilon,

            "avg_q_change": avg_q_change,
            "max_q_change": max_q_change,

            "q_updates": q_updates,
            "episode_q_updates": episode_q_updates,
            "learned_states": learned_states,

            "tactical_moves": tactical_moves,
            "epsilon_moves": epsilon_moves,
            "q_policy_moves": q_policy_moves,

            "use_q_learning": use_q_learning,
            "use_tactical": use_tactical,
            "use_adaptive": use_adaptive,
            "use_epsilon": use_epsilon
        }

        with open(
            self.filename,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=FIELDS
            )

            writer.writerow(row)
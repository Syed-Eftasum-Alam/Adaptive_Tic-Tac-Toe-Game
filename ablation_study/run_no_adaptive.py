# ============================================================
# run_no_adaptive.py
#
# ABLATION 2:
# Remove Adaptive Difficulty
#
# Q-learning          = ON
# Tactical policy     = ON
# Adaptive difficulty = OFF
# Epsilon-greedy      = ON
# ============================================================

from ablation_engine import run_experiment


SEEDS = [1, 2, 3, 4, 5]

EPISODES_PER_SEED = 100

OPPONENT = "strong"


if __name__ == "__main__":

    run_experiment(

        model_name="Without Adaptive Difficulty",

        seeds=SEEDS,

        episodes_per_seed=EPISODES_PER_SEED,

        opponent_type=OPPONENT,

        use_q_learning=True,

        use_tactical=True,

        use_adaptive=False,

        use_epsilon=True,

        output_file="ablation_results.csv"
    )
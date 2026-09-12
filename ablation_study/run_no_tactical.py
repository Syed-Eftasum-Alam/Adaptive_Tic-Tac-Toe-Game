# ============================================================
# run_no_tactical.py
#
# ABLATION 1:
# Remove Tactical Policy
#
# Q-learning          = ON
# Tactical policy     = OFF
# Adaptive difficulty = ON
# Epsilon-greedy      = ON
# ============================================================

from ablation_engine import run_experiment


SEEDS = [1, 2, 3, 4, 5]

EPISODES_PER_SEED = 100

OPPONENT = "strong"


if __name__ == "__main__":

    run_experiment(

        model_name="Without Tactical Policy",

        seeds=SEEDS,

        episodes_per_seed=EPISODES_PER_SEED,

        opponent_type=OPPONENT,

        use_q_learning=True,

        use_tactical=False,

        use_adaptive=True,

        use_epsilon=True,

        output_file="ablation_results.csv"
    )
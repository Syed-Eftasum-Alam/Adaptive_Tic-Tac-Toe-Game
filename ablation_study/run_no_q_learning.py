# ============================================================
# run_no_q_learning.py
#
# ABLATION 3:
# Remove Q-learning
#
# Q-learning          = OFF
# Tactical policy     = ON
# Adaptive difficulty = ON
# Epsilon-greedy      = OFF
#
# This becomes a rule-based tactical baseline.
# ============================================================

from ablation_engine import run_experiment


SEEDS = [1, 2, 3, 4, 5]

EPISODES_PER_SEED = 100

OPPONENT = "strong"


if __name__ == "__main__":

    run_experiment(

        model_name="Without Q-Learning",

        seeds=SEEDS,

        episodes_per_seed=EPISODES_PER_SEED,

        opponent_type=OPPONENT,

        use_q_learning=False,

        use_tactical=True,

        use_adaptive=True,

        use_epsilon=False,

        output_file="ablation_results.csv"
    )
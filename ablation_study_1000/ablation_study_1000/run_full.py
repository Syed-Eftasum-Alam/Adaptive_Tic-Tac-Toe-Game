from ablation_engine import run_experiment


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

SEEDS = [
    1,
    2,
    3,
    4,
    5
]

TRAIN_EPISODES = 800
EVAL_EPISODES = 200

OPPONENTS = [
    "random",
    "heuristic",
    "strong"
]

OUTPUT_DIRECTORY = (
    "ablation_output_1000"
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for opponent in OPPONENTS:

        print()
        print(
            "#" * 75
        )

        print(
            "RUNNING FULL MODEL"
        )

        print(
            f"Opponent: {opponent}"
        )

        print(
            "#" * 75
        )

        output_file = (
            f"{OUTPUT_DIRECTORY}/"
            f"ablation_results_full_"
            f"{opponent}_1000.csv"
        )

        run_experiment(

            model_name="Full",

            opponent_name=opponent,

            seeds=SEEDS,

            train_episodes=TRAIN_EPISODES,

            eval_episodes=EVAL_EPISODES,

            output_file=output_file
        )

    print()
    print(
        "FULL MODEL EXPERIMENT COMPLETED."
    )
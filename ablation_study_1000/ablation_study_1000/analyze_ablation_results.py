import os
import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

INPUT_DIRECTORY = (
    "ablation_output_1000"
)

OUTPUT_DIRECTORY = (
    "ablation_output_1000"
)

TRAIN_EPISODES = 800
EVAL_EPISODES = 200


# ============================================================
# FILE DISCOVERY
# ============================================================

def load_results():

    pattern = os.path.join(
        INPUT_DIRECTORY,
        "ablation_results_*_1000.csv"
    )

    files = sorted(
        glob.glob(pattern)
    )

    if not files:

        raise FileNotFoundError(
            "No ablation result files were found.\n"
            f"Expected files inside: "
            f"{INPUT_DIRECTORY}"
        )

    print()
    print(
        "Result files found:"
    )

    for file in files:
        print(
            f"  {file}"
        )

    frames = []

    for file in files:

        df = pd.read_csv(
            file
        )

        frames.append(
            df
        )

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    return combined


# ============================================================
# BASIC RESULT CHECKS
# ============================================================

def validate_results(df):

    print()
    print(
        "=" * 75
    )

    print(
        "VALIDATING RESULTS"
    )

    print(
        "=" * 75
    )

    expected_rows = (
        4
        * 3
        * 5
        * 1000
    )

    actual_rows = len(df)

    print(
        f"Expected rows: {expected_rows}"
    )

    print(
        f"Actual rows  : {actual_rows}"
    )

    if actual_rows != expected_rows:

        print(
            "WARNING: The total number of rows "
            "does not equal 60,000."
        )

        print(
            "This usually means one or more "
            "experiments were not completed."
        )

    # --------------------------------------------------------
    # Check train/evaluation counts.
    # --------------------------------------------------------

    phase_counts = (
        df.groupby(
            [
                "model",
                "opponent",
                "seed",
                "phase"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    print()
    print(
        "Phase counts per model/opponent/seed:"
    )

    print(
        phase_counts
    )

    # --------------------------------------------------------
    # Check Q updates during evaluation.
    # --------------------------------------------------------

    eval_rows = df[
        df["phase"] == "eval"
    ]

    q_update_values = (
        eval_rows
        .groupby(
            [
                "model",
                "opponent",
                "seed"
            ]
        )["q_updates_enabled"]
        .sum()
    )

    if q_update_values.sum() != 0:

        raise RuntimeError(
            "ERROR: Q updates were enabled "
            "during evaluation."
        )

    print()
    print(
        "PASS: Q updates are disabled "
        "during evaluation."
    )


# ============================================================
# MAIN PERFORMANCE SUMMARY
# ============================================================

def create_evaluation_summary(df):

    evaluation = df[
        df["phase"] == "eval"
    ].copy()

    # --------------------------------------------------------
    # Convert result to binary indicators.
    # --------------------------------------------------------

    evaluation["agent_win"] = (
        evaluation["result"]
        == "agent_win"
    ).astype(int)

    evaluation["draw"] = (
        evaluation["result"]
        == "draw"
    ).astype(int)

    evaluation["player_win"] = (
        evaluation["result"]
        == "player_win"
    ).astype(int)

    # --------------------------------------------------------
    # Per seed.
    # --------------------------------------------------------

    per_seed = (
        evaluation
        .groupby(
            [
                "model",
                "opponent",
                "seed"
            ]
        )
        .agg(
            win_rate=(
                "agent_win",
                "mean"
            ),

            draw_rate=(
                "draw",
                "mean"
            ),

            loss_rate=(
                "player_win",
                "mean"
            ),

            average_reward=(
                "reward",
                "mean"
            ),

            average_moves=(
                "move_count",
                "mean"
            ),

            final_skill=(
                "skill",
                "last"
            ),

            final_difficulty=(
                "difficulty",
                "last"
            ),

            average_difficulty=(
                "difficulty",
                "mean"
            ),

            average_effective_epsilon=(
                "effective_epsilon",
                "mean"
            ),

            final_learned_states=(
                "learned_states",
                "last"
            ),
        )
        .reset_index()
    )

    # Convert rates to percentages.
    per_seed["win_rate"] *= 100
    per_seed["draw_rate"] *= 100
    per_seed["loss_rate"] *= 100

    # --------------------------------------------------------
    # DDA balance metrics.
    # --------------------------------------------------------

    per_seed["balance_error"] = (
        abs(
            per_seed["win_rate"]
            - 50.0
        )
    )

    per_seed[
        "within_45_55_percent"
    ] = (
        (
            per_seed["win_rate"] >= 45.0
        )
        &
        (
            per_seed["win_rate"] <= 55.0
        )
    ).astype(int)

    # --------------------------------------------------------
    # Mean and standard deviation over seeds.
    # --------------------------------------------------------

    summary = (
        per_seed
        .groupby(
            [
                "model",
                "opponent"
            ]
        )
        .agg(
            win_rate_mean=(
                "win_rate",
                "mean"
            ),

            win_rate_std=(
                "win_rate",
                "std"
            ),

            draw_rate_mean=(
                "draw_rate",
                "mean"
            ),

            draw_rate_std=(
                "draw_rate",
                "std"
            ),

            loss_rate_mean=(
                "loss_rate",
                "mean"
            ),

            loss_rate_std=(
                "loss_rate",
                "std"
            ),

            reward_mean=(
                "average_reward",
                "mean"
            ),

            reward_std=(
                "average_reward",
                "std"
            ),

            final_skill_mean=(
                "final_skill",
                "mean"
            ),

            final_skill_std=(
                "final_skill",
                "std"
            ),

            final_difficulty_mean=(
                "final_difficulty",
                "mean"
            ),

            final_difficulty_std=(
                "final_difficulty",
                "std"
            ),

            average_difficulty_mean=(
                "average_difficulty",
                "mean"
            ),

            balance_error_mean=(
                "balance_error",
                "mean"
            ),

            balance_error_std=(
                "balance_error",
                "std"
            ),

            within_45_55_percent=(
                "within_45_55_percent",
                "mean"
            ),

            learned_states_mean=(
                "final_learned_states",
                "mean"
            ),
        )
        .reset_index()
    )

    # Percentage of seeds that achieved the
    # 45-55% balance interval.
    summary[
        "within_45_55_percent"
    ] *= 100

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    per_seed_file = os.path.join(
        OUTPUT_DIRECTORY,
        "summary_per_seed.csv"
    )

    summary_file = os.path.join(
        OUTPUT_DIRECTORY,
        "summary_by_model_opponent.csv"
    )

    per_seed.to_csv(
        per_seed_file,
        index=False
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print()
    print(
        "=" * 75
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "=" * 75
    )

    display_columns = [
        "model",
        "opponent",
        "win_rate_mean",
        "win_rate_std",
        "draw_rate_mean",
        "loss_rate_mean",
        "reward_mean",
        "balance_error_mean",
        "within_45_55_percent",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {summary_file}"
    )

    print(
        f"Saved: {per_seed_file}"
    )

    return summary, per_seed


# ============================================================
# GRAPH 1
# AGENT WIN RATE
# ============================================================

def plot_agent_win_rate(
    summary
):

    opponents = [
        "random",
        "heuristic",
        "strong"
    ]

    models = [
        "Full",
        "No Tactical",
        "No Adaptive",
        "No Q"
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    x = np.arange(
        len(opponents)
    )

    width = 0.18

    for index, model in enumerate(
        models
    ):

        values = []

        errors = []

        for opponent in opponents:

            row = summary[
                (
                    summary["model"]
                    == model
                )
                &
                (
                    summary["opponent"]
                    == opponent
                )
            ]

            if len(row) == 0:

                values.append(
                    np.nan
                )

                errors.append(
                    0
                )

            else:

                values.append(
                    row[
                        "win_rate_mean"
                    ].iloc[0]
                )

                errors.append(
                    row[
                        "win_rate_std"
                    ].iloc[0]
                )

        ax.bar(
            x
            + (
                index
                - 1.5
            )
            * width,

            values,

            width,

            yerr=errors,

            capsize=4,

            label=model
        )

    ax.set_xlabel(
        "Opponent"
    )

    ax.set_ylabel(
        "Agent Win Rate (%)"
    )

    ax.set_title(
        "Agent Win Rate During Frozen Evaluation"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            "Random",
            "Heuristic",
            "Strong Tactical"
        ]
    )

    ax.set_ylim(
        0,
        100
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIRECTORY,
        "01_agent_win_rate.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 2
# GAME OUTCOMES
# ============================================================

def plot_game_outcomes(
    summary
):

    opponents = [
        "random",
        "heuristic",
        "strong"
    ]

    models = [
        "Full",
        "No Tactical",
        "No Adaptive",
        "No Q"
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    for axis, opponent in zip(
        axes,
        opponents
    ):

        subset = summary[
            summary["opponent"]
            == opponent
        ].copy()

        subset = (
            subset
            .set_index("model")
            .reindex(models)
        )

        x = np.arange(
            len(models)
        )

        axis.bar(
            x,
            subset[
                "win_rate_mean"
            ],
            label="Agent Win"
        )

        axis.bar(
            x,
            subset[
                "draw_rate_mean"
            ],
            bottom=subset[
                "win_rate_mean"
            ],
            label="Draw"
        )

        axis.bar(
            x,
            subset[
                "loss_rate_mean"
            ],
            bottom=(
                subset[
                    "win_rate_mean"
                ]
                +
                subset[
                    "draw_rate_mean"
                ]
            ),
            label="Player Win"
        )

        axis.set_title(
            opponent.title()
        )

        axis.set_xticks(
            x
        )

        axis.set_xticklabels(
            [
                "Full",
                "No Tac.",
                "No Adapt.",
                "No Q"
            ],
            rotation=20
        )

        axis.set_ylim(
            0,
            100
        )

        axis.set_ylabel(
            "Percentage (%)"
        )

    axes[0].legend()

    plt.suptitle(
        "Evaluation Game Outcomes"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIRECTORY,
        "02_game_outcomes.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 3
# LEARNING CURVES
# ============================================================

def plot_learning_curves(
    df
):

    training = df[
        df["phase"] == "train"
    ].copy()

    training["agent_win"] = (
        training["result"]
        == "agent_win"
    ).astype(float)

    grouped = (
        training
        .groupby(
            [
                "model",
                "opponent",
                "episode"
            ]
        )["agent_win"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for model in [
        "Full",
        "No Tactical",
        "No Adaptive",
        "No Q"
    ]:

        subset = grouped[
            grouped["model"]
            == model
        ]

        # Show heuristic training curve,
        # which is the most informative comparison.
        subset = subset[
            subset["opponent"]
            == "heuristic"
        ]

        if len(subset) == 0:
            continue

        rolling = (
            subset
            .sort_values("episode")
            .set_index("episode")[
                "agent_win"
            ]
            .rolling(
                50,
                min_periods=1
            )
            .mean()
            * 100
        )

        ax.plot(
            rolling.index,
            rolling.values,
            label=model
        )

    ax.axvline(
        TRAIN_EPISODES,
        linestyle="--",
        label="Train/Evaluation Boundary"
    )

    ax.set_xlabel(
        "Episode"
    )

    ax.set_ylabel(
        "Agent Win Rate (%)"
    )

    ax.set_title(
        "Training Learning Curves - Heuristic Opponent"
    )

    ax.set_xlim(
        1,
        TRAIN_EPISODES + EVAL_EPISODES
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIRECTORY,
        "03_learning_curves.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 4
# WIN RATE OVER TIME
# ============================================================

def plot_win_rate_over_time(
    df
):

    training = df[
        df["phase"] == "train"
    ].copy()

    training["agent_win"] = (
        training["result"]
        == "agent_win"
    ).astype(float)

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for model in [
        "Full",
        "No Tactical",
        "No Adaptive",
        "No Q"
    ]:

        subset = training[
            (
                training["model"]
                == model
            )
            &
            (
                training["opponent"]
                == "heuristic"
            )
        ]

        if len(subset) == 0:
            continue

        # Aggregate across seeds.
        grouped = (
            subset
            .groupby("episode")[
                "agent_win"
            ]
            .mean()
            .rolling(
                50,
                min_periods=1
            )
            .mean()
            * 100
        )

        ax.plot(
            grouped.index,
            grouped.values,
            label=model
        )

    ax.set_xlabel(
        "Training Episode"
    )

    ax.set_ylabel(
        "Mean Agent Win Rate (%)"
    )

    ax.set_title(
        "Training Win Rate Over Time - Heuristic Opponent"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIRECTORY,
        "04_win_rate_over_time.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 5
# Q VALUE CONVERGENCE
# ============================================================

def plot_q_value_convergence(
    df
):

    training = df[
        df["phase"] == "train"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for model in [
        "Full",
        "No Tactical",
        "No Adaptive"
    ]:

        subset = training[
            (
                training["model"]
                == model
            )
            &
            (
                training["opponent"]
                == "heuristic"
            )
        ]

        if len(subset) == 0:
            continue

        grouped = (
            subset
            .groupby("episode")[
                "avg_abs_q_change"
            ]
            .mean()
            .rolling(
                50,
                min_periods=1
            )
            .mean()
        )

        ax.plot(
            grouped.index,
            grouped.values,
            label=model
        )

    ax.set_xlabel(
        "Training Episode"
    )

    ax.set_ylabel(
        "Average |ΔQ|"
    )

    ax.set_title(
        "Q-Value Update Magnitude During Training"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIRECTORY,
        "05_q_value_convergence.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 6
# LEARNED STATES
# ============================================================

def plot_learned_states(
    df
):

    training = df[
        df["phase"] == "train"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for model in [
        "Full",
        "No Tactical",
        "No Adaptive",
        "No Q"
    ]:

        subset = training[
            (
                training["model"]
                == model
            )
            &
            (
                training["opponent"]
                == "heuristic"
            )
        ]

        if len(subset) == 0:
            continue

        grouped = (
            subset
            .groupby("episode")[
                "learned_states"
            ]
            .mean()
        )

        ax.plot(
            grouped.index,
            grouped.values,
            label=model
        )

    ax.set_xlabel(
        "Training Episode"
    )

    ax.set_ylabel(
        "Number of Learned States"
    )

    ax.set_title(
        "Growth of Learned State Space"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIRECTORY,
        "06_learned_states.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 7
# PLAYER SKILL
# ============================================================

def plot_player_skill(
    df
):

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for model in [
        "Full",
        "No Tactical",
        "No Adaptive",
        "No Q"
    ]:

        subset = df[
            (
                df["model"]
                == model
            )
            &
            (
                df["opponent"]
                == "heuristic"
            )
        ]

        if len(subset) == 0:
            continue

        grouped = (
            subset
            .groupby("episode")[
                "skill"
            ]
            .mean()
        )

        ax.plot(
            grouped.index,
            grouped.values,
            label=model
        )

    ax.axvline(
        TRAIN_EPISODES,
        linestyle="--",
        label="Train/Evaluation Boundary"
    )

    ax.set_xlabel(
        "Episode"
    )

    ax.set_ylabel(
        "Player Skill"
    )

    ax.set_title(
        "Dynamic Player Skill Progression"
    )

    ax.set_ylim(
        0,
        100
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIRECTORY,
        "07_player_skill.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 8
# DIFFICULTY ADAPTATION
# ============================================================

def plot_difficulty(
    df
):

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for model in [
        "Full",
        "No Tactical",
        "No Adaptive",
        "No Q"
    ]:

        subset = df[
            (
                df["model"]
                == model
            )
            &
            (
                df["opponent"]
                == "heuristic"
            )
        ]

        if len(subset) == 0:
            continue

        grouped = (
            subset
            .groupby("episode")[
                "difficulty"
            ]
            .mean()
        )

        ax.plot(
            grouped.index,
            grouped.values,
            label=model
        )

    ax.axvline(
        TRAIN_EPISODES,
        linestyle="--",
        label="Train/Evaluation Boundary"
    )

    ax.set_xlabel(
        "Episode"
    )

    ax.set_ylabel(
        "Difficulty Level"
    )

    ax.set_title(
        "Dynamic Difficulty Adaptation"
    )

    ax.set_ylim(
        1,
        5
    )

    ax.set_yticks(
        [
            1,
            2,
            3,
            4,
            5
        ]
    )

    ax.set_yticklabels(
        [
            "Very Easy",
            "Easy",
            "Normal",
            "Hard",
            "Expert"
        ]
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIRECTORY,
        "08_difficulty_adaptation.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 9
# EPSILON
# ============================================================

def plot_epsilon(
    df
):

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for model in [
        "Full",
        "No Tactical",
        "No Adaptive"
    ]:

        subset = df[
            (
                df["model"]
                == model
            )
            &
            (
                df["opponent"]
                == "heuristic"
            )
            &
            (
                df["phase"]
                == "train"
            )
        ]

        if len(subset) == 0:
            continue

        grouped = (
            subset
            .groupby("episode")[
                "epsilon"
            ]
            .mean()
        )

        ax.plot(
            grouped.index,
            grouped.values,
            label=model
        )

    ax.set_xlabel(
        "Training Episode"
    )

    ax.set_ylabel(
        "Stored Epsilon"
    )

    ax.set_title(
        "Epsilon Decay During Training"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIRECTORY,
        "09_epsilon.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


# ============================================================
# GRAPH 10
# ACTION SELECTION
# ============================================================

def plot_action_selection(
    df
):

    evaluation = df[
        df["phase"] == "eval"
    ].copy()

    summary = (
        evaluation
        .groupby(
            "model"
        )[
            [
                "tactical_moves",
                "epsilon_moves",
                "q_policy_moves"
            ]
        ]
        .sum()
    )

    # No-Q fallback actions are not included in
    # q_policy_moves by design.

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    summary.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Number of Actions"
    )

    ax.set_title(
        "Action Selection During Frozen Evaluation"
    )

    ax.tick_params(
        axis="x",
        rotation=20
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIRECTORY,
        "10_action_selection.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


# ============================================================
# CREATE OPPONENT SUMMARY
# ============================================================

def create_opponent_summary(
    summary
):

    rows = []

    for opponent in [
        "random",
        "heuristic",
        "strong"
    ]:

        subset = summary[
            summary["opponent"]
            == opponent
        ]

        if len(subset) == 0:
            continue

        rows.append({

            "opponent":
                opponent,

            "best_model":
                subset.loc[
                    subset[
                        "win_rate_mean"
                    ].idxmax(),
                    "model"
                ],

            "best_win_rate":
                subset[
                    "win_rate_mean"
                ].max(),

            "best_balance_model":
                subset.loc[
                    subset[
                        "balance_error_mean"
                    ].idxmin(),
                    "model"
                ],

            "lowest_balance_error":
                subset[
                    "balance_error_mean"
                ].min(),
        })

    opponent_summary = pd.DataFrame(
        rows
    )

    path = os.path.join(
        OUTPUT_DIRECTORY,
        "summary_by_opponent.csv"
    )

    opponent_summary.to_csv(
        path,
        index=False
    )

    print()
    print(
        "Opponent-level summary:"
    )

    print(
        opponent_summary.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    df = load_results()

    validate_results(
        df
    )

    summary, per_seed = (
        create_evaluation_summary(
            df
        )
    )

    create_opponent_summary(
        summary
    )

    plot_agent_win_rate(
        summary
    )

    plot_game_outcomes(
        summary
    )

    plot_learning_curves(
        df
    )

    plot_win_rate_over_time(
        df
    )

    plot_q_value_convergence(
        df
    )

    plot_learned_states(
        df
    )

    plot_player_skill(
        df
    )

    plot_difficulty(
        df
    )

    plot_epsilon(
        df
    )

    plot_action_selection(
        df
    )

    print()
    print(
        "=" * 75
    )

    print(
        "ALL ANALYSIS COMPLETED."
    )

    print(
        "=" * 75
    )

    print()
    print(
        f"Output directory:"
        f" {OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":

    main()
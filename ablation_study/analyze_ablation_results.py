# ============================================================
# analyze_ablation_results.py
#
# Generates graphs and statistical summaries for the
# ablation experiments.
# ============================================================

import os

import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "ablation_results.csv"

OUTPUT_DIR = "ablation_output"

MOVING_WINDOW = 10


# ============================================================
# Setup
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


df = pd.read_csv(INPUT_FILE)


if df.empty:

    raise ValueError(
        "ablation_results.csv is empty."
    )


print()
print("=" * 80)
print("ABLATION STUDY ANALYSIS")
print("=" * 80)

print(
    f"Total records: {len(df)}"
)

print(
    "\nModels:"
)

for model in df["model"].unique():

    print(
        f"  - {model}"
    )


# ============================================================
# Helper function
# ============================================================

def save_show(
    filename,
    title
):

    plt.title(title)

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"Saved: {path}"
    )

    plt.show()


# ============================================================
# 1. FINAL PERFORMANCE SUMMARY
# ============================================================

summary = (
    df.groupby(
        ["model", "seed"]
    )
    .agg(
        agent_win_rate=(
            "result",
            lambda x:
            100.0 * (x == "A").mean()
        ),

        player_win_rate=(
            "result",
            lambda x:
            100.0 * (x == "P").mean()
        ),

        draw_rate=(
            "result",
            lambda x:
            100.0 * (x == "D").mean()
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

        final_epsilon=(
            "epsilon",
            "last"
        ),

        final_learned_states=(
            "learned_states",
            "last"
        ),

        final_q_updates=(
            "q_updates",
            "last"
        )
    )
    .reset_index()
)


print()
print("=" * 80)
print("PER-SEED PERFORMANCE")
print("=" * 80)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 2. MEAN ± STANDARD DEVIATION
# ============================================================

statistical_summary = (
    summary.groupby("model")
    .agg(

        win_rate_mean=(
            "agent_win_rate",
            "mean"
        ),

        win_rate_std=(
            "agent_win_rate",
            "std"
        ),

        player_win_mean=(
            "player_win_rate",
            "mean"
        ),

        player_win_std=(
            "player_win_rate",
            "std"
        ),

        draw_mean=(
            "draw_rate",
            "mean"
        ),

        draw_std=(
            "draw_rate",
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

        moves_mean=(
            "average_moves",
            "mean"
        ),

        moves_std=(
            "average_moves",
            "std"
        )
    )
    .reset_index()
)


print()
print("=" * 80)
print("MEAN ± STANDARD DEVIATION")
print("=" * 80)

for _, row in statistical_summary.iterrows():

    print(
        f"\n{row['model']}"
    )

    print(
        f"  Agent Win Rate : "
        f"{row['win_rate_mean']:.2f} ± "
        f"{row['win_rate_std']:.2f}%"
    )

    print(
        f"  Player Win Rate: "
        f"{row['player_win_mean']:.2f} ± "
        f"{row['player_win_std']:.2f}%"
    )

    print(
        f"  Draw Rate      : "
        f"{row['draw_mean']:.2f} ± "
        f"{row['draw_std']:.2f}%"
    )

    print(
        f"  Average Reward : "
        f"{row['reward_mean']:.3f} ± "
        f"{row['reward_std']:.3f}"
    )


statistical_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ablation_statistical_summary.csv"
    ),
    index=False
)


# ============================================================
# 3. AGENT WIN RATE
# ============================================================

win_rates = (
    summary.groupby("model")
    ["agent_win_rate"]
    .mean()
)


plt.figure(
    figsize=(10, 6)
)

win_rates.plot(
    kind="bar"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Agent Win Rate (%)"
)

save_show(
    "01_agent_win_rate.png",
    "Ablation Study: Agent Win Rate"
)


# ============================================================
# 4. GAME OUTCOME COMPARISON
# ============================================================
# ============================================================
# 4. GAME OUTCOME COMPARISON
# ============================================================

outcomes = (
    df.groupby("model")["result"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    * 100
)

# Make sure all possible outcomes exist.
# This is necessary because an experiment may contain
# zero draws, zero wins, or zero losses.
for result_code in ["A", "P", "D"]:
    if result_code not in outcomes.columns:
        outcomes[result_code] = 0.0

outcomes = outcomes.rename(
    columns={
        "A": "Agent Win",
        "P": "Player Win",
        "D": "Draw"
    }
)

# Explicitly enforce the desired column order.
outcomes = outcomes[
    ["Agent Win", "Player Win", "Draw"]
]

print()
print("=" * 80)
print("GAME OUTCOME PERCENTAGES")
print("=" * 80)

print(
    outcomes.to_string()
)

plt.figure(
    figsize=(11, 6)
)

outcomes.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Percentage (%)"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.legend(
    title="Outcome"
)

plt.grid(
    axis="y",
    alpha=0.3
)

save_show(
    "02_game_outcomes.png",
    "Game Outcome Comparison"
)

# ============================================================
# 5. LEARNING CURVES
# ============================================================

episode_reward = (
    df.groupby(
        ["model", "episode"]
    )["reward"]
    .mean()
    .reset_index()
)


plt.figure(
    figsize=(12, 7)
)


for model in episode_reward["model"].unique():

    data = episode_reward[
        episode_reward["model"] == model
    ]

    moving = (
        data["reward"]
        .rolling(
            MOVING_WINDOW,
            min_periods=1
        )
        .mean()
    )

    plt.plot(
        data["episode"],
        moving,
        label=model
    )


plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Moving Average Reward"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)


save_show(
    "03_learning_curves.png",
    "Learning Curves Across Ablation Models"
)


# ============================================================
# 6. CUMULATIVE WIN RATE
# ============================================================

episode_win = (
    df.groupby(
        ["model", "episode"]
    )["agent_win_rate"]
    .mean()
    .reset_index()
)


plt.figure(
    figsize=(12, 7)
)


for model in episode_win["model"].unique():

    data = episode_win[
        episode_win["model"] == model
    ]

    plt.plot(
        data["episode"],
        data["agent_win_rate"],
        label=model
    )


plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Agent Win Rate (%)"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)


save_show(
    "04_win_rate_over_time.png",
    "Agent Win Rate Over Training"
)


# ============================================================
# 7. Q-VALUE CONVERGENCE
# ============================================================

q_change = (
    df.groupby(
        ["model", "episode"]
    )["avg_q_change"]
    .mean()
    .reset_index()
)


plt.figure(
    figsize=(12, 7)
)


for model in q_change["model"].unique():

    data = q_change[
        q_change["model"] == model
    ]

    moving = (
        data["avg_q_change"]
        .rolling(
            MOVING_WINDOW,
            min_periods=1
        )
        .mean()
    )

    plt.plot(
        data["episode"],
        moving,
        label=model
    )


plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Average |ΔQ|"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)


save_show(
    "05_q_value_convergence.png",
    "Q-value Convergence"
)


# ============================================================
# 8. LEARNED Q-TABLE STATES
# ============================================================

states = (
    df.groupby(
        ["model", "episode"]
    )["learned_states"]
    .mean()
    .reset_index()
)


plt.figure(
    figsize=(12, 7)
)


for model in states["model"].unique():

    data = states[
        states["model"] == model
    ]

    plt.plot(
        data["episode"],
        data["learned_states"],
        label=model
    )


plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Number of Learned States"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)


save_show(
    "06_learned_states.png",
    "Q-table Growth"
)


# ============================================================
# 9. PLAYER SKILL PROGRESSION
# ============================================================

skill = (
    df.groupby(
        ["model", "episode"]
    )["skill"]
    .mean()
    .reset_index()
)


plt.figure(
    figsize=(12, 7)
)


for model in skill["model"].unique():

    data = skill[
        skill["model"] == model
    ]

    plt.plot(
        data["episode"],
        data["skill"],
        label=model
    )


plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Player Skill"
)

plt.ylim(
    0,
    100
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)


save_show(
    "07_player_skill.png",
    "Player Skill Progression"
)


# ============================================================
# 10. DIFFICULTY ADAPTATION
# ============================================================

difficulty = (
    df.groupby(
        ["model", "episode"]
    )["difficulty"]
    .mean()
    .reset_index()
)


plt.figure(
    figsize=(12, 7)
)


for model in difficulty["model"].unique():

    data = difficulty[
        difficulty["model"] == model
    ]

    plt.plot(
        data["episode"],
        data["difficulty"],
        label=model
    )


plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Difficulty Level"
)

plt.yticks(
    [1, 2, 3, 4, 5]
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)


save_show(
    "08_difficulty_adaptation.png",
    "Adaptive Difficulty Progression"
)


# ============================================================
# 11. EPSILON
# ============================================================

epsilon = (
    df.groupby(
        ["model", "episode"]
    )["epsilon"]
    .mean()
    .reset_index()
)


plt.figure(
    figsize=(12, 7)
)


for model in epsilon["model"].unique():

    data = epsilon[
        epsilon["model"] == model
    ]

    plt.plot(
        data["episode"],
        data["epsilon"],
        label=model
    )


plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Epsilon"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)


save_show(
    "09_epsilon.png",
    "Exploration Rate During Training"
)


# ============================================================
# 12. ACTION SELECTION
# ============================================================

action_stats = (
    df.groupby("model")[
        [
            "tactical_moves",
            "epsilon_moves",
            "q_policy_moves"
        ]
    ]
    .mean()
)


action_stats.columns = [
    "Tactical",
    "Epsilon",
    "Q-policy"
]


plt.figure(
    figsize=(11, 6)
)


action_stats.plot(
    kind="bar",
    figsize=(11, 6)
)


plt.xlabel(
    "Model"
)

plt.ylabel(
    "Average Number of Moves per Episode"
)

save_show(
    "10_action_selection.png",
    "Action Selection Behavior"
)


# ============================================================
# 13. FINAL EPISODE METRICS
# ============================================================

final_episode = (
    df.groupby(
        ["model", "seed"]
    )
    .tail(1)
)


final_metrics = (
    final_episode.groupby("model")
    [
        [
            "skill",
            "difficulty",
            "epsilon",
            "learned_states",
            "q_updates"
        ]
    ]
    .mean()
)


print()
print("=" * 80)
print("FINAL MODEL METRICS")
print("=" * 80)

print(
    final_metrics.to_string()
)


final_metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_model_metrics.csv"
    )
)


# ============================================================
# 14. Final report
# ============================================================

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"\nAll graphs and tables are available in:"
    f"\n{OUTPUT_DIR}/"
)

print()
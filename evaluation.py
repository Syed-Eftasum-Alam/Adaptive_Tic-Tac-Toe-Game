import json
import os
import math
import matplotlib.pyplot as plt

DATA_FILE = "users_data.txt"
OUTPUT_DIR = "evaluation_output"


def load_users_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Keep evaluation.py in the same folder "
            f"as users_data.txt."
        )

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("users_data.txt must contain a JSON object.")

    return data


def choose_user(users):
    if not users:
        raise ValueError("No users found in users_data.txt.")

    print("\nAvailable users:")
    keys = list(users.keys())

    for i, key in enumerate(keys, start=1):
        display_name = users[key].get("username", key)
        print(f"{i}. {display_name}")

    while True:
        choice = input("\nSelect user number: ").strip()

        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(keys):
                return keys[index]

        print("Invalid selection. Try again.")


def safe_list(profile, key):
    value = profile.get(key, [])
    return value if isinstance(value, list) else []


def reconstruct_episode_history(profile):
    """
    Tries to use saved episode_history if available.
    Otherwise reconstructs a simplified history from recent_results.
    """
    history = safe_list(profile, "episode_history")

    if history:
        return history

    recent_results = safe_list(profile, "recent_results")
    history = []

    skill = profile.get("player_skill", 50)
    difficulty = profile.get("difficulty_level", 1)
    epsilon = profile.get("epsilon", 1.0)

    # If only recent_results are available, we cannot recover exact historical
    # values. We create a best-effort sequence for outcome-based evaluation.
    for i, result in enumerate(recent_results, start=1):
        history.append({
            "episode": i,
            "result": result,
            "skill": skill,
            "skill_change": 0,
            "difficulty": difficulty,
            "difficulty_name": profile.get("difficulty_name", "Unknown"),
            "epsilon": epsilon,
            "avg_q_change": 0.0,
            "max_q_change": 0.0,
            "q_updates": 0,
            "learned_states": len(profile.get("q_table", {})),
            "player_win_rate": 0.0
        })

    return history


def reward_from_result(result):
    if result == "A":
        return 10
    elif result == "D":
        return 3
    elif result == "P":
        return -10
    return 0


def moving_average(values, window=5):
    if not values:
        return []

    result = []

    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        result.append(sum(chunk) / len(chunk))

    return result


def cumulative_agent_win_rate(results):
    rates = []
    wins = 0

    for i, result in enumerate(results, start=1):
        if result == "A":
            wins += 1
        rates.append(100.0 * wins / i)

    return rates


def calculate_difficulty_changes(history):
    """
    Returns the per-episode change in difficulty level.

    Example:
        difficulty: [1, 1, 2, 3, 2]
        change:     [0, 0, +1, +1, -1]
    """
    if not history:
        return []

    changes = []
    previous = None

    for h in history:
        current = int(h.get("difficulty", 1))

        if previous is None:
            changes.append(0)
        else:
            changes.append(current - previous)

        previous = current

    return changes


def difficulty_change_label(change):
    if change > 0:
        return f"UP (+{change})"
    elif change < 0:
        return f"DOWN ({change})"
    return "SAME"


def print_profile_summary(profile, history):
    username = profile.get("username", "Unknown")

    cumulative_games = profile.get("cumulative_games", len(history))
    player_wins = profile.get("cumulative_player_wins", 0)
    agent_wins = profile.get("cumulative_agent_wins", 0)
    draws = profile.get("cumulative_draws", 0)

    if cumulative_games > 0:
        player_rate = 100.0 * player_wins / cumulative_games
        agent_rate = 100.0 * agent_wins / cumulative_games
        draw_rate = 100.0 * draws / cumulative_games
    else:
        player_rate = agent_rate = draw_rate = 0.0

    print("\n" + "=" * 78)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 78)
    print(f"User                     : {username}")
    print(f"Total games              : {cumulative_games}")
    print(f"Player wins              : {player_wins} ({player_rate:.2f}%)")
    print(f"RL agent wins            : {agent_wins} ({agent_rate:.2f}%)")
    print(f"Draws                    : {draws} ({draw_rate:.2f}%)")
    print(f"Current expertise        : {profile.get('player_skill', 'N/A')}/100")
    print(
        f"Current difficulty       : "
        f"{profile.get('difficulty_name', 'N/A')} "
        f"({profile.get('difficulty_level', 'N/A')}/5)"
    )
    print(f"Current epsilon          : {profile.get('epsilon', 'N/A')}")
    print(f"Learned Q-table states   : {len(profile.get('q_table', {}))}")
    print(f"Total Q updates          : {profile.get('cumulative_q_updates', 'N/A')}")
    print(f"Last Q value             : {profile.get('last_q_value', 'N/A')}")
    print("=" * 78)


def print_episode_table(history):
    if not history:
        print("\nNo episode-level history is available.")
        return

    diff_changes = calculate_difficulty_changes(history)

    print("\nEPISODE-LEVEL EVALUATION")
    print("-" * 126)
    print(
        f"{'Ep':<4}"
        f"{'Result':<8}"
        f"{'Reward':<8}"
        f"{'Skill':<8}"
        f"{'dSkill':<9}"
        f"{'Diff':<7}"
        f"{'dDiff':<8}"
        f"{'Diff Change':<14}"
        f"{'Epsilon':<10}"
        f"{'Avg|dQ|':<12}"
        f"{'Max|dQ|':<12}"
        f"{'Qupd':<7}"
        f"{'States':<8}"
    )
    print("-" * 126)

    for index, h in enumerate(history):
        result = h.get("result", "-")
        reward = reward_from_result(result)
        d_diff = diff_changes[index]

        print(
            f"{h.get('episode', '-'):<4}"
            f"{result:<8}"
            f"{reward:<8}"
            f"{h.get('skill', 0):<8}"
            f"{h.get('skill_change', 0):<9}"
            f"{h.get('difficulty', 0):<7}"
            f"{d_diff:<8}"
            f"{difficulty_change_label(d_diff):<14}"
            f"{float(h.get('epsilon', 0)):<10.3f}"
            f"{float(h.get('avg_q_change', 0)):<12.4f}"
            f"{float(h.get('max_q_change', 0)):<12.4f}"
            f"{h.get('q_updates', 0):<7}"
            f"{h.get('learned_states', 0):<8}"
        )

    print("-" * 126)

def save_table_to_txt(profile, history, output_path):
    username = profile.get("username", "Unknown")

    cumulative_games = profile.get("cumulative_games", len(history))
    player_wins = profile.get("cumulative_player_wins", 0)
    agent_wins = profile.get("cumulative_agent_wins", 0)
    draws = profile.get("cumulative_draws", 0)

    lines = []
    lines.append("=" * 122)
    lines.append("MODEL EVALUATION REPORT")
    lines.append("=" * 122)
    lines.append(f"User: {username}")
    lines.append(f"Total games: {cumulative_games}")
    lines.append(f"Player wins: {player_wins}")
    lines.append(f"RL agent wins: {agent_wins}")
    lines.append(f"Draws: {draws}")
    lines.append(f"Current expertise: {profile.get('player_skill', 'N/A')}/100")
    lines.append(
        f"Current difficulty: {profile.get('difficulty_name', 'N/A')} "
        f"({profile.get('difficulty_level', 'N/A')}/5)"
    )
    lines.append(f"Current epsilon: {profile.get('epsilon', 'N/A')}")
    lines.append(f"Learned states: {len(profile.get('q_table', {}))}")
    lines.append(f"Total Q updates: {profile.get('cumulative_q_updates', 'N/A')}")
    lines.append(f"Last Q value: {profile.get('last_q_value', 'N/A')}")
    lines.append("")

    if history:
        diff_changes = calculate_difficulty_changes(history)

        lines.append("EPISODE TABLE")
        lines.append("-" * 122)
        lines.append(
            f"{'Ep':<4}{'R':<4}{'Reward':<8}{'Skill':<8}{'dSkill':<9}"
            f"{'Diff':<7}{'dDiff':<8}{'Diff Change':<14}"
            f"{'Eps':<9}{'Avg|dQ|':<12}{'Max|dQ|':<12}"
            f"{'Qupd':<7}{'States':<8}"
        )
        lines.append("-" * 122)

        for index, h in enumerate(history):
            result = h.get("result", "-")
            d_diff = diff_changes[index]

            lines.append(
                f"{h.get('episode', '-'):<4}"
                f"{result:<4}"
                f"{reward_from_result(result):<8}"
                f"{h.get('skill', 0):<8}"
                f"{h.get('skill_change', 0):<9}"
                f"{h.get('difficulty', 0):<7}"
                f"{d_diff:<8}"
                f"{difficulty_change_label(d_diff):<14}"
                f"{float(h.get('epsilon', 0)):<9.3f}"
                f"{float(h.get('avg_q_change', 0)):<12.4f}"
                f"{float(h.get('max_q_change', 0)):<12.4f}"
                f"{h.get('q_updates', 0):<7}"
                f"{h.get('learned_states', 0):<8}"
            )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def save_graph(fig, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def plot_reward_curve(history):
    if not history:
        return None

    episodes = [h.get("episode", i + 1) for i, h in enumerate(history)]
    rewards = [reward_from_result(h.get("result", "")) for h in history]
    avg_rewards = moving_average(rewards, window=5)

    fig = plt.figure(figsize=(9, 5))
    plt.plot(episodes, rewards, marker="o", label="Episode reward")
    plt.plot(episodes, avg_rewards, linewidth=2, label="5-episode moving average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    return fig


def plot_q_change(history):
    if not history:
        return None

    episodes = [h.get("episode", i + 1) for i, h in enumerate(history)]
    avg_q = [float(h.get("avg_q_change", 0)) for h in history]

    fig = plt.figure(figsize=(9, 5))
    plt.plot(episodes, avg_q, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Average |ΔQ|")
    plt.title("Q-value Update Magnitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_agent_win_rate(history):
    if not history:
        return None

    results = [h.get("result", "") for h in history]
    episodes = list(range(1, len(results) + 1))
    win_rates = cumulative_agent_win_rate(results)

    fig = plt.figure(figsize=(9, 5))
    plt.plot(episodes, win_rates, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Agent Win Rate (%)")
    plt.title("Cumulative RL Agent Win Rate")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_expertise(history):
    if not history:
        return None

    episodes = [h.get("episode", i + 1) for i, h in enumerate(history)]
    skills = [float(h.get("skill", 0)) for h in history]

    fig = plt.figure(figsize=(9, 5))
    plt.plot(episodes, skills, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Player Expertise")
    plt.title("Player Expertise Progression")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_difficulty(history):
    if not history:
        return None

    episodes = [h.get("episode", i + 1) for i, h in enumerate(history)]
    difficulty = [float(h.get("difficulty", 1)) for h in history]

    fig = plt.figure(figsize=(9, 5))
    plt.step(episodes, difficulty, where="mid")
    plt.scatter(episodes, difficulty)
    plt.xlabel("Episode")
    plt.ylabel("Difficulty Level")
    plt.title("Adaptive Difficulty by Episode")
    plt.ylim(0.5, 5.5)
    plt.yticks([1, 2, 3, 4, 5])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig



def plot_dynamic_difficulty_change(history):
    """
    Shows:
      1. Actual difficulty level at each episode.
      2. Episodes where difficulty increased.
      3. Episodes where difficulty decreased.
      4. Episodes where difficulty stayed unchanged.
    """
    if not history:
        return None

    episodes = [h.get("episode", i + 1) for i, h in enumerate(history)]
    difficulty = [int(h.get("difficulty", 1)) for h in history]
    changes = calculate_difficulty_changes(history)

    fig = plt.figure(figsize=(11, 6))

    # Step curve is appropriate because difficulty is discrete (1-5).
    plt.step(
        episodes,
        difficulty,
        where="mid",
        linewidth=2.5,
        label="Difficulty level"
    )

    # Separate markers make dynamic changes visually obvious.
    up_x = [ep for ep, change in zip(episodes, changes) if change > 0]
    up_y = [d for d, change in zip(difficulty, changes) if change > 0]

    down_x = [ep for ep, change in zip(episodes, changes) if change < 0]
    down_y = [d for d, change in zip(difficulty, changes) if change < 0]

    same_x = [
        ep for i, (ep, change) in enumerate(zip(episodes, changes))
        if change == 0 and i != 0
    ]
    same_y = [
        d for i, (d, change) in enumerate(zip(difficulty, changes))
        if change == 0 and i != 0
    ]

    if up_x:
        plt.scatter(
            up_x,
            up_y,
            marker="^",
            s=110,
            label="Difficulty increased",
            zorder=3
        )

    if down_x:
        plt.scatter(
            down_x,
            down_y,
            marker="v",
            s=110,
            label="Difficulty decreased",
            zorder=3
        )

    if same_x:
        plt.scatter(
            same_x,
            same_y,
            marker="o",
            s=45,
            label="Difficulty unchanged",
            zorder=2
        )

    # Annotate only actual changes.
    for ep, level, change in zip(episodes, difficulty, changes):
        if change > 0:
            plt.annotate(
                f"+{change}",
                (ep, level),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center"
            )
        elif change < 0:
            plt.annotate(
                str(change),
                (ep, level),
                textcoords="offset points",
                xytext=(0, -16),
                ha="center"
            )

    plt.xlabel("Episode")
    plt.ylabel("Difficulty Level")
    plt.title("Dynamic Difficulty Adaptation Across Episodes")
    plt.ylim(0.5, 5.5)
    plt.yticks(
        [1, 2, 3, 4, 5],
        ["1 - Very Easy", "2 - Easy", "3 - Normal", "4 - Hard", "5 - Expert"]
    )
    plt.xticks(episodes)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    return fig

def plot_epsilon(history):
    if not history:
        return None

    episodes = [h.get("episode", i + 1) for i, h in enumerate(history)]
    epsilon = [float(h.get("epsilon", 0)) for h in history]

    fig = plt.figure(figsize=(9, 5))
    plt.plot(episodes, epsilon, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Exploration Rate (Epsilon)")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_outcomes(profile):
    labels = ["Player Wins", "RL Agent Wins", "Draws"]
    values = [
        profile.get("cumulative_player_wins", 0),
        profile.get("cumulative_agent_wins", 0),
        profile.get("cumulative_draws", 0)
    ]

    fig = plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylabel("Count")
    plt.title("Overall Win / Loss / Draw Distribution")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    return fig


def plot_learned_states(history):
    if not history:
        return None

    episodes = [h.get("episode", i + 1) for i, h in enumerate(history)]
    states = [float(h.get("learned_states", 0)) for h in history]

    fig = plt.figure(figsize=(9, 5))
    plt.plot(episodes, states, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Learned States")
    plt.title("Q-table State Growth")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def create_combined_dashboard(profile, history):
    if not history:
        return None

    episodes = [h.get("episode", i + 1) for i, h in enumerate(history)]
    results = [h.get("result", "") for h in history]
    rewards = [reward_from_result(r) for r in results]
    avg_rewards = moving_average(rewards, window=5)
    skills = [float(h.get("skill", 0)) for h in history]
    difficulty = [float(h.get("difficulty", 1)) for h in history]
    epsilon = [float(h.get("epsilon", 0)) for h in history]
    q_change = [float(h.get("avg_q_change", 0)) for h in history]
    states = [float(h.get("learned_states", 0)) for h in history]
    win_rate = cumulative_agent_win_rate(results)

    fig = plt.figure(figsize=(14, 12))

    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(episodes, rewards, marker="o", label="Reward")
    ax1.plot(episodes, avg_rewards, linewidth=2, label="Moving avg")
    ax1.set_title("Reward per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(episodes, q_change, marker="o")
    ax2.set_title("Average |ΔQ|")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Q-value change")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(episodes, skills, marker="o", label="Expertise")
    ax3.step(episodes, [d * 20 for d in difficulty], where="mid",
             label="Difficulty × 20")
    ax3.set_title("Expertise and Difficulty Adaptation")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Scale")
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(episodes, epsilon, marker="o")
    ax4.set_title("Exploration Rate")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Epsilon")
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(episodes, win_rate, marker="o")
    ax5.set_title("Cumulative Agent Win Rate")
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Win Rate (%)")
    ax5.set_ylim(0, 100)
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(3, 2, 6)
    ax6.plot(episodes, states, marker="o")
    ax6.set_title("Q-table State Growth")
    ax6.set_xlabel("Episode")
    ax6.set_ylabel("Learned States")
    ax6.grid(True, alpha=0.3)

    username = profile.get("username", "Unknown")
    fig.suptitle(f"RL Model Evaluation Dashboard - {username}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    return fig


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    users = load_users_data()
    user_key = choose_user(users)
    profile = users[user_key]

    history = reconstruct_episode_history(profile)

    print_profile_summary(profile, history)
    print_episode_table(history)

    safe_username = "".join(
        ch if ch.isalnum() or ch in "-_" else "_"
        for ch in profile.get("username", user_key)
    )

    report_path = os.path.join(
        OUTPUT_DIR,
        f"{safe_username}_evaluation_report.txt"
    )
    save_table_to_txt(profile, history, report_path)

    figures = []

    graph_functions = [
        ("reward_curve.png", plot_reward_curve(history)),
        ("q_value_change.png", plot_q_change(history)),
        ("agent_win_rate.png", plot_agent_win_rate(history)),
        ("expertise_progression.png", plot_expertise(history)),
        ("difficulty_adaptation.png", plot_difficulty(history)),
        ("dynamic_difficulty_change.png", plot_dynamic_difficulty_change(history)),
        ("epsilon_curve.png", plot_epsilon(history)),
        ("outcome_distribution.png", plot_outcomes(profile)),
        ("learned_states.png", plot_learned_states(history)),
        ("evaluation_dashboard.png", create_combined_dashboard(profile, history))
    ]

    print("\nGenerated files:")

    for filename, fig in graph_functions:
        if fig is not None:
            path = save_graph(fig, f"{safe_username}_{filename}")
            figures.append(path)
            print(f"- {path}")

    print(f"- {report_path}")

    print(
        "\nEvaluation complete. Open the evaluation_output folder "
        "to view all graphs and the tabular report."
    )

    if figures:
        plt.show()


if __name__ == "__main__":
    main()
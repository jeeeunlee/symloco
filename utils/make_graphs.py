import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

LOGS_DIR = "logs"


def make_df(logs_dir: str):
    dfs = []
    for test in next(os.walk(logs_dir))[1]:
        if test == "old_logs":
            continue
        test_df = pd.read_csv(f"{logs_dir}/{test}/progress.csv")
        test_df["test_name"] = test
        dfs.append(test_df)
    return pd.concat(dfs)


if __name__ == "__main__":
    df = make_df(LOGS_DIR)

    sns.set_theme()
    ax = sns.lineplot(
        x="time/total_timesteps", y="rollout/ep_rew_mean", hue="test_name", data=df
    )
    ax.set(title="reward over time", xlabel="timesteps", ylabel="reward")
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(f"{LOGS_DIR}/reward_plot.png")

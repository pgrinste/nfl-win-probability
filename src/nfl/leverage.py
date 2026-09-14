"""High-leverage play analysis: how much each play moved win probability.

For every play we score the pre-play situation, then build an approximate
post-play situation (updated score differential from the data; down &
distance advanced by the result; clock taken from the next play's actual
remaining time in the same game - a 32 s decrement for the final play)
and score that too. Leverage = |WP_after - WP_before|.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FEATURES = ["score_differential", "down", "ydstogo", "time_frac", "field_pos_frac",
            "elo_diff"]


def post_state(row, time_frac_post):
    """Approximate the situation after this play completes."""
    sd = row["score_differential_post"] if pd.notna(row.get("score_differential_post")) else row["score_differential"]
    if bool(row.get("first_down", False)):
        down, ydstogo = 1.0, 10.0
    else:
        gained = max(0.0, float(row.get("yards_gained") or 0.0))
        down = min(4.0, row["down"] + 1)
        ydstogo = max(1.0, row["ydstogo"] - gained)
    return {
        "score_differential": sd,
        "down": down,
        "ydstogo": ydstogo,
        "time_frac": time_frac_post,            # next play's actual clock (v2)
        "field_pos_frac": row["field_pos_frac"],
        "elo_diff": row["elo_diff"],             # team strength doesn't change mid-play
    }


def compute_leverage(df, model):
    pre = df[FEATURES].to_numpy()
    wp_before = model.predict_proba(pre)[:, 1]

    # post-play clock: the next play's actual remaining time in this game
    # (rows are in play order); final plays fall back to a 32 s decrement.
    nxt = df.groupby("game_id")["time_frac"].shift(-1)
    post_time = nxt.fillna((df["time_frac"] - 32.0 / 3600.0).clip(lower=0.0))

    post_rows = [post_state(r, t) for (_, r), t in zip(df.iterrows(), post_time)]
    post = pd.DataFrame(post_rows)[FEATURES].to_numpy()
    wp_after = model.predict_proba(post)[:, 1]

    out = df[["game_id", "season", "week", "home_team", "away_team", "posteam", "defteam", "desc"]].copy()
    out["wp_before"] = wp_before
    out["wp_after"] = wp_after
    out["leverage"] = np.abs(wp_after - wp_before)
    return out


def plot_top_plays(lev, path, n=15):
    top = lev.nlargest(n, "leverage").iloc[::-1]
    labels = []
    for _, r in top.iterrows():
        d = str(r["desc"])[:48].replace("\n", " ")
        labels.append(f"{int(r['season'])} {r['posteam']} vs {r.get('defteam', '?')}  WP {r['wp_before']:.2f}->{r['wp_after']:.2f}\n{d}")
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(range(len(top)), top["leverage"], color="steelblue")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("|change in win probability|")
    ax.set_title("Highest-leverage plays (tuned model, all seasons)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(df, baseline, tuned, path):
    test = df[df["season"] > 2023]
    X, y = test[FEATURES], test["posteam_won"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model, color in (("logistic baseline", baseline, "gray"), ("tuned XGBoost", tuned, "crimson")):
        p = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, p)
        from sklearn.metrics import auc

        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"Win-probability model ROC (test: seasons >2023, n={len(test)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_game_flow(lev, path, n_games=5):
    """Interactive game-flow charts for the highest-leverage games (plotly HTML).

    Win probability is shown from the home team's perspective across every
    play; the plays that moved it most are marked with their description.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    top_games = lev.groupby("game_id")["leverage"].sum().nlargest(n_games)
    seasons = lev.groupby("game_id")["season"].first()
    fig = make_subplots(
        rows=n_games, cols=1,
        subplot_titles=[f"{int(seasons[gid])} {gid}" for gid in top_games.index],
        vertical_spacing=0.12,
    )
    for row_i, gid in enumerate(top_games.index, start=1):
        g = lev[lev["game_id"] == gid]
        home_wp = np.where(g["home_team"] == g["posteam"], g["wp_before"], 1.0 - g["wp_before"])
        fig.add_trace(
            go.Scatter(x=np.arange(len(g)), y=home_wp, mode="lines",
                       line=dict(color="#4c72b0", width=1.5),
                       name=f"{g['home_team'].iloc[0]} win probability",
                       showlegend=(row_i == 1), hovertext=[f"play {i + 1}" for i in range(len(g))],
                       hoverinfo="x+y+text"),
            row=row_i, col=1,
        )
        hot = g.nlargest(5, "leverage")
        idx_in_g = [list(g.index).index(i) for i in hot.index]
        fig.add_trace(
            go.Scatter(x=[np.arange(len(g))[k] for k in idx_in_g],
                       y=[home_wp[k] for k in idx_in_g], mode="markers",
                       marker=dict(size=11, color=np.where(hot["wp_after"] > hot["wp_before"],
                                                            "#2ecc71", "#e74c3c")),
                       name="high-leverage plays",
                       showlegend=(row_i == 1),
                       text=[f"WP {r['wp_before']:.2f} -> {r['wp_after']:.2f}<br>"
                             f"{str(r['desc'])[:80].replace(chr(10), ' ')}"
                             for _, r in hot.iterrows()],
                       hoverinfo="x+y+text"),
            row=row_i, col=1,
        )
        fig.update_yaxes(title_text=None, range=[0, 1], row=row_i, col=1)
    fig.update_layout(height=320 * n_games + 140, title="Game flow: home-team win probability with high-leverage plays marked",
                      template="plotly_dark", legend=dict(orientation="h", y=1.02))
    fig.write_html(path)


def main():
    import joblib

    df = pd.read_parquet(os.path.join(REPO_ROOT, "data", "curated", "nfl_plays.parquet"))
    tuned = joblib.load(os.path.join(REPO_ROOT, "artifacts", "tuned.joblib"))
    baseline = joblib.load(os.path.join(REPO_ROOT, "artifacts", "baseline.joblib"))

    lev = compute_leverage(df, tuned)
    os.makedirs(os.path.join(REPO_ROOT, "output"), exist_ok=True)
    plot_top_plays(lev, os.path.join(REPO_ROOT, "output", "top_leverage_plays.png"))
    plot_roc(df, baseline, tuned, os.path.join(REPO_ROOT, "output", "model_roc.png"))
    plot_game_flow(lev, os.path.join(REPO_ROOT, "output", "game_flow_interactive.html"))

    top = lev.nlargest(10, "leverage")
    print("top-10 leverage plays:")
    for _, r in top.iterrows():
        d = str(r["desc"])[:70].replace("\n", " ")
        print(f"  {int(r['season'])} W{int(r['week'])} {r['posteam']} vs {r.get('defteam','?')}: WP {r['wp_before']:.3f}->{r['wp_after']:.3f} | {d}")


if __name__ == "__main__":
    main()

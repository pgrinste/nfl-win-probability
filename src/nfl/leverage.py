"""High-leverage play analysis: how much each play moved win probability.

For every play we score the pre-play situation, then build an approximate
post-play situation (updated score differential from the data; down &
distance advanced by the result; clock held constant - a documented v1
simplification) and score that too. Leverage = |WP_after - WP_before|.
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
FEATURES = ["score_differential", "down", "ydstogo", "time_frac", "field_pos_frac"]


def post_state(row):
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
        "time_frac": row["time_frac"],          # clock held constant (v1)
        "field_pos_frac": row["field_pos_frac"],
    }


def compute_leverage(df, model):
    pre = df[FEATURES].to_numpy()
    wp_before = model.predict_proba(pre)[:, 1]

    post_rows = [post_state(r) for _, r in df.iterrows()]
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


def main():
    import joblib

    df = pd.read_parquet(os.path.join(REPO_ROOT, "data", "curated", "nfl_plays.parquet"))
    tuned = joblib.load(os.path.join(REPO_ROOT, "artifacts", "tuned.joblib"))
    baseline = joblib.load(os.path.join(REPO_ROOT, "artifacts", "baseline.joblib"))

    lev = compute_leverage(df, tuned)
    os.makedirs(os.path.join(REPO_ROOT, "output"), exist_ok=True)
    plot_top_plays(lev, os.path.join(REPO_ROOT, "output", "top_leverage_plays.png"))
    plot_roc(df, baseline, tuned, os.path.join(REPO_ROOT, "output", "model_roc.png"))

    top = lev.nlargest(10, "leverage")
    print("top-10 leverage plays:")
    for _, r in top.iterrows():
        d = str(r["desc"])[:70].replace("\n", " ")
        print(f"  {int(r['season'])} W{int(r['week'])} {r['posteam']} vs {r.get('defteam','?')}: WP {r['wp_before']:.3f}->{r['wp_after']:.3f} | {d}")


if __name__ == "__main__":
    main()

"""Train win-probability models with a season-based split.

- Baseline: logistic regression (weights reproducible from config/default.yaml).
- Tuned: XGBoost, hyperparameters selected on the 2023 validation season and
  written to config/tuned.yaml (gitignored in the public repo). Final model is
  trained on <=2023 and evaluated on unseen seasons >=2024.

Note: LightGBM crashed with an access violation on some Windows/Intel builds;
XGBoost is used instead - same algorithm family, different binary.
"""

import os

import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CONFIG_DIR = os.path.join(REPO_ROOT, "config")
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")

FEATURES = ["score_differential", "down", "ydstogo", "time_frac", "field_pos_frac"]
TUNE_SEASON = 2023          # validation season for hyperparameter selection
TEST_FROM = TUNE_SEASON + 1  # unseen seasons for the final report


def _xgb(**over):
    base = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=7,
        n_jobs=4,
    )
    base.update(over)
    return XGBClassifier(**base)


def tune_and_fit(df):
    train = df[df["season"] < TUNE_SEASON]
    val = df[df["season"] == TUNE_SEASON]
    test = df[df["season"] >= TEST_FROM]

    # --- baseline logistic (fit on everything pre-test for a fair comparison) ---
    base_data = df[df["season"] < TEST_FROM]
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    baseline.fit(base_data[FEATURES], base_data["posteam_won"])

    # --- tune max_depth on the validation season ---
    best = None
    for depth in (4, 6, 8):
        m = _xgb(max_depth=depth)
        m.fit(
            train[FEATURES], train["posteam_won"],
            eval_set=[(val[FEATURES], val["posteam_won"])],
            verbose=False,
        )
        auc_val = roc_auc_score(val["posteam_won"], m.predict_proba(val[FEATURES])[:, 1])
        print(f"max_depth={depth}: val AUC {auc_val:.4f}")
        if best is None or auc_val > best[0]:
            best = (auc_val, depth)

    _, best_depth = best
    tuned = _xgb(max_depth=best_depth)
    pre_test = df[df["season"] < TEST_FROM]
    tuned.fit(pre_test[FEATURES], pre_test["posteam_won"])

    # --- report on unseen seasons ---
    results = {}
    for name, model in (("logistic_baseline", baseline), ("xgboost_tuned", tuned)):
        p = model.predict_proba(test[FEATURES])[:, 1]
        results[name] = {
            "auc_test": roc_auc_score(test["posteam_won"], p),
            "n_train": len(pre_test),
            "n_test": len(test),
        }

    # --- persist tuned params + models (artifacts are gitignored) ---
    with open(os.path.join(CONFIG_DIR, "tuned.yaml"), "w") as f:
        yaml.safe_dump(
            {
                "model": "xgboost_tuned",
                "n_estimators": 400,
                "learning_rate": 0.05,
                "max_depth": best_depth,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "note": f"selected on {TUNE_SEASON} validation AUC; evaluated on seasons >={TEST_FROM}",
            },
            f,
        )
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    import joblib

    joblib.dump(baseline, os.path.join(ARTIFACTS_DIR, "baseline.joblib"))
    joblib.dump(tuned, os.path.join(ARTIFACTS_DIR, "tuned.joblib"))
    return baseline, tuned, results


def main():
    df = pd.read_parquet(os.path.join(REPO_ROOT, "data", "curated", "nfl_plays.parquet"))
    print(f"dataset: {len(df)} plays, seasons {df['season'].min()}-{df['season'].max()}")
    baseline, tuned, results = tune_and_fit(df)
    for name, r in results.items():
        print(f"{name}: AUC(test >= {TEST_FROM})={r['auc_test']:.4f}  train={r['n_train']} test={r['n_test']}")


if __name__ == "__main__":
    main()

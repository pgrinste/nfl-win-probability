"""Fetch and clean NFL play-by-play data.

Source: nfl-data-py (Python port of nflfastR/nflreadr), which pulls the
standardized play-by-play files from AWS. One season downloads in a few
seconds; we cache raw files under data/raw/ so re-runs are free.
"""

import os

import pandas as pd

import nfl_data_py as ndp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
CURATED_DIR = os.path.join(REPO_ROOT, "data", "curated")

KEEP_COLS = [
    "game_id", "season", "week", "home_team", "away_team",
    "posteam", "defteam", "down", "ydstogo", "yardline_100",
    "game_seconds_remaining", "score_differential", "score_differential_post",
    "yards_gained", "play_type", "first_down", "result",
    "passer", "home_coach", "away_coach", "desc",
]


def fetch_seasons(years, cache=True):
    """Download play-by-play for the given seasons (cached to data/raw)."""
    os.makedirs(RAW_DIR, exist_ok=True)
    frames = []
    for year in years:
        path = os.path.join(RAW_DIR, f"pbp_{year}.parquet")
        if cache and os.path.exists(path):
            df = pd.read_parquet(path)
        else:
            df = ndp.import_pbp_data(years=[year])
            if cache:
                df.to_parquet(path, index=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _game_finals(raw):
    """Final score per game (last row by order_sequence carries the totals)."""
    last = raw.sort_values("order_sequence").groupby("game_id", as_index=False).tail(1)
    finals = {}
    for _, r in last.iterrows():
        h, a = float(r["total_home_score"]), float(r["total_away_score"])
        winner = r["home_team"] if h > a else (r["away_team"] if a > h else None)
        finals[r["game_id"]] = dict(
            season=r["season"], week=r["week"],
            home=r["home_team"], away=r["away_team"],
            score_home=h, score_away=a, winner=winner,
        )
    return finals


def _game_winners(raw):
    """Final score per game -> winning team."""
    return {g: f["winner"] for g, f in _game_finals(raw).items() if f["winner"]}


def _elo_before_games(finals):
    """Each team's ELO rating before each game, from this dataset alone.

    Self-contained (no external ratings feed): initial 1500, K = 20 + 4 per full
    7-point margin tier (capped at 36), standard logistic expectation. Games are
    processed in chronological order so ratings accumulate across the era.
    """
    elo = {}

    def get(team):
        if team not in elo:
            elo[team] = 1500.0
        return elo[team]

    before = {}
    ordered = sorted(finals.items(), key=lambda kv: (kv[1]["season"], kv[1]["week"], kv[0]))
    for gid, g in ordered:
        h, a = g["home"], g["away"]
        eh, ea = get(h), get(a)
        before[gid] = {h: eh, a: ea}
        if g["winner"] is None:
            continue  # OT tie: no update (rare)
        margin = abs(g["score_home"] - g["score_away"])
        k = min(36.0, 20.0 + 4.0 * int(margin // 7))
        exp_h = 1.0 / (1.0 + 10 ** ((ea - eh) / 400.0))
        if g["winner"] == h:
            elo[h] = eh + k * (1.0 - exp_h)
            elo[a] = ea - k * (1.0 - exp_h)
        else:
            elo[h] = eh - k * exp_h
            elo[a] = ea + k * exp_h
    return before


def build_dataset(raw):
    """Slim the raw table to modelling columns and add a win label.

    Label: 1 if the team on offense at this play won the game, else 0.
    (Ties/OT are resolved by final score; rare.)
    """
    winners = _game_winners(raw)
    df = raw[KEEP_COLS].copy()
    df["posteam_won"] = [1 if winners.get(g) == p else 0 for g, p in zip(df["game_id"], df["posteam"])]

    # keep plays with a defined situation (drops kickoffs and odd rows)
    df = df.dropna(subset=["down", "ydstogo", "score_differential"])
    df = df[df["play_type"].isin(["pass", "rush", "field_goal_attempt", "extra_point_attempt", "two_point_conv_attempt"])]

    # derived features used by the models
    df["time_frac"] = (df["game_seconds_remaining"] / 3600.0).clip(0, 1)
    df["field_pos_frac"] = (df["yardline_100"].fillna(50.0) / 100.0).clip(0, 1)

    # team strength: ELO gap at game start (offense minus defense), see _elo_before_games
    elo_before = _elo_before_games(_game_finals(raw))
    df["elo_diff"] = [
        elo_before[g][p] - elo_before[g][d]
        for g, p, d in zip(df["game_id"], df["posteam"], df["defteam"])
    ]
    return df.reset_index(drop=True)


def save_curated(df):
    os.makedirs(CURATED_DIR, exist_ok=True)
    path = os.path.join(CURATED_DIR, "nfl_plays.parquet")
    df.to_parquet(path, index=False)
    return path

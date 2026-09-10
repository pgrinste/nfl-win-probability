import pandas as pd

from nfl.data import build_dataset


def _raw():
    rows = []
    # game 1: home wins 20-10; offense is home on both plays shown
    for i, (sd, sd_post) in enumerate([(0.0, 3.0), (3.0, 7.0)]):
        rows.append(
            {
                "game_id": "g1", "season": 2024, "week": 5,
                "home_team": "NE", "away_team": "BUF",
                "posteam": "NE", "defteam": "BUF",
                "down": 1 + i, "ydstogo": 10.0 - i * 3, "yardline_100": 25.0 + i * 10,
                "game_seconds_remaining": 3600 - i * 40,
                "score_differential": sd, "score_differential_post": sd_post,
                "yards_gained": 7.0, "play_type": "pass", "first_down": False,
                "result": "Cmp", "passer": "QB1", "home_coach": "C1", "away_coach": "C2",
                "desc": "pass complete for 7 yards",
                "order_sequence": i + 1,
                "total_home_score": 20.0 if i == 1 else 3.0,
                "total_away_score": 10.0 if i == 1 else 3.0,
            }
        )
    # game 2: away wins; offense is away
    rows.append(
        {
            "game_id": "g2", "season": 2024, "week": 5,
            "home_team": "NE", "away_team": "BUF",
            "posteam": "BUF", "defteam": "NE",
            "down": 1, "ydstogo": 10.0, "yardline_100": 40.0,
            "game_seconds_remaining": 1800,
            "score_differential": -3.0, "score_differential_post": -3.0,
            "yards_gained": 2.0, "play_type": "rush", "first_down": False,
            "result": "Rush", "passer": None, "home_coach": "C1", "away_coach": "C2",
            "desc": "run for 2 yards",
            "order_sequence": 1,
            "total_home_score": 7.0, "total_away_score": 14.0,
        }
    )
    return pd.DataFrame(rows)


def test_labels_follow_game_winner():
    df = build_dataset(_raw())
    g1 = df[df["game_id"] == "g1"]["posteam_won"].unique()
    g2 = df[df["game_id"] == "g2"]["posteam_won"].unique()
    assert list(g1) == [1]   # home on offense, home wins
    assert list(g2) == [1]   # away on offense, away wins


def test_derived_features():
    df = build_dataset(_raw())
    r0 = df[df["game_id"] == "g1"].iloc[0]
    assert abs(r0["time_frac"] - 1.0) < 1e-9
    assert abs(r0["field_pos_frac"] - 0.25) < 1e-9

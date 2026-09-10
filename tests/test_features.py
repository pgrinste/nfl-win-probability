from nfl.features import situation_features


def _play(**kw):
    base = {
        "score_home": 14,
        "score_away": 7,
        "possession": "home",
        "down": 2,
        "distance_to_first": 8,
        "time_remaining_s": 1800,
        "yard_line": 35.0,   # offense at its own 35 (yards from own goal)
    }
    base.update(kw)
    return base


def test_features_for_home_offense():
    f = situation_features(_play())
    assert f["score_diff"] == 7          # home leads by 7 and is on offense
    assert f["down"] == 2.0
    assert f["distance_to_first"] == 8.0
    assert abs(f["time_frac"] - 0.5) < 1e-9
    assert 0.0 <= f["field_pos_frac"] <= 1.0


def test_features_flip_for_away_offense():
    f = situation_features(_play(possession="away"))
    assert f["score_diff"] == -7         # away trails by 7 and is on offense


def test_time_clamped_at_zero():
    f = situation_features(_play(time_remaining_s=-5))
    assert f["time_frac"] == 0.0

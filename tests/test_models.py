import os

from nfl.features import situation_features
from nfl.models import WinProbModel

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")


def test_baseline_model_predicts_in_unit_interval():
    model = WinProbModel(CONFIG)
    f = situation_features(
        {
            "score_home": 21,
            "score_away": 7,
            "possession": "home",
            "down": 1,
            "distance_to_first": 10,
            "time_remaining_s": 2400,
            "yard_line": 65.0,
        }
    )
    p = model.predict(f)
    assert 0.0 < p < 1.0
    assert p > 0.7   # big lead, early down, deep in territory -> strong favorite


def test_baseline_model_penalizes_late_downs():
    model = WinProbModel(CONFIG)
    base = {
        "score_home": 14,
        "score_away": 14,
        "possession": "home",
        "time_remaining_s": 3600,
        "yard_line": 50.0,
    }
    p_first = model.predict(situation_features({**base, "down": 1, "distance_to_first": 10}))
    p_fourth = model.predict(situation_features({**base, "down": 4, "distance_to_first": 12}))
    assert p_first > p_fourth

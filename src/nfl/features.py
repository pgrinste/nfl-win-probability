"""Situation features for win-probability modelling.

A "play row" is a dict with the fields produced by the data pipeline
(see src/nfl/data.py once the fetch step lands). The feature builder here
is pure and unit-tested so it can be validated before real data arrives.
"""


def situation_features(play):
    """Reduce one play-by-play row to model features.

    All quantities are from the perspective of the team on offense.
    """
    score_diff = _score_diff_for_offense(play)
    return {
        "score_diff": score_diff,
        "down": float(play["down"]),
        "distance_to_first": float(play["distance_to_first"]),
        "time_frac": max(0.0, play["time_remaining_s"]) / 3600.0,
        "field_pos_frac": min(max(float(play["yard_line"]), 0.0), 100.0) / 100.0,
    }


def _score_diff_for_offense(play):
    """Lead (in points) of the team on offense at this moment."""
    if play.get("possession") == "home":
        return play["score_home"] - play["score_away"]
    return play["score_away"] - play["score_home"]

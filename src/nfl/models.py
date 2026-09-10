"""Config-driven win-probability models.

Two tiers (see README):
- logistic_baseline: weights ship in config/default.yaml, fully reproducible.
- gradient_boosting_tuned: hyperparameters live in config/tuned.yaml which is
  gitignored; the code structure is public but the tuned values are not.
"""

import math
import os

import yaml


class WinProbModel:
    def __init__(self, config_path="config/default.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.model_name = self.config["model"]
        if self.model_name == "logistic_baseline":
            self.weights = dict(self.config["weights"])
        elif self.model_name == "gradient_boosting_tuned":
            tuned_path = os.path.join(os.path.dirname(config_path), "tuned.yaml")
            if not os.path.exists(tuned_path):
                raise FileNotFoundError(
                    f"model '{self.model_name}' needs {tuned_path} (gitignored; see README)"
                )
            with open(tuned_path) as f:
                self.tuned = yaml.safe_load(f)
        else:
            raise ValueError(f"unknown model: {self.model_name}")

    def predict(self, features):
        """Win probability in [0, 1] for the team on offense."""
        if self.model_name == "logistic_baseline":
            z = self.weights["intercept"]
            for name in ("score_diff", "down", "distance_to_first", "time_frac", "field_pos_frac"):
                z += self.weights[name] * features[name]
            return 1.0 / (1.0 + math.exp(-z))
        raise NotImplementedError("tuned model lands with the training pipeline")

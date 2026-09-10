# nfl-win-probability

Win probability and high-leverage play analysis for the modern NFL era (2015-2025).

## What it does

1. Fetches play-by-play data (nflreadr / official sources) and cleans it into a documented parquet dataset — that cleaned dataset is also published on Kaggle.
2. Builds situation features per play: score differential, down & distance, clock, field position.
3. Fits a win-probability model (logistic baseline → gradient boosting).
4. Ranks plays by how much they moved win probability ("high-leverage" analysis) and looks at play-calling tendencies by coach and quarterback.

## Two-tier configuration (why the repo is licensed GPLv3)

The code is fully public so you can verify it works, but not everything in it is meant to be copied:

- **`config/default.yaml`** — ships with the repo. The logistic baseline model and its weights are completely reproducible from here.
- **`config/tuned.yaml`** — gitignored. The tuned gradient-boosting hyperparameters live here so a competitor can see the architecture without getting the magic numbers. Tuned config available on request.

The GPLv3 license means anyone who reuses this code commercially must open-source their derivative work. If that's not for you, ask before vendoring.

## Layout

```
config/         model configuration (default.yaml public, tuned.yaml local)
data/raw/       raw downloads (gitignored)
src/nfl/        package: features, models, pipeline (pipeline lands next)
tests/          unit tests — run with pytest
artifacts/      trained models (gitignored)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt pytest   # Windows; use bin/pip on Linux/macOS
set PYTHONPATH=src                                      # Windows (or export PYTHONPATH=src)
python -m pytest tests/
```

## Status

Early stage: features and the baseline model are in place and tested. The data pipeline, tuned model, and leverage charts land next.

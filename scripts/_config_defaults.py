"""Shared helper for CLI config-path defaults across the scripts/ entry
points.

config/*.yaml (the "real" configs -- model.yaml, features.yaml, etc.) are
gitignored on purpose, because they're where each user's local tuning
lives and shouldn't be fought over in version control. Only
config/*_example.yaml (plus config/universe.txt) is tracked, so a fresh
clone has no config/model.yaml etc. at all.

Rather than requiring a bootstrap step (copying every *_example.yaml to
its real name before anything will run) or pointing script defaults
permanently at the example files (which would mean either quietly using
generic settings forever, or the user's own tuning dirtying the tracked
example -- defeating the point of keeping a clean template), each
default here resolves at CLI-parse time:

    config/{name}.yaml           if it exists (your local config), else
    config/{name}_example.yaml   (the tracked template)

This means:
  - A fresh clone runs every script out of the box, no setup script needed.
  - config/*_example.yaml is never written to, so it stays a clean,
    reviewable template.
  - As soon as you `cp config/model_example.yaml config/model.yaml` and
    start editing it, every script picks up your local file automatically
    -- no flag needed, and it's already gitignored.
"""
from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path("config")


def resolve_config_default(name: str) -> str:
    """Return 'config/{name}.yaml' if that file exists, else
    'config/{name}_example.yaml'. Used as an argparse `default=`, so it's
    only ever a fallback -- passing --model-config (etc.) explicitly
    always wins."""
    real = CONFIG_DIR / f"{name}.yaml"
    if real.exists():
        return str(real)
    return str(CONFIG_DIR / f"{name}_example.yaml")

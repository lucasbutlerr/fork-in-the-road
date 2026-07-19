"""Generic YAML -> pydantic config loader, shared by every config file in
the project. New config files (scanning_criteria.yaml, labeling.yaml,
model.yaml, ...) just call load_config(path, TheirSchema) -- nothing here
needs to change as new schemas are added in schemas.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


class ConfigError(Exception):
    """Raised for a missing, unreadable, or schema-invalid config file.
    Wraps the underlying YAML/pydantic error together with the file path,
    so the exception message alone is enough to go fix the right file."""


def load_config(path: str | Path, schema: type[ModelT]) -> ModelT:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Malformed YAML in {path}: {exc}") from exc

    try:
        return schema.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config in {path}:\n{exc}") from exc

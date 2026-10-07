from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    required_sections = {"app", "model", "generation", "languages"}
    missing_sections = required_sections.difference(config.keys())

    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise ValueError(f"Missing configuration section(s): {missing}")

    return config

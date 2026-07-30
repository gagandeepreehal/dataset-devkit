from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from dataset_devkit.config import GlobalConfig, load_config


@pytest.fixture
def config_factory() -> Callable[[], GlobalConfig]:
    config_path = Path(__file__).parents[1] / "examples" / "dataset_config.json"
    return lambda: load_config(config_path)

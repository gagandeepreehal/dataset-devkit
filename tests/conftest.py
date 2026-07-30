from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from dataset_devkit.config import GlobalConfig, load_config
from dataset_devkit.features import SceneFeatures
from dataset_devkit.provenance import SourceFingerprint

FeatureFactory = Callable[..., SceneFeatures]


@pytest.fixture
def config_factory() -> Callable[[], GlobalConfig]:
    config_path = Path(__file__).parents[1] / "examples" / "dataset_config.json"
    return lambda: load_config(config_path)


@pytest.fixture
def feature_factory() -> FeatureFactory:
    source = SourceFingerprint(
        "https://example.blob.core.windows.net",
        "recordings",
        "mcap-h265/a.mcap",
        '"e"',
        1,
    )
    base = SceneFeatures(
        "scene", "scene", source, source.blob_path, (), ("moving",), "front", ("front",), 0,
        (0,), (), (), (0.0,), (), (True,), (), 10.0, 5.0, 0.5, 0.5, 0.5, 0.5, 0.5,
        0, 0, 0.0, (0.0,), (), 0.0, 0.0, 0.0, (), 1.0, 1.0, (), 0.0, 0.0,
    )

    def make(**changes: Any) -> SceneFeatures:
        return replace(base, **changes)

    return make

"""Canonical copy-paste scenario-rule templates."""

from __future__ import annotations

from dataset_devkit.config import ScenarioRuleConfig


def canonical_scenario_rules(
    *, quota: int, annotation_label: str = "category/replace-me"
) -> tuple[ScenarioRuleConfig, ...]:
    """Return validated canonical rules; replace quotas and annotation label as needed."""
    return (
        ScenarioRuleConfig(name="Straight", quota=quota, required_all_tags=["straight", "moving"]),
        ScenarioRuleConfig(name="Stopping", quota=quota, required_all_tags=["stopping"]),
        ScenarioRuleConfig(name="Left Turn", quota=quota, required_all_tags=["left_turn"]),
        ScenarioRuleConfig(name="Right Turn", quota=quota, required_all_tags=["right_turn"]),
        ScenarioRuleConfig(
            name="Left Curvature", quota=quota, required_all_tags=["left_curvature"]
        ),
        ScenarioRuleConfig(
            name="Right Curvature", quota=quota, required_all_tags=["right_curvature"]
        ),
        ScenarioRuleConfig(
            name="Annotation Category", quota=quota, required_all_labels=[annotation_label]
        ),
    )

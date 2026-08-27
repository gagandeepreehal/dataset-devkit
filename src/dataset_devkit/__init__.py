"""Public package interface for dataset-devkit."""

from dataset_devkit.config import validate_config_schema_and_runtime
from dataset_devkit.dataset import Dataset, DatasetFormatError
from dataset_devkit.export import ExportEvidence, ExportResult, export_dataset
from dataset_devkit.services import (
    BuildResult,
    BuildRuntime,
    InspectionSummary,
    build_dataset,
    inspect_dataset,
)
from dataset_devkit.split import (
    split_selected_scenes,
    validate_scene_split,
    write_split_extension,
)
from dataset_devkit.validation import (
    DatasetValidationError,
    ValidationFinding,
    ValidationReport,
    finalize_dataset,
    validate_dataset,
)

__all__ = [
    "Dataset",
    "DatasetFormatError",
    "ExportEvidence",
    "ExportResult",
    "BuildResult",
    "BuildRuntime",
    "DatasetValidationError",
    "InspectionSummary",
    "ValidationFinding",
    "ValidationReport",
    "__version__",
    "split_selected_scenes",
    "export_dataset",
    "build_dataset",
    "finalize_dataset",
    "inspect_dataset",
    "validate_config_schema_and_runtime",
    "validate_scene_split",
    "validate_dataset",
    "write_split_extension",
]
__version__ = "0.1.0"

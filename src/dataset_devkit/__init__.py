"""Public package interface for dataset-devkit."""

from dataset_devkit.config import validate_config_schema_and_runtime
from dataset_devkit.dataset import Dataset

__all__ = ["Dataset", "__version__", "validate_config_schema_and_runtime"]
__version__ = "0.1.0"

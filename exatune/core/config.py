"""
Configuration management for ExaTune experiments using Pydantic.

This module provides configuration classes for defining and validating
experiment specifications including model parameters, hyperparameter spaces,
SLURM settings, and evaluation metrics.
"""

from pathlib import Path, PosixPath, WindowsPath
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator


#Made change for error 1 on Feb 12
# This ensures Path objects are converted to strings when saving to YAML.
# Without this, yaml.dump() creates Python-specific tags like:
#   !!python/object/apply:pathlib.PosixPath
# which yaml.safe_load() cannot deserialize, causing the worker to crash.
def _path_representer(dumper: yaml.Dumper, data: Path) -> yaml.Node:
    """Convert Path objects to strings for YAML serialization."""
    return dumper.represent_str(str(data))


# Register representers for all Path types
yaml.add_representer(Path, _path_representer)
yaml.add_representer(PosixPath, _path_representer)
yaml.add_representer(WindowsPath, _path_representer)

class HyperparameterSpec(BaseModel):
    """Specification for a single hyperparameter's search space."""

    values: Optional[List[Any]] = Field(
        None, description="Discrete values to search (for categorical or explicit list)"
    )
    min: Optional[float] = Field(None, description="Minimum value for continuous range")
    max: Optional[float] = Field(None, description="Maximum value for continuous range")
    step: Optional[float] = Field(None, description="Step size for continuous range")
    scale: Literal["linear", "log"] = Field("linear", description="Scale type: 'linear' or 'log'")
    type: Literal["int", "float", "categorical"] = Field("float", description="Parameter type")

    @field_validator("values", "min", "max")
    @classmethod
    def check_specification(cls, v: Any, info: Any) -> Any:
        """Validate that either values or min/max are provided."""
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate that configuration is complete after initialization."""
        if self.values is None and (self.min is None or self.max is None):
            raise ValueError("Either 'values' or both 'min' and 'max' must be specified")
        if self.values is not None and (self.min is not None or self.max is not None):
            raise ValueError("Cannot specify both 'values' and 'min/max'")


class ModelConfig(BaseModel):
    """Configuration for the machine learning model."""

    type: Literal["sklearn", "xgboost"] = Field(
        ..., description="Model framework: 'sklearn' or 'xgboost'"
    )
    class_name: str = Field(
        ..., alias="class", description="Model class name (e.g., 'RandomForestClassifier')"
    )
    task: Literal["classification", "regression"] = Field("classification", description="Task type")
    fixed_params: Dict[str, Any] = Field(
        default_factory=dict, description="Fixed hyperparameters not being tuned"
    )

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class DatasetConfig(BaseModel):
    """Configuration for the dataset."""

    name: Optional[str] = Field(None, description="Dataset name (for built-in datasets)")
    path: Optional[Path] = Field(None, description="Path to custom dataset")
    target_column: Optional[str] = Field(
        None, description="Target column name (for custom datasets)"
    )
    test_size: float = Field(0.2, ge=0.0, le=1.0, description="Test set proportion")
    validation_size: float = Field(0.0, ge=0.0, le=1.0, description="Validation set proportion")
    random_state: Optional[int] = Field(42, description="Random seed for reproducibility")
    stratify: bool = Field(True, description="Whether to stratify splits")

    @field_validator("path")
    @classmethod
    def check_dataset_specified(cls, v: Optional[Path], info: Any) -> Optional[Path]:
        """Validate that either name or path is provided."""
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate that either name or path is specified."""
        if self.name is None and self.path is None:
            raise ValueError("Either 'name' or 'path' must be specified for dataset")


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""

    cv_folds: int = Field(5, ge=2, description="Number of cross-validation folds")
    cv_strategy: Literal["kfold", "stratified", "group"] = Field(
        "stratified", description="Cross-validation strategy"
    )
    scoring: str = Field("accuracy", description="Primary scoring metric")
    additional_metrics: List[str] = Field(
        default_factory=list, description="Additional metrics to compute"
    )
    n_jobs: int = Field(-1, description="Number of parallel jobs for CV (-1 = all cores)")


class SlurmConfig(BaseModel):
    """Configuration for SLURM job submission."""

    partition: Optional[str] = Field(None, description="SLURM partition name")
    time: str = Field("01:00:00", description="Time limit (HH:MM:SS)")
    memory: str = Field("4G", description="Memory per job")
    cpus_per_task: int = Field(1, ge=1, description="CPUs per task")
    nodes: int = Field(1, ge=1, description="Number of nodes")
    account: Optional[str] = Field(None, description="SLURM account")
    qos: Optional[str] = Field(None, description="Quality of service")
    email: Optional[str] = Field(None, description="Email for notifications")
    email_type: Literal["NONE", "BEGIN", "END", "FAIL", "ALL"] = Field(
        "FAIL", description="Email notification type"
    )
    array_size: Optional[int] = Field(
        None, description="Maximum concurrent array jobs (None = unlimited)"
    )
    modules: List[str] = Field(
        default_factory=list,
        description="Environment modules to load before running jobs (e.g., ['python/3.10', 'scipy-stack/2023b'])",
    )
    python_environment: Optional[str] = Field(
        None, description="Path to Python virtual environment to activate (e.g., '~/exatune_env')"
    )
    additional_directives: Dict[str, str] = Field(
        default_factory=dict, description="Additional SLURM directives"
    )


class ExperimentConfig(BaseModel):
    """Configuration for an ExaTune experiment."""

    name: str = Field(..., description="Experiment name")
    output_dir: Path = Field(Path("./results"), description="Directory for results and outputs")
    description: Optional[str] = Field(None, description="Experiment description")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    random_seed: Optional[int] = Field(42, description="Global random seed")
    save_models: bool = Field(False, description="Whether to save trained models")
    checkpoint_interval: int = Field(100, ge=1, description="Save checkpoint every N jobs")


class ExaTuneConfig(BaseModel):
    """Complete configuration for an ExaTune hyperparameter search experiment."""

    experiment: ExperimentConfig
    model: ModelConfig
    hyperparameters: Dict[str, Union[List[Any], HyperparameterSpec]] = Field(
        ..., description="Hyperparameter search space"
    )
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    slurm: SlurmConfig

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExaTuneConfig":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            ExaTuneConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExaTuneConfig":
        """
        Create configuration from a dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            ExaTuneConfig instance
        """
        return cls(**config_dict)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.

        Args:
            path: Path where to save the configuration
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="python"), f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.

        Returns:
            Configuration as dictionary
        """
        return self.model_dump(mode="python")

    def get_hyperparameter_specs(self) -> Dict[str, HyperparameterSpec]:
        """
        Get normalized hyperparameter specifications.

        Converts simple lists to HyperparameterSpec objects.

        Returns:
            Dictionary mapping parameter names to specifications
        """
        specs: Dict[str, HyperparameterSpec] = {}

        for name, value in self.hyperparameters.items():
            if isinstance(value, HyperparameterSpec):
                specs[name] = value
            elif isinstance(value, list):
                # Convert simple list to HyperparameterSpec
                specs[name] = HyperparameterSpec(values=value, type=self._infer_type(value))
            else:
                raise ValueError(f"Invalid hyperparameter specification for '{name}': {value}")

        return specs

    @staticmethod
    def _infer_type(values: List[Any]) -> Literal["int", "float", "categorical"]:
        """
        Infer the parameter type from a list of values.

        Args:
            values: List of parameter values

        Returns:
            Inferred type: 'int', 'float', or 'categorical'
        """
        if not values:
            return "categorical"

        # Check if all values are integers
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            return "int"

        # Check if all values are numeric
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            return "float"

        # Otherwise, treat as categorical
        return "categorical"


def load_config(path: Union[str, Path]) -> ExaTuneConfig:
    """
    Convenience function to load a configuration file.

    Args:
        path: Path to configuration file (YAML or JSON)

    Returns:
        ExaTuneConfig instance
    """
    return ExaTuneConfig.from_yaml(path)


"""
Hyperparameter grid generation for exhaustive search.

This module provides functionality to generate all possible combinations
of hyperparameters from configuration specifications.
"""

import hashlib
import itertools
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np

from exatune.core.config import HyperparameterSpec


class GridGenerator:
    """
    Generate exhaustive hyperparameter grid from specifications.

    This class takes hyperparameter specifications and generates all
    possible combinations for exhaustive search.
    """

    def __init__(self, hyperparameter_specs: Dict[str, HyperparameterSpec]):
        """
        Initialize the grid generator.

        Args:
            hyperparameter_specs: Dictionary mapping parameter names to specifications
        """
        self.specs = hyperparameter_specs
        self.param_names = list(hyperparameter_specs.keys())
        self._grid: Optional[List[Dict[str, Any]]] = None

    def generate_values(self, spec: HyperparameterSpec) -> List[Any]:
        """
        Generate all values for a single hyperparameter.

        Args:
            spec: Hyperparameter specification

        Returns:
            List of all values to search for this parameter
        """
        if spec.values is not None:
            # Discrete values explicitly provided
            return spec.values

        # Generate values from min/max/step
        if spec.min is None or spec.max is None:
            raise ValueError("Either values or min/max must be specified")

        if spec.scale == "linear":
            values = self._generate_linear_range(
                spec.min, spec.max, spec.step, spec.type
            )
        elif spec.scale == "log":
            values = self._generate_log_range(
                spec.min, spec.max, spec.step, spec.type
            )
        else:
            raise ValueError(f"Unknown scale type: {spec.scale}")

        return values

    @staticmethod
    def _generate_linear_range(
        min_val: float,
        max_val: float,
        step: Optional[float],
        param_type: str
    ) -> List[Any]:
        """
        Generate linearly-spaced values.

        Args:
            min_val: Minimum value
            max_val: Maximum value
            step: Step size (if None, returns [min, max])
            param_type: Parameter type ('int', 'float', 'categorical')

        Returns:
            List of values
        """
        if step is None:
            values = [min_val, max_val]
        else:
            # Use np.arange for consistent floating point handling
            values = list(np.arange(min_val, max_val + step/2, step))

        if param_type == "int":
            values = [int(v) for v in values]

        return values

    @staticmethod
    def _generate_log_range(
        min_val: float,
        max_val: float,
        step: Optional[float],
        param_type: str
    ) -> List[Any]:
        """
        Generate logarithmically-spaced values.

        Args:
            min_val: Minimum value (must be > 0)
            max_val: Maximum value
            step: Number of points per decade (if None, returns [min, max])
            param_type: Parameter type

        Returns:
            List of values
        """
        if min_val <= 0:
            raise ValueError("Log scale requires min_val > 0")

        if step is None:
            values = [min_val, max_val]
        else:
            # step represents number of points per decade
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            n_points = int((log_max - log_min) * step) + 1
            values = list(np.logspace(log_min, log_max, n_points))

        if param_type == "int":
            # Remove duplicates after rounding
            values = sorted(list(set(int(np.round(v)) for v in values)))

        return values

    def estimate_grid_size(self) -> int:
        """
        Estimate the total number of hyperparameter combinations.

        Returns:
            Total number of combinations
        """
        total = 1
        for spec in self.specs.values():
            values = self.generate_values(spec)
            total *= len(values)
        return total

    def generate_grid(self) -> List[Dict[str, Any]]:
        """
        Generate the complete hyperparameter grid.

        Returns:
            List of dictionaries, each representing one hyperparameter configuration
        """
        if self._grid is not None:
            return self._grid

        # Generate values for each parameter
        param_values = {
            name: self.generate_values(spec)
            for name, spec in self.specs.items()
        }

        # Generate all combinations
        param_names = list(param_values.keys())
        value_lists = [param_values[name] for name in param_names]

        self._grid = [
            dict(zip(param_names, combination))
            for combination in itertools.product(*value_lists)
        ]

        return self._grid

    def generate_grid_iterator(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """
        Generate hyperparameter configurations one at a time (memory efficient).

        Yields:
            Tuple of (index, configuration dictionary)
        """
        # Generate values for each parameter
        param_values = {
            name: self.generate_values(spec)
            for name, spec in self.specs.items()
        }

        param_names = list(param_values.keys())
        value_lists = [param_values[name] for name in param_names]

        for idx, combination in enumerate(itertools.product(*value_lists)):
            yield idx, dict(zip(param_names, combination))

    def get_config_by_index(self, index: int) -> Dict[str, Any]:
        """
        Get a specific configuration by its index.

        Args:
            index: Configuration index (0-based)

        Returns:
            Hyperparameter configuration dictionary

        Raises:
            IndexError: If index is out of bounds
        """
        grid = self.generate_grid()
        if index < 0 or index >= len(grid):
            raise IndexError(
                f"Index {index} out of bounds for grid of size {len(grid)}"
            )
        return grid[index]

    @staticmethod
    def config_to_id(config: Dict[str, Any]) -> str:
        """
        Generate a unique ID for a hyperparameter configuration.

        Args:
            config: Hyperparameter configuration dictionary

        Returns:
            Unique hash string identifying this configuration
        """
        # Sort keys for consistent hashing
        config_str = str(sorted(config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the hyperparameter grid.

        Returns:
            Dictionary with grid statistics
        """
        param_summaries = {}
        for name, spec in self.specs.items():
            values = self.generate_values(spec)
            param_summaries[name] = {
                "n_values": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "type": spec.type,
                "scale": spec.scale,
            }

        return {
            "n_parameters": len(self.specs),
            "total_combinations": self.estimate_grid_size(),
            "parameters": param_summaries,
        }

    def print_summary(self) -> None:
        """Print a human-readable summary of the grid."""
        summary = self.get_summary()

        print("=" * 60)
        print("Hyperparameter Grid Summary")
        print("=" * 60)
        print(f"Number of parameters: {summary['n_parameters']}")
        print(f"Total combinations: {summary['total_combinations']:,}")
        print("\nParameter Details:")
        print("-" * 60)

        for name, info in summary["parameters"].items():
            print(f"\n{name}:")
            print(f"  Type: {info['type']}")
            print(f"  Scale: {info['scale']}")
            print(f"  Number of values: {info['n_values']}")
            if info['min'] is not None and info['max'] is not None:
                print(f"  Range: {info['min']} to {info['max']}")

        print("=" * 60)


def generate_hyperparameter_grid(
    hyperparameter_specs: Dict[str, HyperparameterSpec]
) -> List[Dict[str, Any]]:
    """
    Convenience function to generate a hyperparameter grid.

    Args:
        hyperparameter_specs: Dictionary mapping parameter names to specifications

    Returns:
        List of all hyperparameter configurations
    """
    generator = GridGenerator(hyperparameter_specs)
    return generator.generate_grid()

"""Tests for hyperparameter grid generation."""

import pytest
from exatune.core.grid_generator import GridGenerator
from exatune.core.config import HyperparameterSpec


def test_grid_generator_discrete_values():
    """Test grid generation with discrete values."""
    specs = {
        "param1": HyperparameterSpec(values=[1, 2, 3], type="int"),
        "param2": HyperparameterSpec(values=["a", "b"], type="categorical"),
    }

    generator = GridGenerator(specs)
    grid = generator.generate_grid()

    assert len(grid) == 6  # 3 * 2 = 6 combinations
    assert all(isinstance(config, dict) for config in grid)
    assert all("param1" in config and "param2" in config for config in grid)


def test_grid_generator_estimate_size(sample_hyperparameter_specs):
    """Test grid size estimation."""
    generator = GridGenerator(sample_hyperparameter_specs)
    estimated_size = generator.estimate_grid_size()

    assert estimated_size == 3 * 3 * 2  # 18 combinations
    assert estimated_size == len(generator.generate_grid())


def test_grid_generator_iterator(sample_hyperparameter_specs):
    """Test grid iterator for memory efficiency."""
    generator = GridGenerator(sample_hyperparameter_specs)

    configs = list(generator.generate_grid_iterator())

    assert len(configs) == 18
    assert all(isinstance(idx, int) and isinstance(config, dict) for idx, config in configs)


def test_grid_generator_summary(sample_hyperparameter_specs):
    """Test grid summary generation."""
    generator = GridGenerator(sample_hyperparameter_specs)
    summary = generator.get_summary()

    assert summary["n_parameters"] == 3
    assert summary["total_combinations"] == 18
    assert "parameters" in summary
    assert len(summary["parameters"]) == 3


def test_config_to_id():
    """Test configuration ID generation."""
    config1 = {"a": 1, "b": 2}
    config2 = {"b": 2, "a": 1}  # Same config, different order
    config3 = {"a": 1, "b": 3}  # Different config

    id1 = GridGenerator.config_to_id(config1)
    id2 = GridGenerator.config_to_id(config2)
    id3 = GridGenerator.config_to_id(config3)

    assert id1 == id2  # Same configs should have same ID
    assert id1 != id3  # Different configs should have different IDs
    assert len(id1) == 16  # ID should be 16 characters

"""
Base model wrapper interface for ExaTune.

This module defines the abstract base class that all model wrappers must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
from numpy.typing import NDArray


class BaseModelWrapper(ABC):
    """
    Abstract base class for model wrappers.

    All model wrappers (scikit-learn, XGBoost, etc.) must inherit from this
    class and implement its abstract methods.
    """

    def __init__(
        self,
        model_class: str,
        hyperparameters: Dict[str, Any],
        task: str = "classification",
        random_state: Optional[int] = None,
    ):
        """
        Initialize the model wrapper.

        Args:
            model_class: Name of the model class (e.g., 'RandomForestClassifier')
            hyperparameters: Dictionary of hyperparameters for the model
            task: Task type ('classification' or 'regression')
            random_state: Random seed for reproducibility
        """
        self.model_class = model_class
        self.hyperparameters = hyperparameters
        self.task = task
        self.random_state = random_state
        self.model: Optional[Any] = None
        self._is_fitted = False

    @abstractmethod
    def _create_model(self) -> Any:
        """
        Create and return the underlying model instance.

        Returns:
            Model instance with specified hyperparameters
        """
        pass

    @abstractmethod
    def fit(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        **kwargs: Any
    ) -> "BaseModelWrapper":
        """
        Fit the model on training data.

        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional arguments for fitting

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """
        Make predictions on new data.

        Args:
            X: Features to predict on

        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def predict_proba(self, X: NDArray[Any]) -> NDArray[Any]:
        """
        Predict class probabilities (classification only).

        Args:
            X: Features to predict on

        Returns:
            Class probabilities

        Raises:
            ValueError: If task is not classification or model doesn't support probabilities
        """
        pass

    def score(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        metric: str = "default"
    ) -> float:
        """
        Calculate a performance metric on the given data.

        Args:
            X: Features
            y: True targets
            metric: Metric name ('default' uses model's default score method)

        Returns:
            Score value
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before scoring")

        if metric == "default":
            return self.model.score(X, y)

        # Import metrics on demand
        from sklearn.metrics import get_scorer

        scorer = get_scorer(metric)
        return scorer(self.model, X, y)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to disk.

        Args:
            path: Path where to save the model
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "model_class": self.model_class,
            "hyperparameters": self.hyperparameters,
            "task": self.task,
            "random_state": self.random_state,
            "wrapper_class": self.__class__.__name__,
        }

        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseModelWrapper":
        """
        Load a saved model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded model wrapper instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        # Create wrapper instance
        wrapper = cls(
            model_class=model_data["model_class"],
            hyperparameters=model_data["hyperparameters"],
            task=model_data["task"],
            random_state=model_data["random_state"],
        )

        wrapper.model = model_data["model"]
        wrapper._is_fitted = True

        return wrapper

    def get_params(self) -> Dict[str, Any]:
        """
        Get the model's hyperparameters.

        Returns:
            Dictionary of hyperparameters
        """
        return self.hyperparameters.copy()

    def set_params(self, **params: Any) -> "BaseModelWrapper":
        """
        Set the model's hyperparameters.

        Args:
            **params: Hyperparameters to update

        Returns:
            Self for method chaining
        """
        self.hyperparameters.update(params)
        self.model = None  # Reset model
        self._is_fitted = False
        return self

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._is_fitted

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return (
            f"{self.__class__.__name__}("
            f"model_class='{self.model_class}', "
            f"task='{self.task}', "
            f"n_params={len(self.hyperparameters)})"
        )

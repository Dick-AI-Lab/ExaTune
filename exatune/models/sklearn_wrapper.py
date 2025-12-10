"""
Scikit-learn model wrapper for ExaTune.

This module provides a wrapper for scikit-learn estimators.
"""

import importlib
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from exatune.models.base import BaseModelWrapper


class SklearnModelWrapper(BaseModelWrapper):
    """
    Wrapper for scikit-learn estimators.

    Supports any scikit-learn classifier or regressor.
    """

    def __init__(
        self,
        model_class: str,
        hyperparameters: Dict[str, Any],
        task: str = "classification",
        random_state: Optional[int] = None,
    ):
        """
        Initialize the scikit-learn model wrapper.

        Args:
            model_class: Name of the sklearn model class (e.g., 'RandomForestClassifier')
            hyperparameters: Dictionary of hyperparameters
            task: Task type ('classification' or 'regression')
            random_state: Random seed for reproducibility
        """
        super().__init__(model_class, hyperparameters, task, random_state)
        self._model_module: Optional[str] = None

    def _get_model_class(self) -> type:
        """
        Dynamically import and return the model class.

        Returns:
            Model class

        Raises:
            ImportError: If model class cannot be imported
        """
        # Common sklearn module paths
        common_modules = [
            "sklearn.ensemble",
            "sklearn.linear_model",
            "sklearn.tree",
            "sklearn.svm",
            "sklearn.naive_bayes",
            "sklearn.neighbors",
            "sklearn.neural_network",
            "sklearn.discriminant_analysis",
            "sklearn.gaussian_process",
        ]

        # Try cached module first
        if self._model_module is not None:
            try:
                module = importlib.import_module(self._model_module)
                return getattr(module, self.model_class)
            except (ImportError, AttributeError):
                pass

        # Search common modules
        for module_path in common_modules:
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, self.model_class):
                    self._model_module = module_path
                    return getattr(module, self.model_class)
            except ImportError:
                continue

        raise ImportError(
            f"Could not find scikit-learn class '{self.model_class}'. "
            f"Searched in: {', '.join(common_modules)}"
        )

    def _create_model(self) -> Any:
        """
        Create and return the sklearn model instance.

        Returns:
            Sklearn estimator instance
        """
        model_cls = self._get_model_class()

        # Prepare parameters
        params = self.hyperparameters.copy()

        # Add random_state if the model supports it and it's not already set
        if self.random_state is not None and "random_state" not in params:
            # Check if model class accepts random_state
            import inspect
            sig = inspect.signature(model_cls.__init__)
            if "random_state" in sig.parameters:
                params["random_state"] = self.random_state

        return model_cls(**params)

    def fit(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        **kwargs: Any
    ) -> "SklearnModelWrapper":
        """
        Fit the sklearn model on training data.

        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional arguments passed to fit (e.g., sample_weight)

        Returns:
            Self for method chaining
        """
        if self.model is None:
            self.model = self._create_model()

        self.model.fit(X, y, **kwargs)
        self._is_fitted = True

        return self

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """
        Make predictions on new data.

        Args:
            X: Features to predict on

        Returns:
            Predictions

        Raises:
            ValueError: If model is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X: NDArray[Any]) -> NDArray[Any]:
        """
        Predict class probabilities (classification only).

        Args:
            X: Features to predict on

        Returns:
            Class probabilities

        Raises:
            ValueError: If model is not fitted or doesn't support predict_proba
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if self.task != "classification":
            raise ValueError(
                f"predict_proba is only available for classification tasks, "
                f"got task='{self.task}'"
            )

        if not hasattr(self.model, "predict_proba"):
            raise ValueError(
                f"Model {self.model_class} does not support predict_proba"
            )

        return self.model.predict_proba(X)

    def get_feature_importances(self) -> Optional[NDArray[Any]]:
        """
        Get feature importances if available.

        Returns:
            Feature importance array or None if not available
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted to get feature importances")

        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            # For linear models, use absolute coefficient values
            return np.abs(self.model.coef_).ravel()
        else:
            return None


def create_sklearn_model(
    model_class: str,
    hyperparameters: Dict[str, Any],
    task: str = "classification",
    random_state: Optional[int] = None,
) -> SklearnModelWrapper:
    """
    Convenience function to create a scikit-learn model wrapper.

    Args:
        model_class: Name of the sklearn model class
        hyperparameters: Dictionary of hyperparameters
        task: Task type ('classification' or 'regression')
        random_state: Random seed

    Returns:
        SklearnModelWrapper instance
    """
    return SklearnModelWrapper(
        model_class=model_class,
        hyperparameters=hyperparameters,
        task=task,
        random_state=random_state,
    )

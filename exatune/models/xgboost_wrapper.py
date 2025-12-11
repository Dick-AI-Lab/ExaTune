"""
XGBoost model wrapper for ExaTune.

This module provides a wrapper for XGBoost models.
"""

from typing import Any, Dict, Optional

from numpy.typing import NDArray

from exatune.models.base import BaseModelWrapper


class XGBoostModelWrapper(BaseModelWrapper):
    """
    Wrapper for XGBoost models.

    Supports XGBClassifier and XGBRegressor from the xgboost package.
    """

    def __init__(
        self,
        model_class: str,
        hyperparameters: Dict[str, Any],
        task: str = "classification",
        random_state: Optional[int] = None,
    ):
        """
        Initialize the XGBoost model wrapper.

        Args:
            model_class: Name of the XGBoost model class (e.g., 'XGBClassifier')
            hyperparameters: Dictionary of hyperparameters
            task: Task type ('classification' or 'regression')
            random_state: Random seed for reproducibility
        """
        super().__init__(model_class, hyperparameters, task, random_state)

    def _create_model(self) -> Any:
        """
        Create and return the XGBoost model instance.

        Returns:
            XGBoost model instance

        Raises:
            ImportError: If xgboost is not installed
            ValueError: If model_class is not recognized
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is not installed. Install it with: pip install xgboost")

        # Map model class name to XGBoost class
        model_map = {
            "XGBClassifier": xgb.XGBClassifier,
            "XGBRegressor": xgb.XGBRegressor,
            "XGBRFClassifier": xgb.XGBRFClassifier,
            "XGBRFRegressor": xgb.XGBRFRegressor,
        }

        if self.model_class not in model_map:
            raise ValueError(
                f"Unknown XGBoost model class: {self.model_class}. "
                f"Available: {list(model_map.keys())}"
            )

        model_cls = model_map[self.model_class]

        # Prepare parameters
        params = self.hyperparameters.copy()

        # Add random_state if not already set
        if self.random_state is not None and "random_state" not in params:
            params["random_state"] = self.random_state

        # Set evaluation metric based on task if not specified
        if "eval_metric" not in params:
            if self.task == "classification":
                params["eval_metric"] = "logloss"
            else:
                params["eval_metric"] = "rmse"

        return model_cls(**params)

    def fit(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        eval_set: Optional[list] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "XGBoostModelWrapper":
        """
        Fit the XGBoost model on training data.

        Args:
            X: Training features
            y: Training targets
            eval_set: List of (X, y) tuples for evaluation during training
            early_stopping_rounds: Stop if no improvement for this many rounds
            verbose: Whether to print training progress
            **kwargs: Additional arguments passed to fit

        Returns:
            Self for method chaining
        """
        if self.model is None:
            self.model = self._create_model()

        fit_params = kwargs.copy()

        if eval_set is not None:
            fit_params["eval_set"] = eval_set

        if early_stopping_rounds is not None:
            fit_params["early_stopping_rounds"] = early_stopping_rounds

        if not verbose:
            fit_params["verbose"] = False

        self.model.fit(X, y, **fit_params)
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
            ValueError: If model is not fitted or task is not classification
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if self.task != "classification":
            raise ValueError(
                f"predict_proba is only available for classification tasks, "
                f"got task='{self.task}'"
            )

        return self.model.predict_proba(X)

    def get_feature_importances(self) -> NDArray[Any]:
        """
        Get feature importances from the trained model.

        Returns:
            Feature importance array

        Raises:
            ValueError: If model is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted to get feature importances")

        return self.model.feature_importances_

    def get_booster(self) -> Any:
        """
        Get the underlying XGBoost Booster object.

        Returns:
            XGBoost Booster

        Raises:
            ValueError: If model is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted to get booster")

        return self.model.get_booster()

    def get_num_boosting_rounds(self) -> int:
        """
        Get the number of boosting rounds actually used.

        Returns:
            Number of boosting rounds

        Raises:
            ValueError: If model is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted")

        return self.model.get_booster().num_boosted_rounds()


def create_xgboost_model(
    model_class: str,
    hyperparameters: Dict[str, Any],
    task: str = "classification",
    random_state: Optional[int] = None,
) -> XGBoostModelWrapper:
    """
    Convenience function to create an XGBoost model wrapper.

    Args:
        model_class: Name of the XGBoost model class
        hyperparameters: Dictionary of hyperparameters
        task: Task type ('classification' or 'regression')
        random_state: Random seed

    Returns:
        XGBoostModelWrapper instance
    """
    return XGBoostModelWrapper(
        model_class=model_class,
        hyperparameters=hyperparameters,
        task=task,
        random_state=random_state,
    )

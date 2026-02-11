"""
Tests for XGBoost model wrapper (exatune.models.xgboost_wrapper).
"""

import pytest
import tempfile
from pathlib import Path

from exatune.models.xgboost_wrapper import XGBoostModelWrapper

# Check if xgboost is properly available (needs libomp on macOS)
try:
    import xgboost as xgb
    _xgb_available = True
except Exception:
    _xgb_available = False

xgb_required = pytest.mark.skipif(not _xgb_available, reason="XGBoost not available (missing libomp?)")


class TestXGBoostWrapperInitialization:
    """Tests for XGBoostModelWrapper initialization."""

    def test_create_classifier(self):
        """Test creating an XGBoost classifier."""
        wrapper = XGBoostModelWrapper(
            model_class="XGBClassifier",
            hyperparameters={"max_depth": 3, "learning_rate": 0.1},
            task="classification",
            random_state=42
        )
        assert wrapper.model is None  # Not trained yet
        assert wrapper.task == "classification"

    def test_create_regressor(self):
        """Test creating an XGBoost regressor."""
        wrapper = XGBoostModelWrapper(
            model_class="XGBRegressor",
            hyperparameters={"max_depth": 5, "learning_rate": 0.05},
            task="regression",
            random_state=42
        )
        assert wrapper.task == "regression"

    def test_default_hyperparameters(self):
        """Test wrapper with default hyperparameters."""
        wrapper = XGBoostModelWrapper(
            model_class="XGBClassifier",
            hyperparameters={},
            task="classification",
            random_state=42
        )
        assert wrapper.model is None


@xgb_required
class TestXGBoostWrapperTraining:
    """Tests for XGBoost model training."""

    @pytest.fixture
    def iris_data(self):
        """Load iris dataset for testing."""
        from sklearn.datasets import load_iris
        data = load_iris()
        return data.data, data.target

    def test_fit_predict_classifier(self, iris_data):
        """Test fitting and predicting with classifier."""
        X, y = iris_data
        wrapper = XGBoostModelWrapper(
            model_class="XGBClassifier",
            hyperparameters={"max_depth": 3, "n_estimators": 10},
            task="classification",
            random_state=42
        )

        wrapper.fit(X, y)
        predictions = wrapper.predict(X)

        assert len(predictions) == len(y)
        assert predictions.min() >= 0
        assert predictions.max() <= 2  # 3 classes

    def test_score_classifier(self, iris_data):
        """Test scoring classifier."""
        X, y = iris_data
        wrapper = XGBoostModelWrapper(
            model_class="XGBClassifier",
            hyperparameters={"max_depth": 5, "n_estimators": 50},
            task="classification",
            random_state=42
        )

        wrapper.fit(X, y)
        score = wrapper.score(X, y)

        assert 0.0 <= score <= 1.0
        assert score > 0.9  # Should get high accuracy on training data

    def test_feature_importance(self, iris_data):
        """Test feature importance extraction."""
        X, y = iris_data
        wrapper = XGBoostModelWrapper(
            model_class="XGBClassifier",
            hyperparameters={"max_depth": 3, "n_estimators": 10},
            task="classification",
            random_state=42
        )

        wrapper.fit(X, y)
        importances = wrapper.get_feature_importances()

        assert importances is not None
        assert len(importances) == X.shape[1]
        assert all(imp >= 0 for imp in importances)

    def test_early_stopping(self, iris_data):
        """Test training with early stopping."""
        from sklearn.model_selection import train_test_split

        X, y = iris_data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        wrapper = XGBoostModelWrapper(
            model_class="XGBClassifier",
            hyperparameters={
                "max_depth": 3,
                "n_estimators": 100,
            },
            task="classification",
            random_state=42
        )

        # Fit with validation set and early stopping
        wrapper.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)

        # Should have trained
        assert wrapper.model is not None


@xgb_required
class TestXGBoostWrapperRegression:
    """Tests for XGBoost regression."""

    @pytest.fixture
    def regression_data(self):
        """Create simple regression data."""
        import numpy as np
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] * 2 + X[:, 1] * -1 + np.random.randn(100) * 0.1
        return X, y

    def test_fit_predict_regressor(self, regression_data):
        """Test fitting and predicting with regressor."""
        X, y = regression_data
        wrapper = XGBoostModelWrapper(
            model_class="XGBRegressor",
            hyperparameters={"max_depth": 3, "n_estimators": 50},
            task="regression",
            random_state=42
        )

        wrapper.fit(X, y)
        predictions = wrapper.predict(X)

        assert len(predictions) == len(y)

    def test_score_regressor(self, regression_data):
        """Test scoring regressor (R^2 score)."""
        X, y = regression_data
        wrapper = XGBoostModelWrapper(
            model_class="XGBRegressor",
            hyperparameters={"max_depth": 5, "n_estimators": 50},
            task="regression",
            random_state=42
        )

        wrapper.fit(X, y)
        score = wrapper.score(X, y)

        # R^2 score should be high for training data
        assert score > 0.8


@xgb_required
class TestXGBoostWrapperSerialization:
    """Tests for XGBoost model serialization."""

    @pytest.fixture
    def trained_wrapper(self):
        """Create a trained XGBoost wrapper for testing."""
        from sklearn.datasets import load_iris
        data = load_iris()
        X, y = data.data, data.target

        wrapper = XGBoostModelWrapper(
            model_class="XGBClassifier",
            hyperparameters={"max_depth": 3, "n_estimators": 10},
            task="classification",
            random_state=42
        )
        wrapper.fit(X, y)
        return wrapper

    def test_save_load(self, trained_wrapper):
        """Test saving and loading XGBoost model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "xgboost_model.pkl"

            # Save
            trained_wrapper.save(save_path)
            assert save_path.exists()

            # Load
            loaded_wrapper = XGBoostModelWrapper.load(save_path)
            assert loaded_wrapper.model is not None

            # Test predictions are same
            from sklearn.datasets import load_iris
            X = load_iris().data
            pred1 = trained_wrapper.predict(X)
            pred2 = loaded_wrapper.predict(X)

            import numpy as np
            assert np.array_equal(pred1, pred2)


@xgb_required
class TestXGBoostWrapperAdvanced:
    """Tests for advanced XGBoost features."""

    @pytest.fixture
    def iris_data(self):
        """Load iris dataset for testing."""
        from sklearn.datasets import load_iris
        data = load_iris()
        return data.data, data.target

    def test_get_booster(self, iris_data):
        """Test accessing raw XGBoost booster."""
        X, y = iris_data
        wrapper = XGBoostModelWrapper(
            model_class="XGBClassifier",
            hyperparameters={"max_depth": 3, "n_estimators": 10},
            task="classification",
            random_state=42
        )

        wrapper.fit(X, y)
        booster = wrapper.get_booster()

        assert booster is not None
        # Check it's an XGBoost Booster object
        assert hasattr(booster, "predict")

    def test_custom_objective(self, iris_data):
        """Test XGBoost with custom parameters."""
        X, y = iris_data
        wrapper = XGBoostModelWrapper(
            model_class="XGBClassifier",
            hyperparameters={
                "max_depth": 3,
                "n_estimators": 10,
                "objective": "multi:softmax",
                "num_class": 3
            },
            task="classification",
            random_state=42
        )

        wrapper.fit(X, y)
        predictions = wrapper.predict(X)

        assert len(predictions) == len(y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

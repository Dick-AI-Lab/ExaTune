"""
Tests for sklearn model wrapper (exatune.models.sklearn_wrapper).
"""

import pytest
import tempfile
from pathlib import Path

from exatune.models.sklearn_wrapper import SklearnModelWrapper


class TestSklearnWrapperInitialization:
    """Tests for SklearnModelWrapper initialization."""

    def test_create_decision_tree(self):
        """Test creating a decision tree wrapper."""
        wrapper = SklearnModelWrapper(
            model_class="DecisionTreeClassifier",
            hyperparameters={"max_depth": 5},
            task="classification",
            random_state=42
        )
        assert wrapper.model_class == "DecisionTreeClassifier"
        assert wrapper.hyperparameters == {"max_depth": 5}
        assert wrapper.model is None  # Lazy creation

    def test_create_random_forest(self):
        """Test creating a random forest wrapper."""
        wrapper = SklearnModelWrapper(
            model_class="RandomForestClassifier",
            hyperparameters={"n_estimators": 100, "max_depth": 10},
            task="classification",
            random_state=42
        )
        assert wrapper.hyperparameters["n_estimators"] == 100
        assert wrapper.hyperparameters["max_depth"] == 10

    def test_invalid_class_name(self):
        """Test that invalid class name raises error on fit."""
        wrapper = SklearnModelWrapper(
            model_class="InvalidClass",
            hyperparameters={},
            task="classification",
            random_state=42
        )
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        with pytest.raises(ImportError):
            wrapper.fit(X, y)

    def test_invalid_hyperparameter(self):
        """Test that invalid hyperparameter raises error on fit."""
        wrapper = SklearnModelWrapper(
            model_class="DecisionTreeClassifier",
            hyperparameters={"invalid_param": 123},
            task="classification",
            random_state=42
        )
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        with pytest.raises(Exception):
            wrapper.fit(X, y)


class TestSklearnWrapperTraining:
    """Tests for model training."""

    @pytest.fixture
    def iris_data(self):
        """Load iris dataset for testing."""
        from sklearn.datasets import load_iris
        data = load_iris()
        return data.data, data.target

    def test_fit_predict(self, iris_data):
        """Test fitting and predicting."""
        X, y = iris_data
        wrapper = SklearnModelWrapper(
            model_class="DecisionTreeClassifier",
            hyperparameters={"max_depth": 3},
            task="classification",
            random_state=42
        )

        wrapper.fit(X, y)
        predictions = wrapper.predict(X)

        assert len(predictions) == len(y)
        assert predictions.shape == y.shape

    def test_score(self, iris_data):
        """Test scoring."""
        X, y = iris_data
        wrapper = SklearnModelWrapper(
            model_class="DecisionTreeClassifier",
            hyperparameters={"max_depth": 5},
            task="classification",
            random_state=42
        )

        wrapper.fit(X, y)
        score = wrapper.score(X, y)

        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should get decent accuracy on training data

    def test_feature_importance(self, iris_data):
        """Test feature importance extraction."""
        X, y = iris_data
        wrapper = SklearnModelWrapper(
            model_class="RandomForestClassifier",
            hyperparameters={"n_estimators": 10},
            task="classification",
            random_state=42
        )

        wrapper.fit(X, y)
        importances = wrapper.get_feature_importances()

        assert importances is not None
        assert len(importances) == X.shape[1]
        assert all(imp >= 0 for imp in importances)


class TestSklearnWrapperSerialization:
    """Tests for model serialization."""

    @pytest.fixture
    def trained_wrapper(self):
        """Create a trained wrapper for testing."""
        from sklearn.datasets import load_iris
        data = load_iris()
        X, y = data.data, data.target

        wrapper = SklearnModelWrapper(
            model_class="DecisionTreeClassifier",
            hyperparameters={"max_depth": 3},
            task="classification",
            random_state=42
        )
        wrapper.fit(X, y)
        return wrapper

    def test_save_load(self, trained_wrapper):
        """Test saving and loading model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pkl"

            # Save
            trained_wrapper.save(save_path)
            assert save_path.exists()

            # Load
            loaded_wrapper = SklearnModelWrapper.load(save_path)
            assert loaded_wrapper.model is not None

            # Test predictions are same
            from sklearn.datasets import load_iris
            X = load_iris().data
            pred1 = trained_wrapper.predict(X)
            pred2 = loaded_wrapper.predict(X)

            import numpy as np
            assert np.array_equal(pred1, pred2)


class TestSklearnWrapperRegression:
    """Tests for regression models."""

    @pytest.fixture
    def regression_data(self):
        """Create simple regression data."""
        import numpy as np
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] * 2 + X[:, 1] * -1 + np.random.randn(100) * 0.1
        return X, y

    def test_linear_regression(self, regression_data):
        """Test linear regression wrapper."""
        X, y = regression_data
        wrapper = SklearnModelWrapper(
            model_class="LinearRegression",
            hyperparameters={},
            task="regression",
            random_state=42
        )

        wrapper.fit(X, y)
        predictions = wrapper.predict(X)

        assert len(predictions) == len(y)

        # Check reasonable fit
        from sklearn.metrics import r2_score
        r2 = r2_score(y, predictions)
        assert r2 > 0.8  # Should fit well


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

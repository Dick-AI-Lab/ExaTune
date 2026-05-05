"""
Model-aware default hyperparameter registry for ExaTune visualizations.

Call `get_defaults(config)` to get the defaults dict for whatever model
the experiment YAML describes.  The result is used to mark the "default"
operating point on heatmaps and 3-D surfaces so viewers can see at a glance
how the default configuration compares to the optimal one.

Supported model class names (from config.model.class_name):
  sklearn:
    RandomForestClassifier / RandomForestRegressor
    SVC / SVR
    GradientBoostingClassifier / GradientBoostingRegressor
    LogisticRegression
    KNeighborsClassifier / KNeighborsRegressor
    DecisionTreeClassifier / DecisionTreeRegressor
    MLPClassifier / MLPRegressor
    RidgeClassifier / Ridge / Lasso / ElasticNet
    AdaBoostClassifier / AdaBoostRegressor
  xgboost:
    XGBClassifier / XGBRegressor  (model.type == "xgboost")
  lightgbm:
    LGBMClassifier / LGBMRegressor
"""

from __future__ import annotations
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Per-model default tables
# ---------------------------------------------------------------------------

_MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {

    # ------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------
    "RandomForestClassifier": {
        "n_estimators":     100,
        "max_depth":        None,   # sklearn default = None (unlimited)
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features":     "sqrt",
        "criterion":        "gini",
    },
    "RandomForestRegressor": {
        "n_estimators":     100,
        "max_depth":        None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features":     1.0,    # sklearn ≥1.1 default
        "criterion":        "squared_error",
    },

    # ------------------------------------------------------------------
    # SVM / SVR
    # ------------------------------------------------------------------
    "SVC": {
        "C":       1.0,
        "kernel":  "rbf",
        "gamma":   "scale",
        "degree":  3,        # only relevant for poly kernel
        "coef0":   0.0,
    },
    "SVR": {
        "C":       1.0,
        "kernel":  "rbf",
        "gamma":   "scale",
        "epsilon": 0.1,
    },

    # ------------------------------------------------------------------
    # Gradient Boosting (sklearn)
    # ------------------------------------------------------------------
    "GradientBoostingClassifier": {
        "n_estimators":   100,
        "learning_rate":  0.1,
        "max_depth":      3,
        "min_samples_split": 2,
        "min_samples_leaf":  1,
        "subsample":      1.0,
        "max_features":   None,
    },
    "GradientBoostingRegressor": {
        "n_estimators":   100,
        "learning_rate":  0.1,
        "max_depth":      3,
        "min_samples_split": 2,
        "min_samples_leaf":  1,
        "subsample":      1.0,
        "max_features":   None,
    },

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    "XGBClassifier": {
        "n_estimators":   100,
        "learning_rate":  0.3,
        "max_depth":      6,
        "subsample":      1.0,
        "colsample_bytree": 1.0,
        "reg_alpha":      0.0,
        "reg_lambda":     1.0,
        "min_child_weight": 1,
        "gamma":          0.0,
    },
    "XGBRegressor": {
        "n_estimators":   100,
        "learning_rate":  0.3,
        "max_depth":      6,
        "subsample":      1.0,
        "colsample_bytree": 1.0,
        "reg_alpha":      0.0,
        "reg_lambda":     1.0,
        "min_child_weight": 1,
        "gamma":          0.0,
    },

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------
    "LGBMClassifier": {
        "n_estimators":   100,
        "learning_rate":  0.1,
        "max_depth":      -1,      # -1 = unlimited
        "num_leaves":     31,
        "min_child_samples": 20,
        "subsample":      1.0,
        "colsample_bytree": 1.0,
        "reg_alpha":      0.0,
        "reg_lambda":     0.0,
    },
    "LGBMRegressor": {
        "n_estimators":   100,
        "learning_rate":  0.1,
        "max_depth":      -1,
        "num_leaves":     31,
        "min_child_samples": 20,
        "subsample":      1.0,
        "colsample_bytree": 1.0,
        "reg_alpha":      0.0,
        "reg_lambda":     0.0,
    },

    # ------------------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------------------
    "LogisticRegression": {
        "C":          1.0,
        "penalty":    "l2",
        "solver":     "lbfgs",
        "max_iter":   100,
    },

    # ------------------------------------------------------------------
    # k-NN
    # ------------------------------------------------------------------
    "KNeighborsClassifier": {
        "n_neighbors": 5,
        "weights":     "uniform",
        "metric":      "minkowski",
        "p":           2,
    },
    "KNeighborsRegressor": {
        "n_neighbors": 5,
        "weights":     "uniform",
        "metric":      "minkowski",
        "p":           2,
    },

    # ------------------------------------------------------------------
    # Decision Tree
    # ------------------------------------------------------------------
    "DecisionTreeClassifier": {
        "max_depth":        None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "criterion":        "gini",
        "max_features":     None,
    },
    "DecisionTreeRegressor": {
        "max_depth":        None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "criterion":        "squared_error",
        "max_features":     None,
    },

    # ------------------------------------------------------------------
    # MLP (Neural Net)
    # ------------------------------------------------------------------
    "MLPClassifier": {
        "hidden_layer_sizes": (100,),
        "activation":  "relu",
        "solver":      "adam",
        "alpha":       0.0001,
        "learning_rate_init": 0.001,
        "max_iter":    200,
    },
    "MLPRegressor": {
        "hidden_layer_sizes": (100,),
        "activation":  "relu",
        "solver":      "adam",
        "alpha":       0.0001,
        "learning_rate_init": 0.001,
        "max_iter":    200,
    },

    # ------------------------------------------------------------------
    # Linear models
    # ------------------------------------------------------------------
    "RidgeClassifier": {"alpha": 1.0},
    "Ridge":           {"alpha": 1.0},
    "Lasso":           {"alpha": 1.0},
    "ElasticNet":      {"alpha": 1.0, "l1_ratio": 0.5},

    # ------------------------------------------------------------------
    # AdaBoost
    # ------------------------------------------------------------------
    "AdaBoostClassifier": {
        "n_estimators":  50,
        "learning_rate": 1.0,
        "algorithm":     "SAMME.R",
    },
    "AdaBoostRegressor": {
        "n_estimators":  50,
        "learning_rate": 1.0,
        "loss":          "linear",
    },
}

# Aliases — map common alternate spellings / short names
_ALIASES: Dict[str, str] = {
    "svc":                         "SVC",
    "svr":                         "SVR",
    "xgboost":                     "XGBClassifier",   # generic xgboost type
    "xgbclassifier":               "XGBClassifier",
    "xgbregressor":                "XGBRegressor",
    "lgbmclassifier":              "LGBMClassifier",
    "lgbmregressor":               "LGBMRegressor",
    "randomforestclassifier":      "RandomForestClassifier",
    "randomforestregressor":       "RandomForestRegressor",
    "gradientboostingclassifier":  "GradientBoostingClassifier",
    "gradientboostingregressor":   "GradientBoostingRegressor",
    "logisticregression":          "LogisticRegression",
    "kneighborsclassifier":        "KNeighborsClassifier",
    "kneighborsregressor":         "KNeighborsRegressor",
    "decisiontreeclassifier":      "DecisionTreeClassifier",
    "decisiontreeregressor":       "DecisionTreeRegressor",
    "mlpclassifier":               "MLPClassifier",
    "mlpregressor":                "MLPRegressor",
    "ridgeclassifier":             "RidgeClassifier",
    "adaboostclassifier":          "AdaBoostClassifier",
    "adaboostregressor":           "AdaBoostRegressor",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_defaults_for_model(class_name: str) -> Dict[str, Any]:
    """Return the sklearn/xgboost default hyperparameters for *class_name*.

    The lookup is case-insensitive and handles fully-qualified names like
    ``sklearn.ensemble.RandomForestClassifier`` by extracting the final
    component.

    Returns an empty dict if the model is unrecognised (so callers can
    still function — they just won't draw a default marker).
    """
    # Strip module path — use only the class name
    short = class_name.split(".")[-1]
    # Direct lookup (preserves original casing)
    if short in _MODEL_DEFAULTS:
        return dict(_MODEL_DEFAULTS[short])
    # Case-insensitive alias lookup
    canonical = _ALIASES.get(short.lower())
    if canonical and canonical in _MODEL_DEFAULTS:
        return dict(_MODEL_DEFAULTS[canonical])
    return {}


def get_defaults(config) -> Dict[str, Any]:
    """Extract model defaults from an ExaTune config object or plain dict.

    Accepts:
      - An ``ExaTuneConfig`` object with a ``model`` attribute
      - A plain ``dict`` (e.g. from ``yaml.safe_load``)
      - ``None`` → returns {}

    The model type field (``model.type``) is also checked so that generic
    ``"xgboost"`` or ``"lightgbm"`` types resolve correctly even when
    ``class_name`` is absent.
    """
    if config is None:
        return {}

    # ---- dict path (raw YAML) ----
    if isinstance(config, dict):
        model_section = config.get("model", {})
        # Support both 'class_name' (iris yaml) and 'class' (california yaml)
        class_name = (
            model_section.get("class_name", "")
            or model_section.get("class", "")
        )
        model_type = model_section.get("type", "")
    else:
        # ---- ExaTuneConfig object ----
        model_section = getattr(config, "model", None)
        if model_section is None:
            return {}
        # Support both attribute names
        class_name = (
            getattr(model_section, "class_name", "") or
            getattr(model_section, "class_", "")     or  # 'class' is a reserved word
            getattr(model_section, "model_class", "") or
            ""
        )
        model_type = getattr(model_section, "type", "") or ""

    # Try class_name first
    if class_name:
        defaults = get_defaults_for_model(class_name)
        if defaults:
            return defaults

    # Fall back to model type
    if model_type:
        defaults = get_defaults_for_model(model_type)
        if defaults:
            return defaults

    return {}


def find_default_index(labels: list, param: str, defaults: Dict[str, Any]) -> Optional[int]:
    """Return the integer axis index for the default value of *param*.

    Args:
        labels:   Ordered list of axis tick values (may be str, int, float, None).
        param:    Hyperparameter name.
        defaults: Dict returned by ``get_defaults`` / ``get_defaults_for_model``.

    Returns:
        Integer index into *labels*, or ``None`` if not found.
    """
    if param not in defaults:
        return None

    default_val = defaults[param]

    # Exact string match
    str_labels = [str(v) for v in labels]
    target = str(default_val)
    if target in str_labels:
        return str_labels.index(target)

    # None / null match — covers max_depth=None → "None" label
    if default_val is None:
        for i, v in enumerate(labels):
            if v is None or str(v).lower() in ("none", "null", "nan"):
                return i
        return None

    # Numeric fuzzy match
    try:
        target_f = float(default_val)
        for i, v in enumerate(labels):
            try:
                if abs(float(v) - target_f) < 1e-9:
                    return i
            except (ValueError, TypeError):
                pass
    except (ValueError, TypeError):
        pass

    return None
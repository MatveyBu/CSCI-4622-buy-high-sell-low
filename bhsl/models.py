"""Model factories and per-model feature importance."""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from bhsl import config

MODEL_NAMES = ["majority", "logreg", "adaboost"]


def make_model(name, seed=config.SEED):
    if name == "majority":
        # Predicts the training base rate for everyone: the floor any model must beat.
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if name == "adaboost":
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=100, learning_rate=0.5, random_state=seed,
        )
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def feature_importance(model, feature_names):
    """Signed standardized coefficients for logreg, impurity importance for AdaBoost."""
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "named_steps") and "logisticregression" in model.named_steps:
        values = model.named_steps["logisticregression"].coef_[0]
    else:
        return None
    return dict(zip(feature_names, np.asarray(values, dtype=float)))

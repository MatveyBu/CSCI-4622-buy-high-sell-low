"""Metrics and market-regime slicing."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

VOL_REGIMES = ["low", "medium", "high"]


def classification_metrics(y_true, prob):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n": len(y_true),
        "base_rate": y_true.mean(),
        "accuracy": accuracy_score(y_true, prob >= 0.5),
        "auc": roc_auc_score(y_true, prob) if both_classes else np.nan,
        "brier": brier_score_loss(y_true, prob, labels=[0, 1]),
        "log_loss": log_loss(y_true, prob, labels=[0, 1]),
    }


def vol_tercile_cutoffs(train_df):
    """Benchmark-volatility tercile cutoffs from training dates only.

    Computed over unique dates so days with more stocks don't count more.
    """
    per_date = train_df.drop_duplicates("Date")["mkt_vol_20d"]
    return per_date.quantile([1 / 3, 2 / 3]).to_numpy()


def assign_regimes(df, cutoffs):
    """Volatility tercile (low/medium/high) and trend (bull/bear) for each row."""
    vol = pd.Categorical.from_codes(np.digitize(df["mkt_vol_20d"], cutoffs), VOL_REGIMES)
    trend = np.where(df["mkt_bull"] == 1, "bull", "bear")
    return pd.DataFrame({"vol_regime": vol, "trend_regime": trend}, index=df.index)


def metrics_by(preds, group_cols):
    """Metrics for each (model, feature_set, *group_cols) slice of a predictions frame."""
    keys = ["model", "feature_set"] + list(group_cols)
    rows = []
    for key, g in preds.groupby(keys, observed=True, sort=True):
        rows.append({**dict(zip(keys, key)), **classification_metrics(g["label"], g["prob"])})
    return pd.DataFrame(rows)

"""Walk-forward train/evaluate models on one feature set and write results.

    python -m bhsl.run_experiment                          # baseline: own-stock features
    python -m bhsl.run_experiment --feature-set own_us_intl

Outputs land in results/<feature_set>/: predictions, metrics overall / by year /
by regime, and feature importances averaged over folds.
"""

import argparse
import time

import pandas as pd

from bhsl import config
from bhsl.evaluate import assign_regimes, metrics_by, vol_tercile_cutoffs
from bhsl.features import FEATURE_SETS
from bhsl.models import MODEL_NAMES, feature_importance, make_model
from bhsl.splits import walk_forward_splits


def run(df, feature_set, models, first_test_year):
    features = FEATURE_SETS[feature_set]
    X = df[features].to_numpy()
    y = df["label"].to_numpy()

    preds, importances = [], []
    for year, train_idx, test_idx in walk_forward_splits(df["Date"], df["label_end_date"], first_test_year):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        regimes = assign_regimes(test, vol_tercile_cutoffs(train))
        for name in models:
            start = time.time()
            model = make_model(name).fit(X[train_idx], y[train_idx])
            prob = model.predict_proba(X[test_idx])[:, 1]
            preds.append(pd.DataFrame({
                "Date": test["Date"].to_numpy(), "Ticker": test["Ticker"].to_numpy(),
                "test_year": year, "model": name, "feature_set": feature_set,
                "label": test["label"].to_numpy(), "excess_return": test["excess_return"].to_numpy(),
                "prob": prob,
                "vol_regime": regimes["vol_regime"].to_numpy(),
                "trend_regime": regimes["trend_regime"].to_numpy(),
            }))
            imp = feature_importance(model, features)
            if imp:
                importances.append({"model": name, "test_year": year, **imp})
            print(f"  {year} {name:<9} train={len(train_idx):>7,} test={len(test_idx):>6,} "
                  f"({time.time() - start:.1f}s)")

    return pd.concat(preds, ignore_index=True), pd.DataFrame(importances)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feature-set", default="own", choices=FEATURE_SETS)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--first-test-year", type=int, default=config.FIRST_TEST_YEAR)
    args = parser.parse_args()

    df = pd.read_parquet(config.DATASET).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    print(f"Feature set '{args.feature_set}' ({len(FEATURE_SETS[args.feature_set])} features), "
          f"{len(df):,} rows")

    preds, importances = run(df, args.feature_set, args.models, args.first_test_year)

    overall = metrics_by(preds, [])
    by_year = metrics_by(preds, ["test_year"])
    by_regime = pd.concat([
        metrics_by(preds, ["vol_regime"]).rename(columns={"vol_regime": "regime"}).assign(regime_type="volatility"),
        metrics_by(preds, ["trend_regime"]).rename(columns={"trend_regime": "regime"}).assign(regime_type="trend"),
    ], ignore_index=True)

    out_dir = config.RESULTS_DIR / args.feature_set
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_dir / "predictions.parquet", index=False)
    overall.to_csv(out_dir / "metrics_overall.csv", index=False)
    by_year.to_csv(out_dir / "metrics_by_year.csv", index=False)
    by_regime.to_csv(out_dir / "metrics_by_regime.csv", index=False)
    if not importances.empty:
        importances.groupby("model").mean(numeric_only=True).drop(columns="test_year").T \
            .to_csv(out_dir / "feature_importance.csv")

    cols = ["model", "n", "base_rate", "accuracy", "auc", "brier"]
    pd.set_option("display.width", 120)
    print(f"\nOverall (pooled over test years {args.first_test_year}+):")
    print(overall[cols].to_string(index=False, float_format="%.4f"))
    print("\nBy regime:")
    print(by_regime[["regime_type", "regime"] + cols].to_string(index=False, float_format="%.4f"))
    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()

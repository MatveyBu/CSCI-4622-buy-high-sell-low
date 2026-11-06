"""Leakage and correctness checks on synthetic data (no downloads needed)."""

import numpy as np
import pandas as pd
import pytest

from bhsl import config
from bhsl.data import align_to_us_calendar
from bhsl.evaluate import assign_regimes, vol_tercile_cutoffs
from bhsl.features import ALL_FEATURES, build_dataset, forward_labels, market_features, own_stock_features
from bhsl.splits import walk_forward_splits


def random_walk(index, columns, seed):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.01, size=(len(index), len(columns)))
    return pd.DataFrame(100 * np.exp(steps.cumsum(axis=0)), index=index, columns=columns)


@pytest.fixture
def synthetic():
    calendar = pd.bdate_range("1999-01-01", "2002-12-31")
    close = random_walk(calendar, ["AAA", "BBB", "CCC"], seed=0)
    volume = random_walk(calendar, close.columns, seed=1) * 1e4
    index_prices = random_walk(calendar, config.CORE_INDEXES, seed=2)
    return calendar, close, volume, index_prices


def test_features_do_not_look_ahead(synthetic):
    """Changing prices after date t must not change any feature on or before t."""
    calendar, close, volume, index_prices = synthetic
    cutoff = calendar[600]

    shocked_close, shocked_volume, shocked_idx = close.copy(), volume.copy(), index_prices.copy()
    shocked_close.loc[calendar > cutoff] *= 3
    shocked_volume.loc[calendar > cutoff] *= 5
    shocked_idx.loc[calendar > cutoff] *= 0.2

    before = calendar <= cutoff
    pd.testing.assert_frame_equal(
        own_stock_features(close, volume).loc[calendar[before]],
        own_stock_features(shocked_close, shocked_volume).loc[calendar[before]],
    )
    pd.testing.assert_frame_equal(
        market_features(index_prices).loc[before],
        market_features(shocked_idx).loc[before],
    )


def test_label_matches_manual_computation(synthetic):
    calendar, close, _, index_prices = synthetic
    bench = index_prices[config.BENCHMARK]
    labels = forward_labels(close, bench, horizon=5)

    t, h = calendar[100], calendar[105]
    expected = (close.loc[h, "AAA"] / close.loc[t, "AAA"] - 1) - (bench[h] / bench[t] - 1)
    row = labels.loc[(t, "AAA")]
    assert row["excess_return"] == pytest.approx(expected)
    assert row["label"] == float(expected > 0)
    assert row["label_end_date"] == h
    # The last `horizon` days have no future prices, so no label.
    assert labels.loc[calendar[-5:], "label"].isna().all()


def test_ffill_limit():
    dates = pd.bdate_range("2020-01-01", periods=10)
    px = pd.DataFrame({"X": [1.0, 2.0] + [np.nan] * 5 + [3.0, 4.0, 5.0]}, index=dates)
    aligned = align_to_us_calendar(px, dates, limit=3)
    assert aligned["X"].iloc[2:5].eq(2.0).all()
    assert aligned["X"].iloc[5:7].isna().all()


def test_walk_forward_purges_overlapping_labels(synthetic):
    calendar, close, volume, index_prices = synthetic
    df = build_dataset(close, volume, index_prices, calendar)
    splits = list(walk_forward_splits(df["Date"], df["label_end_date"], first_test_year=2001))
    assert [year for year, _, _ in splits] == [2001, 2002]
    for year, train_idx, test_idx in splits:
        test_start = pd.Timestamp(year=year, month=1, day=1)
        assert (df["label_end_date"].iloc[train_idx] < test_start).all()
        assert (df["Date"].iloc[test_idx].dt.year == year).all()
        assert not set(train_idx) & set(test_idx)


def test_build_dataset_shape(synthetic):
    calendar, close, volume, index_prices = synthetic
    df = build_dataset(close, volume, index_prices, calendar)
    assert not df[ALL_FEATURES + ["label"]].isna().any().any()
    assert df["Date"].min() >= pd.Timestamp(config.START_DATE)
    # Consecutive sample dates are SAMPLE_EVERY trading days apart.
    positions = calendar.get_indexer(df["Date"].drop_duplicates())
    assert set(np.diff(positions)) == {config.SAMPLE_EVERY}


def test_vol_terciles_use_training_cutoffs():
    train = pd.DataFrame({"Date": pd.date_range("2000-01-01", periods=9), "mkt_vol_20d": np.arange(9.0)})
    cutoffs = vol_tercile_cutoffs(train)
    test = pd.DataFrame({"mkt_vol_20d": [-1.0, 4.0, 100.0], "mkt_bull": [1, 0, 1]})
    regimes = assign_regimes(test, cutoffs)
    assert list(regimes["vol_regime"]) == ["low", "medium", "high"]
    assert list(regimes["trend_regime"]) == ["bull", "bear", "bull"]

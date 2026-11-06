"""Feature and label construction.

Every feature on date t uses only closes up to and including t. Asian and European
markets close before the US session, and TSX closes with NYSE, so day-t index returns
are all known at the US close on day t. The label uses closes t+1 .. t+HORIZON only.
"""

import numpy as np
import pandas as pd

from bhsl import config
from bhsl.data import align_to_us_calendar

ANNUALIZE = np.sqrt(252)

OWN_FEATURES = [
    "ret_1d", "ret_5d", "ret_20d",
    "ma20_ratio", "ma50_ratio", "ma200_ratio",
    "vol_20d", "volume_ratio_20d",
]
US_MARKET_FEATURES = [f"{idx}_ret_{w}d" for idx in config.US_INDEXES for w in (1, 5)]
INTL_FEATURES = [f"{idx}_ret_{w}d" for idx in config.INTL_INDEXES for w in (1, 5)]
REGIME_FEATURES = ["mkt_vol_20d", "mkt_bull", "global_vol_20d", "global_dispersion_1d"]

# Nested feature sets for the experiments. "own" is the baseline; comparing
# "own_us" to "own_us_intl" isolates what international markets add beyond the US market.
FEATURE_SETS = {
    "own": OWN_FEATURES,
    "own_us": OWN_FEATURES + US_MARKET_FEATURES,
    "own_us_intl": OWN_FEATURES + US_MARKET_FEATURES + INTL_FEATURES,
    "all": OWN_FEATURES + US_MARKET_FEATURES + INTL_FEATURES + REGIME_FEATURES,
}
ALL_FEATURES = FEATURE_SETS["all"]


def _stack(frames):
    """{name: Date x Ticker frame} -> long frame indexed by (Date, Ticker)."""
    cols = {name: f.stack(future_stack=True) for name, f in frames.items()}
    out = pd.concat(cols, axis=1)
    out.index.names = ["Date", "Ticker"]
    return out


def own_stock_features(close, volume):
    """Per-stock price/volume features from wide (Date x Ticker) frames."""
    rets = close.pct_change(fill_method=None)
    feats = {
        "ret_1d": rets,
        "ret_5d": close.pct_change(5, fill_method=None),
        "ret_20d": close.pct_change(20, fill_method=None),
        "ma20_ratio": close / close.rolling(20).mean() - 1,
        "ma50_ratio": close / close.rolling(50).mean() - 1,
        "ma200_ratio": close / close.rolling(200).mean() - 1,
        "vol_20d": rets.rolling(20).std() * ANNUALIZE,
        "volume_ratio_20d": (volume / volume.rolling(20).mean()).replace([np.inf, -np.inf], np.nan),
    }
    return _stack(feats)


def market_features(aligned_index_prices):
    """Date-level features from index closes already aligned to the US calendar."""
    px = aligned_index_prices
    rets = px.pct_change(fill_method=None)
    feats = {}
    for idx in config.CORE_INDEXES:
        feats[f"{idx}_ret_1d"] = rets[idx]
        feats[f"{idx}_ret_5d"] = px[idx].pct_change(5, fill_method=None)

    bench = px[config.BENCHMARK]
    vol20 = rets.rolling(20).std() * ANNUALIZE
    ma200 = bench.rolling(200).mean()
    feats["mkt_vol_20d"] = vol20[config.BENCHMARK]
    feats["mkt_bull"] = (bench > ma200).astype(float).where(ma200.notna())
    feats["global_vol_20d"] = vol20[config.CORE_INDEXES].mean(axis=1, skipna=False)
    feats["global_dispersion_1d"] = rets[config.CORE_INDEXES].std(axis=1, skipna=False)
    return pd.DataFrame(feats)


def forward_labels(close, benchmark, horizon=config.HORIZON):
    """Forward excess return over `horizon` days and its binary outperformance label.

    label = 1 if the stock's forward return beats the benchmark's, else 0.
    label_end_date is the last date whose close the label depends on, used to purge
    training rows whose label overlaps the test period.
    """
    fwd_stock = close.shift(-horizon) / close - 1
    fwd_bench = benchmark.shift(-horizon) / benchmark - 1
    excess = fwd_stock.sub(fwd_bench, axis=0)
    label = (excess > 0).astype(float).where(excess.notna())
    out = _stack({"excess_return": excess, "label": label})
    end_dates = pd.Series(close.index, index=close.index).shift(-horizon)
    out["label_end_date"] = end_dates.reindex(out.index.get_level_values("Date")).to_numpy()
    return out


def build_dataset(close, volume, index_prices, calendar):
    """Assemble the modeling table: one row per (sample date, ticker).

    Samples every SAMPLE_EVERY-th US trading day within [START_DATE, END_DATE], and
    drops rows missing any feature in ALL_FEATURES so every feature set is evaluated
    on identical rows.
    """
    close = close.reindex(calendar)
    volume = volume.reindex(calendar)
    aligned = align_to_us_calendar(index_prices, calendar)

    own = own_stock_features(close, volume)
    mkt = market_features(aligned)
    labels = forward_labels(close, aligned[config.BENCHMARK])

    df = own.join(labels).join(mkt, on="Date")

    window = calendar[(calendar >= config.START_DATE) & (calendar <= config.END_DATE)]
    sample_dates = window[::config.SAMPLE_EVERY]
    df = df[df.index.get_level_values("Date").isin(sample_dates)]
    df = df.dropna(subset=ALL_FEATURES + ["label"])
    df["label"] = df["label"].astype(int)
    df["mkt_bull"] = df["mkt_bull"].astype(int)
    return df.reset_index()

"""Build the modeling table (features + labels) from the index file and cached stock prices.

    python -m bhsl.make_dataset
"""

import argparse

from bhsl import config
from bhsl.data import load_index_prices, load_stock_prices, us_calendar
from bhsl.features import ALL_FEATURES, build_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    index_prices = load_index_prices()
    close, volume = load_stock_prices()
    df = build_dataset(close, volume, index_prices, us_calendar(index_prices))

    config.DATASET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.DATASET, index=False)

    print(f"Saved {len(df):,} rows x {len(ALL_FEATURES)} features -> {config.DATASET}")
    print(f"  tickers:      {df['Ticker'].nunique()}")
    print(f"  sample dates: {df['Date'].nunique()} ({df['Date'].min():%Y-%m-%d} .. {df['Date'].max():%Y-%m-%d})")
    print(f"  base rate:    {df['label'].mean():.3f} (share of rows that outperform {config.BENCHMARK})")


if __name__ == "__main__":
    main()

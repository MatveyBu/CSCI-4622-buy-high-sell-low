"""Download and cache daily stock prices for the ticker universe.

    python -m bhsl.download_stocks [--min-start 1999-12-31]
"""

import argparse

import pandas as pd

from bhsl import config
from bhsl.data import download_stock_prices, read_tickers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-start", default="1999-12-31",
                        help="drop tickers whose history starts after this date")
    args = parser.parse_args()

    tickers = read_tickers()
    print(f"Downloading {len(tickers)} tickers {config.DOWNLOAD_START} -> {config.END_DATE} ...")
    prices = download_stock_prices(tickers)

    first = prices.groupby("Ticker")["Date"].min()
    missing = sorted(set(tickers) - set(first.index))
    late = sorted(first[first > pd.Timestamp(args.min_start)].index)
    if missing:
        print(f"No data returned for: {', '.join(missing)}")
    if late:
        print(f"Dropping (history starts after {args.min_start}): {', '.join(late)}")
    prices = prices[~prices["Ticker"].isin(late)]

    config.STOCK_PRICES.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(config.STOCK_PRICES, index=False)
    print(f"Saved {prices['Ticker'].nunique()} tickers, {len(prices):,} rows -> {config.STOCK_PRICES}")


if __name__ == "__main__":
    main()

"""Loading the Kaggle index file and downloading/caching individual stock prices."""

import pandas as pd

from bhsl import config


def load_index_prices(indexes=config.CORE_INDEXES, path=config.INDEX_CSV):
    """Wide frame of adjusted closes (Date x Index), each index on its own trading dates.

    Rows where every price field is empty (~2% of the file) are dropped.
    """
    df = pd.read_csv(path, parse_dates=["Date"], encoding="utf-8-sig")
    df = df.dropna(subset=["Adj Close"])
    df = df[df["Index"].isin(indexes)]
    wide = df.pivot(index="Date", columns="Index", values="Adj Close").sort_index()
    return wide[list(indexes)]


def us_calendar(index_prices):
    """US trading days, taken as the dates the benchmark has a close."""
    return index_prices[config.BENCHMARK].dropna().index


def align_to_us_calendar(index_prices, calendar, limit=config.FFILL_LIMIT):
    """Put every index on the US calendar, forward-filling short gaps only.

    Filling happens on the union of all dates, so a foreign holiday carries forward
    that market's last real close (a 0% return that day). Gaps longer than `limit`
    rows stay NaN. Forward-filling only ever uses past values, so it can't leak.
    """
    return index_prices.ffill(limit=limit).reindex(calendar)


def read_tickers(path=config.TICKERS_FILE):
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def download_stock_prices(tickers, start=config.DOWNLOAD_START, end=config.END_DATE):
    """Download split/dividend-adjusted daily closes and volume from Yahoo Finance.

    Returns a long frame with columns Date, Ticker, Close, Volume.
    """
    import yfinance as yf

    # yfinance treats `end` as exclusive.
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=start, end=end_exclusive, auto_adjust=True,
                      progress=False, threads=True)
    long = {field: raw[field].rename_axis(columns="Ticker").stack(future_stack=True)
            for field in ("Close", "Volume")}
    out = pd.concat(long, axis=1).dropna(subset=["Close"]).reset_index()
    return out.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def load_stock_prices(path=config.STOCK_PRICES):
    """Wide (Date x Ticker) frames of close and volume from the cached download."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m bhsl.download_stocks` first.")
    long = pd.read_parquet(path)
    close = long.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    volume = long.pivot(index="Date", columns="Ticker", values="Volume").sort_index()
    return close, volume

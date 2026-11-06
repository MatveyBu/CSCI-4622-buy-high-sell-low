"""Project-wide constants: paths, universe, horizon, and evaluation settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_CSV = DATA_DIR / "indexData.csv"
TICKERS_FILE = DATA_DIR / "tickers.txt"
STOCK_PRICES = DATA_DIR / "stocks" / "prices.parquet"
DATASET = DATA_DIR / "processed" / "dataset.parquet"
RESULTS_DIR = ROOT / "results"

# Benchmark the label is measured against (also defines the US trading calendar).
BENCHMARK = "NYA"
US_INDEXES = ["NYA", "IXIC"]
INTL_INDEXES = ["GSPTSE", "GDAXI", "N225", "HSI", "SSMI", "N100", "KS11"]
CORE_INDEXES = US_INDEXES + INTL_INDEXES

# Modeling window. Downloads start a year early so 200-day moving averages are
# warmed up by START_DATE. END_DATE is the last NYA row in the Kaggle file.
DOWNLOAD_START = "1999-01-01"
START_DATE = "2000-01-01"
END_DATE = "2021-05-28"

# Label horizon in trading days. Sampling every HORIZON days keeps labels from overlapping.
HORIZON = 5
SAMPLE_EVERY = 5

# Max consecutive US trading days a foreign index may be forward-filled (holidays).
FFILL_LIMIT = 3

# Walk-forward evaluation: train on all years before the test year (expanding window).
FIRST_TEST_YEAR = 2006

SEED = 4622

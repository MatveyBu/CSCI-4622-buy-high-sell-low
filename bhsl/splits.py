"""Walk-forward (expanding window) splits by calendar year."""

import numpy as np
import pandas as pd


def walk_forward_splits(dates, label_end_dates, first_test_year, last_test_year=None):
    """Yield (test_year, train_idx, test_idx) for an expanding window by year.

    Training rows are everything whose label is fully resolved before the test year
    starts. That purges the last few samples of the prior year whose forward window
    reaches into the test year, which would otherwise leak test-period prices.
    """
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    label_end_dates = pd.to_datetime(pd.Series(label_end_dates)).reset_index(drop=True)
    years = dates.dt.year
    last_test_year = last_test_year or int(years.max())

    for year in range(first_test_year, last_test_year + 1):
        test_start = pd.Timestamp(year=year, month=1, day=1)
        train_idx = np.flatnonzero((label_end_dates < test_start).to_numpy())
        test_idx = np.flatnonzero((years == year).to_numpy())
        if len(train_idx) and len(test_idx):
            yield year, train_idx, test_idx

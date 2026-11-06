# Buy High, Sell Low

**Team:** Matvey Bubalo, Anika Nagpal, Jaron Barrett — CSCI 4622

A supervised classifier that predicts whether a US stock will **outperform the NYSE Composite (NYA) over the next 5 trading days**. We use it to answer two questions:

1. Do recent moves in international equity markets add predictive information beyond a stock's own history and the US market?
2. Does that predictive value change across market regimes (low/medium/high volatility, bull/bear)?

The goal is a controlled measurement of *when* global information helps. We are not claiming a trading edge.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

python -m bhsl.download_stocks     # ~1 min: Yahoo Finance prices -> data/stocks/prices.parquet
python -m bhsl.make_dataset        # features + labels -> data/processed/dataset.parquet
python -m bhsl.run_experiment      # baseline (own-stock features) -> results/own/
pytest                             # leakage / correctness tests
```

Run the other feature sets with `python -m bhsl.run_experiment --feature-set {own,own_us,own_us_intl,all}`.

## Repository layout

```
data/
  indexData.csv          Kaggle "Stock Exchange Data" (14 indexes, 1965–2021)   [tracked]
  indexInfo.csv          index -> exchange/region/currency                         [tracked]
  indexProcessed.csv     Kaggle's processed variant (unused)                       [tracked]
  tickers.txt            stock universe (113 S&P 500 names)                        [tracked]
  stocks/                downloaded stock prices                                   [ignored]
  processed/             modeling table                                            [ignored]
bhsl/
  config.py              paths, index lists, dates, horizon, first test year
  data.py                load Kaggle indexes, US-calendar alignment, yfinance download
  features.py            own-stock / market / regime features, labels, dataset assembly
  splits.py              expanding-window walk-forward splits with label purging
  models.py              majority / logistic regression / AdaBoost
  evaluate.py            accuracy, AUC, Brier, log loss; regime assignment
  download_stocks.py     CLI: fetch and cache stock prices
  make_dataset.py        CLI: build the modeling table
  run_experiment.py      CLI: walk-forward train + evaluate, write results
notebooks/               exploration and figures
tests/                   synthetic-data tests for look-ahead, labels, splits
results/<feature_set>/   predictions + metrics CSVs                                [ignored]
```

## Design

**Data.** Index closes come from the Kaggle file. Stock prices (split and dividend adjusted) come from Yahoo Finance via `yfinance`. All series are aligned to the US trading calendar (NYA's dates). Foreign indexes are forward-filled across their own holidays for at most 3 days. Rows with longer gaps are dropped. Index volume is not used because ~38% of it is recorded as 0.

**Samples.** One row per (stock, date). We sample every 5th US trading day from 2000-01-01 to 2021-05-28, so the 5-day labels never overlap. Rows missing any feature are dropped, which keeps every feature set evaluated on *identical* rows. Current build: **119,666 rows, 113 stocks, 1,059 dates.**

**Label.** `excess_return = stock 5-day forward return − NYA 5-day forward return`; `label = 1` if `excess_return > 0`. About 51.6% of rows are positive.

**Features (30).** All are known at the US close on day *t*.

| Group | Features |
|---|---|
| Own stock (8) | 1/5/20-day returns; price ÷ 20/50/200-day MA − 1; 20-day realized vol; volume ÷ 20-day avg volume |
| US market (4) | 1- and 5-day returns of NYA and Nasdaq (IXIC) |
| International (14) | 1- and 5-day returns of TSX, DAX, Nikkei, Hang Seng, SMI, Euronext 100, KOSPI |
| Regime (4) | NYA 20-day vol; bull flag (NYA > 200-day MA); mean 20-day vol across the 9 indexes; cross-index dispersion of 1-day returns |

Asian and European markets close before the US session, and the TSX closes at the same time as NYSE. That makes their day-*t* returns legitimately known at the US close on day *t*.

**Feature sets** are nested so each comparison isolates one group:

| Set | Contents | Role |
|---|---|---|
| `own` | own stock | **baseline** |
| `own_us` | + US market | control: what plain market context gives |
| `own_us_intl` | + international | **main experiment** (compare with `own_us`) |
| `all` | + regime features | full model |

**Validation.** Expanding-window walk-forward by year. For test year *Y* (2006 … 2021), the model trains on every row whose label is fully resolved before Jan 1 of *Y*. Training rows whose 5-day forward window reaches into *Y* are purged. Random splits are never used.

**Regimes.** For each fold, the volatility terciles (low/medium/high) use cutoffs computed from the *training* dates' NYA 20-day volatility. Bull means NYA is above its 200-day MA and bear means it is below. Every metric is reported per regime.

**Models.** `majority` (predicts the training base rate; the floor to beat), `logreg` (standardized logistic regression), and `adaboost` (100 depth-2 trees, learning rate 0.5).

## Baseline results (`own`, test years 2006–2021, pooled)

| Model | n | Accuracy | AUC | Brier |
|---|---|---|---|---|
| majority | 86,106 | 0.5176 | 0.502 | 0.2497 |
| logreg | 86,106 | 0.5144 | 0.510 | 0.2498 |
| adaboost | 86,106 | 0.5168 | 0.512 | 0.2567 |

Own-history features alone carry almost no signal. Accuracy does not beat always predicting "outperform", and AUC is only about 0.51. That matches the weak-form efficiency literature and gives a clean floor to measure the international features against. AUC and Brier score are better guides than accuracy here, because the base rate shifts from year to year. AdaBoost's higher Brier score comes from its poorly calibrated probabilities. The majority model's AUC is slightly above 0.5 only because its constant prediction changes from fold to fold.

A quick smoke test of `all` with logistic regression gives AUC 0.516. Proper comparisons across feature sets are the next step.

## Known limitations

- **Survivorship bias:** the universe is stocks that were still in the S&P 500 and listed in 2021, chosen in hindsight. This affects all feature sets equally, so the *relative* comparison holds up better than the absolute numbers.
- Yahoo's adjusted prices can be revised over time, so re-downloading may shift results slightly.
- Kaggle index data ends in late May / early June 2021, and 2021 is a partial test year.

## Next steps

- [ ] Run all four feature sets and tabulate lift (Δ AUC, Δ Brier) for `own_us_intl` vs `own_us`, overall and per regime
- [ ] Add a significance check for the lift (e.g. bootstrap by date)
- [ ] Look at feature importance over time, especially for 2008 and 2020
- [ ] EDA notebook: coverage, base rate by year, correlations among index returns
- [ ] Stretch goals: lead-lag between markets, learned regimes (e.g. HMM), a simple long-short backtest, regression on `excess_return`

## References

- Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137–1187.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
- Hamao, Y., Masulis, R. W., & Ng, V. (1990). Correlations in price changes and volatility across international stock markets. *Review of Financial Studies*, 3(2), 281–307.
- Mattiuz, C. (n.d.). Stock Exchange Data [Data set]. Kaggle. https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data

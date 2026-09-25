# InvestmentManagement

A research, backtesting and analytics platform for equity factor strategies, built on plug-and-play market-data
vendors (Refinitiv / LSEG today; Yahoo, Tiingo, Marketstack, SimFin adapters included) and persistent analytics
storage in SQL Server (or Databricks).

The reference strategy is a **momentum + machine-learning value** composite traded long/short on the S&P 500,
benchmarked against a float-adjusted reconstruction of the index, the SPY ETF and the ^GSPC index level.

## Why systematic investing

Beating a market index consistently is hard, and the hardest part is not finding an idea - it is executing it
without your own behaviour getting in the way. Discretionary investors must decide *what* to buy, *when* to
rebalance, and *how much* to hold on every single day, and studies of real accounts repeatedly show that the
average person underperforms the very funds they hold because they chase recent winners and sell into panic.

A systematic process removes that human loop from the hot path:

- **Rules, not moods.** The position on every rebalance date is computed from the same formula, so you cannot
  talk yourself out of a sensible trade or into a story-driven one.
- **Scale beyond human capacity.** A single rebalance across the S&P 500 means weighing 500+ stocks on dozens
  of fundamentals and price signals each - far more than any person can compare by hand or hold in their head,
  so a disciplined human inevitably narrows to a familiar handful and leaves the rest uninvestigated. The engine
  scores every name every time, with no fatigue and no home-team bias.
- **Point-in-time discipline.** The engine only ever uses information that was knowable on the rebalance date
  (no peeking at future fundamentals or prices), which is the difference between a backtest you can trust and a
  backtest that lies to you.
- **Costs and turnover are first-class.** Every simulation charges realistic transaction costs and shifts the
  signal by one period, so quoted returns are what you could actually have booked.
- **Reproducible and auditable.** A run is a function of data plus code; anyone (including you, six months later)
  can re-run it, challenge it, and improve it.

This repository is that process made concrete: loaders that gather clean, point-in-time data, an analytics core
that turns it into factors and portfolios, and notebooks that walk from raw prices to a live long/short book.

## Layout

```
analytics/                 vendor-agnostic research engine (pure pandas / numpy / scikit-learn)
  factors/                 Factor & PanelFactor contracts, MomentumFactor, ValueFactor, MLReturnFactor,
                           CompositeFactor, transforms (zscore / rank / winsorize / neutralize), ml_training
  portfolio/               constituent weights & returns, rebalance schedules, quantile long/short weights
  backtest/                run_backtest, BacktestResult, performance metrics, IC / quantile diagnostics
  performance/             plots (+ legacy performance_analytics facade)
  risk/                    PortfolioVaR (historical & parametric)
  data_quality.py          glitch detection / neutralisation for price panels
data_engineering/          everything that touches a vendor or the database
  refinitiv/               shared LSEG plumbing: lazy session, RIC helpers, chunked + retried requests
  eod_data/                PriceVendor base + Refinitiv / Yahoo / Tiingo / Marketstack price vendors
  fundamentals/            18-ratio contract + providers: database (point-in-time), Refinitiv, Yahoo, in-memory
  index_constituents/      index membership reconstruction + float-adjusted shares (Refinitiv)
  security_master/         security master ingestion (Refinitiv, FinanceDatabase)
  loaders/                 pipelines: portfolios & holdings, backtest universe, batched EOD load, fundamentals load
  database/                ORM models, connection factory, read_* / write_* helpers (SQL Server + Databricks)
toolkit/notebooks/         numbered walk-throughs (see below)
toolkit/scripts/           command-line loaders
tests/unit_tests/          pytest suite (runs without a database or vendor)
docs/                      REFACTORING_NOTES.md, STRATEGY_IMPROVEMENTS.md
```

Design rule: **`analytics` never imports a vendor or the database.** It consumes two standard shapes -
a *price panel* (`DataFrame`, index = date, columns = `security_id`) and a *fundamentals panel*
(index = `security_id`, columns = the 18 `RATIO_COLUMNS`) - so any vendor that fills those shapes works.

## Getting started

```sh
git clone https://github.com/menonf/InvestmentManagement.git
cd InvestmentManagement
python -m venv .venv && .venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -e .[refinitiv,dev]                          # editable install: no sys.path hacks anywhere
pytest                                                   # 60+ offline unit tests
```

Database credentials live in the OS keyring under `ihub_sql_connection` (`db`, `uid`, `pwd`; optionally
`server` + `trusted=1` for a local instance). Refinitiv needs LSEG Workspace running; the session opens lazily.

## Notebooks

Run them in order the first time to build the live dataset; `06` runs on synthetic data anywhere.

| # | Notebook | Needs | What it does |
|---|---|---|---|
| 1 | [01_load_reference_data_and_prices.ipynb](toolkit/notebooks/01_load_reference_data_and_prices.ipynb) | DB + LSEG | Loads `.SPX` membership, float-adjusted shares and four base portfolios (SP500 / SPY / GSPC / MAG8), then runs the batched, retried EOD price load with mop-up and holiday pruning. |
| 2 | [02_backtest_sp500_reconstruction.ipynb](toolkit/notebooks/02_backtest_sp500_reconstruction.ipynb) | DB | Reconstructs the S&P 500 under market / equal / mixed weighting and checks it against ^GSPC and SPY (tracking error, drawdowns, VaR). |
| 3 | [03_load_ml_value_fundamentals.ipynb](toolkit/notebooks/03_load_ml_value_fundamentals.ipynb) | DB + LSEG | Loads the point-in-time quarterly history of the 18 fundamental ratios per security, with per-security retries and a no-look-ahead audit. |
| 4 | [04_momentum_ml_value_long_short.ipynb](toolkit/notebooks/04_momentum_ml_value_long_short.ipynb) | DB | Trains the ML value model on a time-based split, scores point-in-time, and builds the composite momentum+value long/short with costs and IC / quintile diagnostics. |
| 5 | [05_stock_screen_long_short.ipynb](toolkit/notebooks/05_stock_screen_long_short.ipynb) | DB | Runs the trained composite on the latest data and shows what the strategy would long and short *today*, with the score distribution and signal scatter. |
| 6 | [06_factor_research_offline_demo.ipynb](toolkit/notebooks/06_factor_research_offline_demo.ipynb) | nothing | Exercises the entire engine end-to-end on synthetic data - runs anywhere, ships with outputs - to demonstrate the analytics core without a vendor or database. |

## Ten lines of research code

```python
from analytics.backtest import prices_to_returns, run_backtest
from analytics.factors import CompositeFactor, MomentumFactor, ValueFactor
from analytics.portfolio import month_starts, quantile_weights

rebalances = month_starts(prices.index)
momentum = MomentumFactor(252, 21).compute(prices)
value = ValueFactor().compute_dynamic(prices, lambda d: provider.get_panel(symbols, str(d.date())), dates=rebalances)
signal = CompositeFactor({"momentum": momentum, "value": value}, rebalance_dates=rebalances).compute(prices)
weights = quantile_weights(signal, quantile=0.1, membership=membership)
result = run_backtest(weights, prices_to_returns(prices), shift_weights=1, cost_bps=10)
result.summary()
```

## Adding a new factor

A factor is one small class; everything downstream (composite blending, point-in-time
re-scoring, backtesting) is inherited from the base, so no engine changes are needed.

- Fundamentals-based factors (value, quality, ...) subclass `PanelFactor` and implement
  `score_panel(panel) -> Series`, scoring one cross-section of the 18 `RATIO_COLUMNS`.
- Price-based factors (momentum, volatility, ...) subclass `Factor` and implement
  `compute(prices, fundamentals=None) -> DataFrame`.
- Register it in `analytics/factors/__init__.py`, then drop it into a backtest via
  `CompositeFactor({"quality": QualityFactor(), ...})` or test it alone with `run_backtest`.
- The data contract: the 18 `RATIO_COLUMNS` (see `data_engineering/fundamentals/ratios.py`)
  are the fundamental-ratio panel every provider emits and the database stores. A hand-built
  factor can read any of those 18 freely - no extra work.
- If your factor needs a metric *outside* the 18 (e.g. accruals, earnings volatility, capex
  efficiency), you have to add it to the pipeline first, in two steps:
    1. **Extend the ratio panel** - add the new raw inputs to `RAW_ITEMS`, add the formula in
       `compute_ratios`, and widen the database `metric_type` filters (in `database_provider.py`,
       `store.py`, `loaders/fundamentals_loader.py`) from `RATIO_COLUMNS` to include the new
       names. Note the new raw data must actually be fetched: the Refinitiv provider only requests
       the fields in `REFINITIV_REQUEST_FIELDS`, so you must add the matching `TR.*` codes there
       (and confirm your LSEG entitlement returns them). Then reload fundamentals so they are stored.
    2. **Retrain only if you want the ML model to use it** - `MLReturnFactor` selects its 18
       features *by name* (`panel.reindex(columns=RATIO_COLUMNS)`), so extra columns are simply
       ignored and the existing trained `.joblib` models keep working untouched. You only need to
       retrain (`analytics.factors.ml_training.train_models`) if you specifically want the ML value
       factor to consume the new metric. A factor you write by hand never needs retraining.

## Command-line loaders

```sh
python toolkit/scripts/load_refinitiv_fundamentals.py --universe SP500 --frequency Q --start 2020-01-01
python toolkit/scripts/load_float_adjusted_shares.py
python toolkit/scripts/repair_eod_prices.py 2025-01-01 2025-12-31
python toolkit/scripts/portfolio_data_load.py --portfolios SP500 SPY --start 2025-01-01 --end 2025-06-30
```

## Technology

pandas / NumPy / SciPy / scikit-learn for research;</br>SQLAlchemy + pyodbc (SQL Server) or Databricks for storage;</br>
`lseg.data` and Refinitiv Terminal;</br> matplotlib for reporting;</br> `ib_insync` & Interactive Brokers Terminal for held position loads.

## Key differentiators

- **Persistent analytics storage.** Unlike most open-source factor tools that recompute everything from raw
  files each run, the platform persists point-in-time fundamentals and EOD prices in SQL Server (or Databricks),
  so backtests and live scoring read a stable, auditable history rather than re-pulling vendors.
- **Vendor-agnostic by contract.** `analytics` depends only on a price panel and a fundamentals panel. Refinitiv
  is the production source, but Yahoo / Tiingo / Marketstack / SimFin adapters and an in-memory provider are
  already wired in, so swapping a vendor never touches the research engine.
- **Modular and composable.** Factors, portfolio weights, rebalance schedules and backtests are independent
  objects that plug together (see the ten-line example above), so new signals or weighting schemes drop in without
  a rewrite.

## Roadmap

Short-term (build out the data and signal layer):

- Extend index reconstructions beyond the S&P 500 (e.g. Nasdaq 100) and broaden the backtest "what-if" tooling.
- Harden the fundamentals coverage and no-look-ahead audit so more vendors can feed the ML value factor.

Medium / long-term direction:

- Layer style, country, industry and thematic factor exposures on top of the current growth & momentum base.
- Add an explicit value factor framing to surface securities trading below intrinsic value.
- Combine the above into a growth-at-a-reasonable-price (GARP) strategy and, eventually, run live capital
  through the framework via the Interactive Brokers integration.

## Reference reading that shaped the strategy:
* *Quantitative Momentum* (Wes Gray)</br>
* *Quantitative Value*(Wes Gray & Tobias Carlisle)</br>
* *The Intelligent Investor* (Benjamin Graham)</br>
* *Factor Investing for Dummies*</br>
* *Build Your Own AI Investor*(Damon Lee)

## Further reading

`docs/STRATEGY_IMPROVEMENTS.md` - prioritised ideas to strengthen the factors and the portfolio.

## Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results.
</br>*Built with ❤️ for the quantitative finance community*

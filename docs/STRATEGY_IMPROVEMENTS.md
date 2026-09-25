# Improving the factors and the strategy

Prioritised suggestions from reviewing the momentum + ML value long/short strategy. Items marked **[built in]** are
already available as options in the refactored code (defaults keep the old behaviour); the rest are design notes.

## A. Things that bias the current backtest (fix first)

1. **Fundamentals publication lag – [built in]** `effective_date` is the fiscal *period end*, but 10-Q/10-K numbers
   are public 30–60 days later. Scoring on the period end lets the model see a quarter before the market did.
   `StaticFundamentalsProvider(availability_lag_days=45)` (and the in-memory provider) shift availability; notebook 04
   uses 45 days. Better still: pull Refinitiv's *announcement* dates (`TR.BSOriginalAnnouncementDate` /
   `TR.ISPeriodEndDate`) and store them as the availability date per row.
2. **Same-day execution – [built in]** the old loop applied a signal built from day *t*'s close to day *t*'s return.
   `run_backtest(shift_weights=1)` trades at the next close. Consider `shift_weights=2` for a conservative
   "signal at close, trade next day, earn from the day after" convention.
3. **Overlapping labels in the ML training set** – monthly snapshots with a 3- or 12-month label overlap heavily,
   so 80/20 splits leak information across the boundary and inflate test scores. **[built in]**
   `time_based_split(embargo_months=FORWARD_MONTHS)` drops the training snapshots whose label window crosses the split.
4. **Survivorship in the composite universe – [built in]** restricting each date to point-in-time index members
   (`membership` mask in `quantile_weights`) stops the book from holding leavers after they left.
5. **Static single split** – one model trained once and used for the whole window is either stale (early years) or
   forward-looking (if the window straddles training data). Move to **walk-forward retraining**: refit every 12
   months on an expanding window and score the next year only. `train_models` + `MLReturnFactor(model_dir=...)`
   make this a loop over year-specific model directories.

## B. Momentum

1. **Volatility scaling – [built in]** `MomentumFactor(vol_scale=True)` divides the return by trailing 3-month
   volatility (risk-managed momentum, Barroso & Santa-Clara 2015). Expect a smaller crash in 2020/2022-type
   reversals and a higher Sharpe at the cost of lower raw return.
2. **Sector-neutral scores – [built in]** `neutralize(scores, sector_map)` demeans within GICS sector (the security
   master already stores `sector`). Cuts the strategy's unintended sector bets, which dominate a decile book on
   the S&P 500.
3. **Residual momentum** – regress each name's returns on the market (and sector) over the lookback and rank the
   residual return. Lower turnover and less crash-prone than raw momentum (Blitz, Huij & Martens 2011).
4. **Signal smoothing / holding buffer** – recompute monthly but only trade names that leave the top/bottom
   *15 %* once they were in the top/bottom 10 %. Halves turnover with little loss of signal.

## C. Value

1. **Transparent baseline – [built in]** `ValueFactor` = mean z-score of E/P, B/P, S/P, EBIT/EV. Run it next to the
   ML factor in notebook 04; if the ML factor cannot beat it out of sample, the model is not adding value beyond the
   ratios it is fed.
2. **What the model predicts – [built in]** `build_modelling_table(demean_target=True)` trains on the return
   *relative to the cross-section*. The raw target makes every tree spend capacity forecasting the market level,
   which a dollar-neutral book never earns.
3. **Ranked / winsorised features** – ratios have fat tails and change regime (P/E of 400 vs −50). Convert each
   ratio to its cross-sectional percentile per snapshot (`rank_pct` / `winsorize` in `analytics.factors.transforms`)
   before training; tree models become far more stable across years.
4. **Model capacity** – `GradientBoostingRegressor(max_depth=10)` and `DecisionTreeRegressor(max_depth=15)` will
   memorise a few thousand rows. Try depth 3–4 with 300–500 trees, `subsample=0.8`, `min_samples_leaf=50`, and
   `HistGradientBoostingRegressor` for speed. Report the *rank* IC on the test window, not MSE.
5. **Add quality and growth items** – the 18 ratios are value/quality oriented; adding earnings-revision, accruals
   (`(NI − CFO) / TA`), asset growth and share issuance gives the model orthogonal information. The ratio contract is
   one list (`RATIO_COLUMNS`); extend it and retrain.
6. **Interpretability** – log feature importances (`est.feature_importances_`) or SHAP values per retrain; a model
   whose top feature flips every year is fitting noise.

## D. Combining signals

1. **Rank blending – [built in]** `CompositeFactor(method="rank")` is robust to a signal with occasional extreme
   z-scores (an ML prediction of +300 %).
2. **IC-weighted blend** – weight each component by its trailing 24-month IC (computed with
   `information_coefficient`), floored at zero. Lets the data down-weight a signal that has stopped working.
3. **Missing-as-neutral** – `combine(missing=0)` treats a missing component as average. For names with no
   fundamentals that means they can still be selected on momentum alone; pass `missing=None` to require both.

## E. Portfolio construction

1. **Costs and turnover – [built in]** `run_backtest(cost_bps=...)` and `BacktestResult.turnover()`. A decile book
   rebalanced monthly on 500 names turns over roughly 50–70 % per month; at 10 bps one-way that is 1–2 % a year,
   more with market impact. Always show net-of-cost results.
2. **Beta-neutral instead of dollar-neutral** – equal long/short notional still carries beta when the long leg is
   higher-beta (momentum longs usually are). Scale the short leg by the ratio of leg betas, or hedge with the SPY line.
3. **Position and liquidity limits – [built in]** `quantile_weights(max_weight=...)`; add a minimum market-cap /
   ADV screen from `shares_outstanding × price` before ranking.
4. **Risk targeting** – scale the whole book to a constant ex-ante volatility (e.g. 10 % annualised) using the
   trailing 63-day realised volatility of `result.returns`; makes Sharpe comparisons across periods meaningful.
5. **Rebalance frequency** – test monthly vs quarterly. Momentum decays over ~3–6 months, value over years; a
   quarterly value refresh with a monthly momentum refresh (different `rebalance_dates` per component) is natural.

## F. Evaluation discipline

* Report **IC and quintile spreads** (notebook 04 §4.6) before any portfolio statistics; they are far less sensitive
  to construction choices than a decile book's Sharpe.
* Keep a **hold-out period** (e.g. the last 12 months) that is never used while iterating on features or parameters.
* Compare each idea against the **transparent baselines** (`ValueFactor`, plain `MomentumFactor`) and against
  `SP500` / `GSPC`; an improvement should show up in IC, Sharpe *and* drawdown, not just total return.
* Track results across parameter neighbours (e.g. quantile 0.1 / 0.15 / 0.2, lookback 6 / 9 / 12 months); a
  strategy that only works at one setting is noise.

## G. Data engineering hygiene that affects results

* Store the **announcement date** alongside `effective_date` in `security_fundamentals` (see A.1).
* Persist the daily **factor scores** with `database.compute_and_store_factors` so signal history is auditable and
  `factor_scores` can feed attribution.
* Keep the **glitch filter** (`neutralize_single_day_glitches`) as a report first: review what it flags before trusting
  a series, and repair the source with `toolkit/scripts/repair_eod_prices.py`.
* Use **total-return** prices for the long/short legs (`adj_close` already includes splits; check dividends are in the
  adjustment for Refinitiv `TR.CLOSEPRICE(Adjusted=1)`), otherwise the short leg is systematically flattered.

# Solari hub demand forecasting — approach, validation and findings

**Run:** `20260925_075244` · **Seed:** `42` (models use `42`..`45`)
· **Hardware:** cuda:0, cuda:1 · **XGBoost** 3.2.0

## 1. Problem and metric

Forecast daily `OrderVolume` for 1,115 hubs over
42 future days (2015-06-20 to 2015-07-31), scored with
RMSLE. Hub scale spans 8x from smallest to
largest, so the model is trained directly on `log1p(OrderVolume)` with an L2 objective — the
loss the metric implies — and predictions are mapped back with `expm1`.

## 2. Approach

- **Models:** a log-space blend of 4 structurally different gradient-boosting
  forecasters, all trained on the GPUs (weights
  XGBoost rolling-origin 0.55, XGBoost multi-gap direct 0.25, CatBoost direct 0.00, XGBoost direct, no lags 0.20,
  chosen on the holiday-free rows of both held-out blocks). No deep learning.
  - _Rolling-origin XGBoost:_ 4 boosters (`hist`, depth 10,
    eta 0.02), 1113 rounds, averaged in log space.
  - _Multi-gap direct XGBoost:_ trained on every open day with `HubID` as a native
    categorical and a pseudo-Huber loss (slope 0.2);
    6 models whose lags end
    7, 14, 21, 28, 35, 42 days back. A test day `h` days ahead uses the smallest
    gap `>= h`, so near days get the freshest history the horizon allows.
  - _Direct CatBoost:_ gaps 14, 42 with ordered target statistics on HubID, Weekday, RegionCluster.
  - _Plain direct XGBoost:_ `HubID` as a categorical plus the operating calendar and metadata,
    no lag features, 2769 rounds.
  - Every direct model's final fit is a bag of 1 seeds; round budgets
    come from early stopping on the holiday-free rows of the back-test block.
- **Rows used:** open days with positive volume. Closed days (`IsOpen == 0`) are predicted as
  exact zeros — in the full history that implication holds without a single exception.
- **Features (128 total):** hub history statistics (overall, by weekday, by promo
  state, by month, with shrinkage), trailing 7/28/91/182/365-day levels and trends, the
  operating calendar supplied for the test window (promo runs, days since/until promo, closure
  runs), calendar seasonality, and hub metadata (format, assortment, competitor distance and
  age, loyalty programme timing), plus 12 regional calendars inferred
  from identical holiday/school-closure patterns, holiday/school/long-closure proximity, and last
  year's seasonal shape around the same date. `AppSessions` exists only in train, so it enters
  solely as hub-level aggregates computed inside each statistics window (and as a lagged level in
  the direct model).
- **Leakage control:** history statistics are always computed from a window ending strictly
  before the window being predicted, and training rows come from _rolling origins_ —
  17 consecutive 42-day windows, each described
  by its own earlier statistics window.
  No training row contributes to any statistic that describes it. Three automated audits assert
  this (Section 8.2).
- **Calibration:** one global multiplier (1.012), minimising the combined
  holiday-free RMSLE of the validation and back-test blocks, applied only when the two blocks'
  own optima agree.
- **Per-hub correction:** evaluated but not applied (gain -0.00020 below the threshold).

## 3. Validation

Time-blocked, never random. Two held-out _future_ blocks of exactly 42 days —
the same horizon as the leaderboard window:

| Block      | Target window           | Purpose                               |
| ---------- | ----------------------- | ------------------------------------- |
| back-test  | 2015-03-28 → 2015-05-08 | hyper-parameter and round selection   |
| validation | 2015-05-09 → 2015-06-19 | unbiased scoring (fixed round budget) |

**Results**

| Model                                                 | RMSLE       |
| ----------------------------------------------------- | ----------- |
| XGBoost single booster (mean of 4)                    | 0.09810     |
| XGBoost 4-booster ensemble                            | 0.09758     |
| XGBoost multi-gap direct (6 gap models)               | 0.09691     |
| CatBoost direct (2 gap models)                        | 0.09772     |
| XGBoost plain direct (no lags)                        | 0.10187     |
| blend                                                 | 0.09429     |
| blend + calibration                                   | 0.09432     |
| blend + calibration + per-hub correction              | **0.09432** |
| blend, test-like rows only (no holiday within 3 days) | 0.08319     |
| blend, excluding open-but-zero rows                   | 0.08339     |
| back-test block, blend (models early-stopped here)    | 0.08881     |

**Naive baselines on the same validation block**

| Baseline               | RMSLE   |
| ---------------------- | ------- |
| global_mean            | 0.36564 |
| hub_mean               | 0.23417 |
| hub_weekday_mean       | 0.18407 |
| hub_weekday_promo_mean | 0.14544 |

**Accuracy decay over the forecast horizon**

| Horizon | RMSLE   |
| ------- | ------- |
| week 1  | 0.13741 |
| week 2  | 0.08318 |
| week 3  | 0.07885 |
| week 4  | 0.08175 |
| week 5  | 0.08881 |
| week 6  | 0.08238 |

## 4. Findings

- Hub identity and recent level dominate: the top features are `HubDowPromoMean`, `HubPromoMean`, `HubTrailPromo91`, `HubTrailDow365`, `HubDowMean`, `PromoActive`, `HolidaysThisWeek`, `HubTrailDow91`.
- Gain share by feature family:

| Family                  | Share |
| ----------------------- | ----- |
| hub history             | 53.9% |
| trailing-window history | 26.0% |
| calendar                | 11.7% |
| promotion calendar      | 5.3%  |
| operating status        | 1.8%  |
| hub metadata            | 0.5%  |
| year-over-year          | 0.5%  |
| network shape           | 0.4%  |

- Promotions lift demand materially and the model captures them: RMSLE on promo days is
  0.0860 vs 0.0986 on
  non-promo days.
- Error is concentrated: the worst 1% of rows carry
  35.8% of the total squared log error.
- Hubs with a renovation gap in their history score
  0.0879 vs 0.0955 for
  the rest, and the first week after a closure is harder
  (0.1048) than settled trading days
  (0.1207).
- Residual log-space bias after calibration is
  +0.00499, i.e. the forecast is effectively unbiased at the network level.

## 5. Limitations

- `RegionalHoliday` takes values 1–3 in training but is uniformly 0 across the test window, so
  the holiday features cannot contribute there. The validation block does contain holidays, so
  the all-rows score is conservative for this test period; model selection therefore uses the
  test-like (holiday-free) rows.
- A 42-day-ahead forecast cannot react to level shifts that begin after the history ends; the
  hub-series plots in Section 15.1 show this is the dominant residual failure mode.
- The calibration multiplier is a single parameter fitted on the validation block, which makes
  the reported validation RMSLE marginally optimistic. Its optimum agrees with the independent
  back-test block, which is the check that keeps it honest.

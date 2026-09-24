# Solari hub demand forecasting — approach, validation and findings

**Run:** `20260924_185033` · **Seed:** `42` (models use `42`..`49`)
· **Hardware:** cuda:0, cuda:1 · **XGBoost** 3.2.0

## 1. Problem and metric

Forecast daily `OrderVolume` for 1,115 hubs over
42 future days (2015-06-20 to 2015-07-31), scored with
RMSLE. Hub scale spans 8x from smallest to
largest, so the model is trained directly on `log1p(OrderVolume)` with an L2 objective — the
loss the metric implies — and predictions are mapped back with `expm1`.

## 2. Approach

- **Models:** a log-space blend of three structurally different gradient-boosting forecasters
  (weights XGBoost rolling-origin 0.15, LightGBM direct 0.70, XGBoost multi-gap direct 0.15,
  chosen on both held-out blocks). No deep learning.
  - _Rolling-origin XGBoost:_ 8 boosters (`hist`, depth 10,
    eta 0.03), 1400 rounds, averaged in log space.
  - _Direct LightGBM (CPU, concurrent with the GPU boosters when GPUs exist):_
    2 boosters, 4249 rounds, trained on every
    open day with `HubID` as a native categorical, a Huber loss (alpha
    0.2) and lags that are all at least 42 days old.
  - _Multi-gap direct XGBoost (GPU):_ 6 models whose lags end
    7, 14, 21, 28, 35, 42 days back. A test day `h` days ahead uses the smallest
    gap `>= h`, so near days get the freshest history the horizon allows.
- **Rows used:** open days with positive volume. Closed days (`IsOpen == 0`) are predicted as
  exact zeros — in the full history that implication holds without a single exception.
- **Features (122 total):** hub history statistics (overall, by weekday, by promo
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
  14 consecutive 42-day windows, each described
  by its own earlier statistics window.
  No training row contributes to any statistic that describes it. Three automated audits assert
  this (Section 8.2).
- **Calibration:** one global multiplier (1.014), selected on the validation block
  and confirmed on an independent back-test block.

## 3. Validation

Time-blocked, never random. Two held-out _future_ blocks of exactly 42 days —
the same horizon as the leaderboard window:

| Block      | Target window           | Purpose                               |
| ---------- | ----------------------- | ------------------------------------- |
| back-test  | 2015-03-28 → 2015-05-08 | hyper-parameter and round selection   |
| validation | 2015-05-09 → 2015-06-19 | unbiased scoring (fixed round budget) |

**Results**

| Model                                              | RMSLE       |
| -------------------------------------------------- | ----------- |
| XGBoost single booster (mean of 8)                 | 0.09890     |
| XGBoost 8-booster ensemble                         | 0.09776     |
| LightGBM direct (2 seeds)                          | 0.09558     |
| XGBoost multi-gap direct (6 gap models)            | 0.09684     |
| blend                                              | 0.09436     |
| blend + calibration                                | **0.09358** |
| blend, excluding open-but-zero rows                | 0.08259     |
| back-test block, blend (models early-stopped here) | 0.08455     |

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
| week 1  | 0.13353 |
| week 2  | 0.07916 |
| week 3  | 0.08222 |
| week 4  | 0.08155 |
| week 5  | 0.08855 |
| week 6  | 0.08482 |

## 4. Findings

- Hub identity and recent level dominate: the top features are `HubDowPromoMean`, `HubPromoMean`, `HubTrailPromo91`, `HubTrailDow365`, `HolidaysThisWeek`, `PromoActive`, `HolidayTomorrow`, `HubDowMean`.
- Gain share by feature family:

| Family                  | Share |
| ----------------------- | ----- |
| hub history             | 54.0% |
| trailing-window history | 22.3% |
| calendar                | 14.8% |
| promotion calendar      | 4.4%  |
| operating status        | 2.9%  |
| hub metadata            | 0.8%  |
| network shape           | 0.5%  |
| year-over-year          | 0.4%  |

- Promotions lift demand materially and the model captures them: RMSLE on promo days is
  0.0850 vs 0.0980 on
  non-promo days.
- Error is concentrated: the worst 1% of rows carry
  36.3% of the total squared log error.
- Hubs with a renovation gap in their history score
  0.0881 vs 0.0946 for
  the rest, and the first week after a closure is harder
  (0.1041) than settled trading days
  (0.1177).
- Residual log-space bias after calibration is
  +0.00028, i.e. the forecast is effectively unbiased at the network level.

## 5. Limitations

- `RegionalHoliday` takes values 1–3 in training but is uniformly 0 across the test window, so
  the holiday features cannot contribute there; the validation block does contain holidays, so
  the reported score is, if anything, conservative for this specific test period.
- A 42-day-ahead forecast cannot react to level shifts that begin after the history ends; the
  hub-series plots in Section 15.1 show this is the dominant residual failure mode.
- The calibration multiplier is a single parameter fitted on the validation block, which makes
  the reported validation RMSLE marginally optimistic. Its optimum agrees with the independent
  back-test block, which is the check that keeps it honest.

# AGENTS.md

Context and ground rules for anyone (human or coding agent) working in this repository. It
combines the competition brief that used to live in `Overview.md`, `Dataset Description.md`
and `Rules.md`. For a plain-language walkthrough of the notebook and its results, read
[README.md](README.md).

## Overview

Company_X operates a network of over 800 micro-fulfillment hubs delivering groceries and
household essentials within 15-30 minutes of order placement. Because delivery windows are
short and hub storage is limited, demand forecasting errors are far more costly here than in
traditional retail: overstocking wastes space and spoils perishables, while understocking
causes cancelled orders and lost trust.

The task is to predict daily order volume for each hub over a future time window, using
historical order data, hub attributes, and promotional/calendar context. Solari's operations
team will use accurate forecasts to drive replenishment schedules, picker/packer staffing, and
delivery fleet allocation.

This is a regression problem evaluated on a held-out future time period, not a random data
split. Submissions are scored using RMSLE (see [Evaluation](#evaluation)).

### Description

Demand forecasting for a large-scale fulfillment network requires understanding both
historical demand patterns and the operational factors that influence them. Participants use
historical order data along with hub, promotional, calendar, and operational information to
predict future daily order volumes for each fulfillment hub.

The dataset reflects real-world business scenarios, requiring careful exploratory data
analysis, feature engineering, and robust validation to build models that generalize well to
unseen future periods. It is a time-series regression problem where success depends on both
forecasting accuracy and a strong understanding of the underlying data.

## Dataset

Historical daily order data is provided for 1,115 Solari hubs. The task is to forecast the
`OrderVolume` column for the test set. Some hubs in the dataset were temporarily closed for
renovation.

### Files (in `data/`)

- `orders_train.csv`: historical daily order volume per hub, including `OrderVolume`, along
  with daily operating context (operational status, promotion activity, day of week, holiday
  indicator).
- `orders_test.csv`: the same structure, covering a future date range for which `OrderVolume`
  must be predicted. Same-day actuals that would not be known in advance are replaced with
  planned/scheduled values.
- `hub_metadata.csv`: static and semi-static hub attributes (format, assortment tier,
  competitor distance and tenure, loyalty program status), joined via `HubID`.
- `sample_submission.csv`: a sample submission file in the correct format.

All files share a common hub identifier for joining.

### Data fields

- `Id`: identifies a (`Hub`, `Date`) pair within the test set.
- `HubID`: a unique identifier for each hub.
- `OrderVolume`: the daily order total for a hub (target variable).
- `AppSessions`: a proxy signal for daily customer activity at the hub.
- `IsOpen`: whether the hub was operating: `0` = closed, `1` = open.
- `RegionalHoliday`: a regional holiday. With few exceptions, hubs run reduced operations on
  these days; schools are also closed on these dates and on weekends. `1` = public holiday,
  `2` = spring/religious holiday, `3` = winter holiday, `0` = none.
- `SchoolClosureFlag`: whether the (`Hub`, `Date`) was affected by a period of local school
  closures.
- `HubFormat`: differentiates between four hub models: `1`, `2`, `3`, `4`.
- `AssortmentTier`: product range carried: `1` = basic, `2` = extended, `3` = premium.
- `CompetitorDistance`: distance in metres to the nearest competing hub.
- `CompetitorOpenSince[Month/Year]`: approximate month/year the nearest competitor opened.
- `PromoActive`: whether a promotion is running at the hub that day.
- `LoyaltyProgram`: participation in Solari's recurring loyalty program: `0` = not
  participating, `1` = participating.
- `LoyaltyProgramSince[Year/Week]`: year and week the hub joined the loyalty program.
- `LoyaltyProgramInterval`: the months each new loyalty program round begins for that hub
  (for example "Feb,May,Aug,Nov").

## Evaluation

Submissions are evaluated on Root Mean Squared Logarithmic Error (RMSLE) between predicted and
actual daily order volume:

$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( \log(p_i + 1) - \log(a_i + 1) \right)^2}$$

where $p_i$ is the predicted order volume, $a_i$ is the actual order volume, and $n$ is the
number of rows in the test set.

RMSLE is used because hub sizes vary by an order of magnitude across the network. A plain RMSE
would let a handful of large, mature hubs dominate the score; RMSLE penalizes proportional
error instead, so accuracy at small and newly launched hubs counts just as much as accuracy at
large ones. RMSLE also handles the many zero-order-volume rows (renovation-closed hubs) more
gracefully than a percentage-based metric like MAPE, which is undefined at zero.

The submission is a CSV with two columns, `Id,OrderVolume`, one row per test `Id`.

## Rules

1. **No external data:** predictions must come solely from the provided files.
2. **No deep learning or generative AI required.**
3. **Submission limit:** maximum of 10 submissions to the leaderboard.
4. **Individual participation:** one account per candidate. No team submissions.
5. **Reproducibility:** notebooks/scripts must run end-to-end without manual intervention.
6. **Random seeds:** fixed and disclosed random seeds must be used.
7. **Documentation:** a 1-2 page written summary of approach, validation, and findings is
   required.
8. **Deadlines:** late submissions after the 24-hour window are not scored.

## Working in this repository

- The whole pipeline is one notebook, `solari-hub-demand-forecasting.ipynb`, written for a
  Kaggle session with two T4 GPUs. It is not meant to be executed locally for a full run.
- Every run artefact lands in `solari_forecast/` (figures, metrics, logs, predictions,
  reports, model metadata). Treat these files as outputs of a run: regenerate them by
  re-running the notebook, do not hand-edit them.
- `solari_forecast/xgb_worker.py` is written by the notebook itself (Section 9). Change the
  `WORKER_SOURCE` string in the notebook, not the file.
- Keep the rules above intact in any change: no external data, no deep learning, the master
  seed `CFG.SEED = 42` stays disclosed, and features must stay leakage-safe (every history
  statistic ends strictly before the window it describes).
- Model binaries (`solari_forecast/models/*.ubj`, `*.cbm`), `*.zip` archives and `.venv/` are
  git-ignored. Do not commit them.
- JSON and Markdown formatting follows `.prettierrc.json` (tabs, width 2).

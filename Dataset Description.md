# Dataset Description

Historical daily order data is provided for 1,115 Solari hubs. Your task is to forecast the `OrderVolume` column for the test set. Note that some hubs in the dataset were temporarily closed for renovation.

## Files

- `orders_train.csv`: Historical data including `OrderVolume`.
- `orders_test.csv`: Historical data excluding `OrderVolume`.
- `sample_submission.csv`: A sample submission file in the correct format.
- `hub_metadata.csv`: Supplemental information about each hub.

## Data Fields

- `Id`: Identifies a (`Hub`, `Date`) pair within the test set.
- `HubID`: A unique identifier for each hub.
- `OrderVolume`: The daily order total for a hub (target variable).
- `AppSessions`: A proxy signal for daily customer activity at the hub.
- `IsOpen`: Indicates whether the hub was operating: `0` = closed, `1` = open.
- `RegionalHoliday`: Indicates a regional holiday. With few exceptions, hubs run reduced operations on these days; schools are also closed on these dates and on weekends. `1` = public holiday, `2` = spring/religious holiday, `3` = winter holiday, `0` = none.
- `SchoolClosureFlag`: Indicates whether the (`Hub`, `Date`) was affected by a period of local school closures.
- `HubFormat`: Differentiates between four hub models: `1`, `2`, `3`, `4`.
- `AssortmentTier`: Product range carried: `1` = basic, `2` = extended, `3` = premium.
- `CompetitorDistance`: Distance in meters to the nearest competing hub.
- `CompetitorOpenSince[Month/Year]`: Approximate month/year the nearest competitor opened.
- `PromoActive`: Indicates whether a promotion is running at the hub that day.
- `LoyaltyProgram`: Indicates participation in Solari's recurring loyalty program: `0` = not participating, `1` = participating.
- `LoyaltyProgramSince[Year/Week]`: Year and week the hub joined the loyalty program.
- `LoyaltyProgramInterval`: The months each new loyalty program round begins for that hub (e.g. "Feb,May,Aug,Nov").

> **Note:** No external data may be used to supplement these files.

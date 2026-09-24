Dataset Description
Historical daily order data is provided for 1,115 Solari hubs. Your task is to forecast the OrderVolume column for the test set. Note that some hubs in the dataset were temporarily closed for renovation.

Files
orders_train.csv — historical data including OrderVolume
orders_test.csv — historical data excluding OrderVolume
sample_submission.csv — a sample submission file in the correct format
hub_metadata.csv — supplemental information about each hub
Data Fields
Id — identifies a (Hub, Date) pair within the test set
HubID — a unique identifier for each hub
OrderVolume — the daily order total for a hub (this is what you are predicting)
AppSessions — a proxy signal for daily customer activity at the hub
IsOpen — indicates whether the hub was operating: 0 = closed, 1 = open
RegionalHoliday — indicates a regional holiday. With few exceptions, hubs run reduced operations on these days; schools are also closed on these dates and on weekends. 1 = public holiday, 2 = spring/religious holiday, 3 = winter holiday, 0 = none
SchoolClosureFlag — indicates whether the (Hub, Date) was affected by a period of local school closures
HubFormat — differentiates between four hub models: 1, 2, 3, 4
AssortmentTier — product range carried: 1 = basic, 2 = extended, 3 = premium
CompetitorDistance — distance in meters to the nearest competing hub
CompetitorOpenSince[Month/Year] — approximate month/year the nearest competitor opened
PromoActive — indicates whether a promotion is running at the hub that day
LoyaltyProgram — indicates participation in Solari's recurring loyalty program: 0 = not participating, 1 = participating
LoyaltyProgramSince[Year/Week] — year and week the hub joined the loyalty program
LoyaltyProgramInterval — the months each new loyalty program round begins for that hub (e.g. "Feb,May,Aug,Nov")
No external data may be used to supplement these files.
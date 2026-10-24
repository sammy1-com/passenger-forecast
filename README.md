# passenger-forecast

Knowing next quarter's demand decides how much stock to order, how many staff to roster, how many seats to put on a route. This repo forecasts monthly airline passengers (1949-1960), a classic series with a clear trend and a seasonal pattern that grows over time.

## Run it

```bash
pip install -r requirements.txt
python src/forecast.py
```

The data (a small CSV from the seaborn-data repo) is downloaded on the first run.

## Methods

- **Seasonal naive**: repeat the last 12 months. Any model has to beat this to be worth using.
- **Holt-Winters**: additive trend, multiplicative seasonality (the seasonal swings get bigger as the series grows, so multiplicative fits).
- **Boosted ratios**: gradient boosting that predicts "this month / same month last year" from the previous three ratios and the month number, then rolls forward recursively. I did it on ratios because trees can't extrapolate a trend on raw values.

The last 24 months are held out. No random split, that would leak the future into training.

## Notes

- 144 points is very little data for anything beyond classical methods, so it's no surprise Holt-Winters is hard to beat.
- Errors in recursive forecasting compound, you can see the boosted model drift further out.
- Next: add prediction intervals, and try SARIMA for comparison.

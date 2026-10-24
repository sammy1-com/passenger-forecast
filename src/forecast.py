"""Forecast monthly passengers: seasonal naive vs Holt-Winters vs boosted trees on lag ratios."""
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from data import load

RESULTS = Path(__file__).resolve().parent.parent / "results"
HORIZON = 24  # hold out the last two years

warnings.filterwarnings("ignore")


def seasonal_naive(train, horizon):
    return np.resize(train.values[-12:], horizon).astype(float)


def holt_winters(train, horizon):
    fit = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12).fit()
    return fit.forecast(horizon).values


def boosted(train, horizon):
    """Predict the ratio to the same month last year from the last 3 ratios + month, recursively."""
    s = train.values.astype(float)
    months = train.index.month
    r = s[12:] / s[:-12]  # r[i] is the ratio for position i + 12

    X, y = [], []
    for t in range(15, len(s)):
        X.append([r[t - 13], r[t - 14], r[t - 15], months[t]])
        y.append(r[t - 12])
    model = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=0)
    model.fit(np.array(X), np.array(y))

    hist, rh, out = list(s), list(r), []
    future = pd.date_range(train.index[-1], periods=horizon + 1, freq="MS")[1:]
    for m in future.month:
        rt = model.predict(np.array([[rh[-1], rh[-2], rh[-3], m]]))[0]
        val = rt * hist[-12]
        hist.append(val)
        rh.append(rt)
        out.append(val)
    return np.array(out)


def main():
    RESULTS.mkdir(exist_ok=True)
    s = load()
    train, test = s.iloc[:-HORIZON], s.iloc[-HORIZON:]

    methods = {
        "seasonal_naive": seasonal_naive,
        "holt_winters": holt_winters,
        "boosted_ratios": boosted,
    }

    forecasts, rows = {}, []
    for name, fn in methods.items():
        f = fn(train, HORIZON)
        forecasts[name] = f
        mae = np.mean(np.abs(test.values - f))
        mape = np.mean(np.abs((test.values - f) / test.values)) * 100
        rows.append((name, mae, mape))
        print(f"{name:15s} mae={mae:.1f}  mape={mape:.1f}%")

    lines = [f"Last {HORIZON} months held out.", "", "| method | MAE (passengers) | MAPE (%) |", "|---|---|---|"]
    lines += [f"| {n} | {m:.1f} | {p:.1f} |" for n, m, p in rows]
    (RESULTS / "metrics.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train.index, train.values, color="grey", label="train")
    ax.plot(test.index, test.values, color="black", label="actual")
    for name, f in forecasts.items():
        ax.plot(test.index, f, "--", label=name)
    ax.set_ylabel("passengers (thousands)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "forecast.png", dpi=140)


if __name__ == "__main__":
    main()

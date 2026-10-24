import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv"
CACHE = Path(__file__).resolve().parent.parent / "data" / "flights.csv"


def load():
    """Monthly airline passengers 1949-1960 as a Series with a month-start index."""
    if not CACHE.exists():
        CACHE.parent.mkdir(exist_ok=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        CACHE.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    df = pd.read_csv(CACHE)
    month = df["month"].astype(str).str[:3]
    idx = pd.to_datetime(df["year"].astype(str) + " " + month, format="%Y %b")
    return pd.Series(df["passengers"].values, index=idx, name="passengers").asfreq("MS")

import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf

SYMBOL = "^NSEI"  # Nifty 50 index on Yahoo
DATA_DIR = Path("data") / "raw"
CSV_PATH = DATA_DIR / "nifty_daily.csv"


def download_incremental():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    today = dt.date.today()

    if CSV_PATH.exists():
        df_old = pd.read_csv(CSV_PATH, parse_dates=["Date"])
        last_date = df_old["Date"].max().date()
        start_date = last_date + dt.timedelta(days=1)
        print(f"Existing data till {last_date}. Fetching from {start_date}...")
    else:
        df_old = None
        start_date = dt.date(2000, 1, 1)
        print(f"No existing CSV. Fetching from {start_date}...")

    if start_date > today:
        print("No new dates to download.")
        return

    end_date = today + dt.timedelta(days=1)

    new = yf.download(
        SYMBOL,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if new.empty:
        print("No new data returned from yfinance.")
        return

    new.reset_index(inplace=True)
    new.rename(
        columns={
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Adj Close": "AdjClose",
            "Volume": "Volume",
        },
        inplace=True,
    )

    new = new[["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"]]

    if df_old is not None:
        df_all = pd.concat([df_old, new], ignore_index=True)
        df_all = (
            df_all.drop_duplicates(subset=["Date"])
            .sort_values("Date")
            .reset_index(drop=True)
        )
    else:
        df_all = new.sort_values("Date").reset_index(drop=True)

    df_all.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df_all)} rows to {CSV_PATH}")


if __name__ == "__main__":
    download_incremental()

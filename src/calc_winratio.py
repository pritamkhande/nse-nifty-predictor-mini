# src/calc_winratio.py

import pandas as pd
from pathlib import Path
import json

DATA_PATH = Path("data/raw/nifty_daily.csv")
HIST_PATH = Path("outputs/nifty_predictions_history.csv")
OUT_PATH = Path("outputs/winratio_last_30.json")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("nifty_daily.csv not found. Run download_nifty.py first.")
    if not HIST_PATH.exists():
        raise FileNotFoundError("nifty_predictions_history.csv not found. Run predict_next_day.py first.")

    # Load OHLC data
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "AdjClose", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute next-day close and actual direction
    df["next_close"] = df["Close"].shift(-1)
    # Set actual_up to NaN when next_close is NaN (no next day yet)
    df["actual_up"] = (df["next_close"] > df["Close"]).astype("float")
    df.loc[df["next_close"].isna(), "actual_up"] = pd.NA

    # Load AI predictions
    hist = pd.read_csv(HIST_PATH)
    hist = hist.sort_values("generated_at_utc", ascending=False).reset_index(drop=True)

    # Take last 30 predictions (by time)
    hist_30 = hist.head(30).copy()

    results = []
    win_count = 0

    for _, row in hist_30.iterrows():
        pred_for_str = str(row["predicted_for"])
        try:
            pred_for_date = pd.to_datetime(pred_for_str).date()
        except Exception:
            continue

        # Match by calendar date
        match = df[df["Date"].dt.date == pred_for_date]
        if match.empty:
            # No market data (holiday or future date)
            continue

        actual_up_val = match["actual_up"].values[0]
        if pd.isna(actual_up_val):
            # No next-day close yet, cannot evaluate this trade
            continue

        actual_up = int(actual_up_val)
        ai_up = 1 if str(row["prediction"]).upper() == "UP" else 0

        win = (ai_up == actual_up)
        win_count += int(win)

        results.append({
            "prediction_for": pred_for_str,
            "ai_prediction": str(row["prediction"]).upper(),
            "prob_up": round(float(row["prob_up"]) * 100, 1),
            "actual_up": actual_up,
            "result": "WIN" if win else "LOSS"
        })

    total = len(results)
    win_ratio = (win_count / total * 100) if total > 0 else 0.0

    output = {
        "total_predictions": total,
        "wins": win_count,
        "loss": total - win_count,
        "win_ratio_percent": round(win_ratio, 2),
        "details": results
    }

    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

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

    # Compute actual next-day movement
    df["next_close"] = df["Close"].shift(-1)
    df["actual_up"] = (df["next_close"] > df["Close"]).astype(int)

    # Load AI predictions
    hist = pd.read_csv(HIST_PATH)
    hist = hist.sort_values("generated_at_utc", ascending=False).reset_index(drop=True)

    # Take last 30 predictions
    hist_30 = hist.head(30).copy()

    results = []
    win_count = 0

    for _, row in hist_30.iterrows():
        pred_for = str(row["predicted_for"])

        # Find matching actual close/next_close
        match = df[df["Date"] == pred_for]

        if match.empty:
            continue  # no market data for holiday yet

        actual_up = int(match["actual_up"].values[0])
        ai_up = 1 if row["prediction"].upper() == "UP" else 0

        win = (ai_up == actual_up)
        win_count += int(win)

        results.append({
            "prediction_for": pred_for,
            "ai_prediction": row["prediction"],
            "prob_up": round(row["prob_up"] * 100, 1),
            "actual_up": actual_up,
            "result": "WIN" if win else "LOSS"
        })

    total = len(results)
    win_ratio = (win_count / total * 100) if total > 0 else 0

    output = {
        "total_predictions": total,
        "wins": win_count,
        "loss": total - win_count,
        "win_ratio_percent": round(win_ratio, 2),
        "details": results
    }

    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

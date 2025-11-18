# src/predict_next_day.py

import json
from pathlib import Path
import datetime as dt

import pandas as pd
import joblib

DATA_PATH = Path("data") / "raw" / "nifty_daily.csv"
MODEL_PATH = Path("models") / "nifty_rf_model.pkl"
OUTPUT_DIR = Path("outputs")
PRED_JSON_PATH = OUTPUT_DIR / "nifty_prediction.json"
HIST_CSV_PATH = OUTPUT_DIR / "nifty_predictions_history.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_features_for_latest(df: pd.DataFrame, feature_cols):
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df.rename(
        columns={
            "Adj Close": "AdjClose",
            "Adj_Close": "AdjClose",
        },
        inplace=True,
    )

    for col in ["Open", "High", "Low", "Close", "AdjClose", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close", "Volume"])

    df["ret_1"] = df["Close"].pct_change()
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"].shift(1)

    for win in [5, 10, 20]:
        df[f"ma_{win}"] = df["Close"].rolling(win).mean()
        df[f"ret_{win}"] = df["Close"].pct_change(win)

    df["vol_mean_20"] = df["Volume"].rolling(20).mean()
    df["vol_norm"] = df["Volume"] / df["vol_mean_20"]

    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError("Not enough valid rows after cleaning to build features.")

    latest_row = df.iloc[-1]
    latest_features = latest_row[feature_cols].values.reshape(1, -1)
    latest_date = latest_row["Date"]

    return latest_features, pd.to_datetime(latest_date).date()


def next_weekday(d: dt.date) -> dt.date:
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run download_nifty.py first.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run train_model.py first.")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    X_latest, last_data_date = make_features_for_latest(df, feature_cols)

    proba_up = float(model.predict_proba(X_latest)[0, 1])
    pred_label = "UP" if proba_up >= 0.5 else "DOWN"

    predicted_for = next_weekday(last_data_date + dt.timedelta(days=1))

    now_utc = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    result = {
        "symbol": "^NSEI",
        "name": "Nifty 50 Index",
        "generated_at_utc": now_utc,
        "last_data_date": last_data_date.isoformat(),
        "predicted_for": predicted_for.isoformat(),
        "prediction": pred_label,
        "prob_up": proba_up,
        "prob_down": float(1.0 - proba_up),
    }

    with open(PRED_JSON_PATH, "w") as f:
        json.dump(result, f, indent=2)

    # Row for history
    row = {
        "generated_at_utc": now_utc,
        "last_data_date": last_data_date.isoformat(),
        "predicted_for": predicted_for.isoformat(),
        "prediction": pred_label,
        "prob_up": proba_up,
        "prob_down": 1.0 - proba_up,
    }

    # Keep at most one row per (last_data_date, predicted_for)
    if HIST_CSV_PATH.exists():
        hist_df = pd.read_csv(HIST_CSV_PATH)
        mask = ~(
            (hist_df["last_data_date"] == row["last_data_date"])
            & (hist_df["predicted_for"] == row["predicted_for"])
        )
        hist_df = hist_df[mask]
        hist_df = pd.concat([hist_df, pd.DataFrame([row])], ignore_index=True)
    else:
        hist_df = pd.DataFrame([row])

    hist_df.to_csv(HIST_CSV_PATH, index=False)

    print(f"Saved latest prediction to {PRED_JSON_PATH}")
    print(f"Updated prediction history at {HIST_CSV_PATH}")
    print(result)


if __name__ == "__main__":
    main()

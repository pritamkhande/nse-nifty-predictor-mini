# src/predict_next_day.py
#
# Live T+1 prediction for Nifty 50 using Close-only features and GradientBoosting.
# Uses best (THRESH, SEPARATION) tuned by calc_winratio.py, if available.

import json
from pathlib import Path
import datetime as dt

import numpy as np
import pandas as pd
import joblib

DATA_PATH = Path("data") / "raw" / "nifty_daily.csv"
MODEL_PATH = Path("models") / "nifty_rf_model.pkl"
OUTPUT_DIR = Path("outputs")
PRED_JSON_PATH = OUTPUT_DIR / "nifty_prediction.json"
HIST_CSV_PATH = OUTPUT_DIR / "nifty_predictions_history.csv"
BEST_THRESH_PATH = OUTPUT_DIR / "best_thresholds.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Same feature set as training / backtest
FEATURE_COLS = ["ret_1", "ret_5", "ret_10", "ret_20", "ma_5", "ma_10", "ma_20"]

# Default thresholds if tuning file not available
DEFAULT_THRESH = 0.70
DEFAULT_SEPARATION = 0.20


def load_thresholds():
    """Load tuned thresholds from best_thresholds.json, or return defaults."""
    if BEST_THRESH_PATH.exists():
        try:
            d = json.loads(BEST_THRESH_PATH.read_text(encoding="utf-8"))
            thresh = float(d.get("thresh", DEFAULT_THRESH))
            sep = float(d.get("separation", DEFAULT_SEPARATION))
            return thresh, sep
        except Exception:
            return DEFAULT_THRESH, DEFAULT_SEPARATION
    else:
        return DEFAULT_THRESH, DEFAULT_SEPARATION


def classify_with_confidence(prob_up: float, thresh: float, separation: float) -> str:
    prob_down = 1.0 - prob_up

    if prob_up >= thresh and (prob_up - prob_down) >= separation:
        return "UP"
    if prob_down >= thresh and (prob_down - prob_up) >= separation:
        return "DOWN"
    return "NO TRADE"


def make_features_for_latest(df: pd.DataFrame):
    """
    Build feature vector for the *last* available trading day using Close-only features.
    Uses the same definitions as in train_model.py.
    """
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df.rename(
        columns={"Adj Close": "AdjClose", "Adj_Close": "AdjClose"},
        inplace=True,
    )

    if "Close" not in df.columns:
        raise ValueError("nifty_daily.csv must contain a 'Close' column.")

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    if df.empty:
        raise ValueError("No valid Close prices available to build features.")

    # Need at least 21 days to compute ret_20 and ma_20
    if len(df) < 21:
        raise ValueError(
            f"Need at least 21 trading days to compute features, found only {len(df)}."
        )

    closes = df["Close"].values

    # Returns
    ret_1 = (closes[-1] / closes[-2]) - 1.0
    ret_5 = (closes[-1] / closes[-6]) - 1.0
    ret_10 = (closes[-1] / closes[-11]) - 1.0
    ret_20 = (closes[-1] / closes[-21]) - 1.0

    # Moving averages
    ma_5 = float(closes[-5:].mean())
    ma_10 = float(closes[-10:].mean())
    ma_20 = float(closes[-20:].mean())

    feat_dict = {
        "ret_1": ret_1,
        "ret_5": ret_5,
        "ret_10": ret_10,
        "ret_20": ret_20,
        "ma_5": ma_5,
        "ma_10": ma_10,
        "ma_20": ma_20,
    }

    x_vec = [feat_dict[name] for name in FEATURE_COLS]

    latest_date = pd.to_datetime(df.iloc[-1]["Date"]).date()
    return np.array(x_vec, dtype="float64").reshape(1, -1), latest_date


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

    X_latest, last_data_date = make_features_for_latest(df)

    prob_up = float(model.predict_proba(X_latest)[0, 1])
    prob_down = float(1.0 - prob_up)

    # Load tuned thresholds
    thresh, sep = load_thresholds()
    pred_label = classify_with_confidence(prob_up, thresh, sep)

    predicted_for = next_weekday(last_data_date + dt.timedelta(days=1))
    now_utc = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    result = {
        "symbol": "^NSEI",
        "name": "Nifty 50 Index",
        "generated_at_utc": now_utc,
        "last_data_date": last_data_date.isoformat(),
        "predicted_for": predicted_for.isoformat(),
        "prediction": pred_label,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "threshold_used": {
            "thresh": thresh,
            "separation": sep,
        },
    }

    with open(PRED_JSON_PATH, "w") as f:
        json.dump(result, f, indent=2)

    row = {
        "generated_at_utc": now_utc,
        "last_data_date": last_data_date.isoformat(),
        "predicted_for": predicted_for.isoformat(),
        "prediction": pred_label,
        "prob_up": prob_up,
        "prob_down": prob_down,
    }

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

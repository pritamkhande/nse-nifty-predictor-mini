# src/predict_next_day.py
#
# Live T+1 prediction for Nifty 50 using Close-only features.
# Features are computed directly for the last day from the last 20 closes.

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

THRESH = 0.70
SEPARATION = 0.20


def classify_with_confidence(prob_up: float) -> str:
    prob_down = 1.0 - prob_up

    if prob_up >= THRESH and (prob_up - prob_down) >= SEPARATION:
        return "UP"
    if prob_down >= THRESH and (prob_down - prob_up) >= SEPARATION:
        return "DOWN"
    return "NO TRADE"


def make_features_for_latest(df: pd.DataFrame, feature_cols):
    """
    Build feature vector for the *last* available trading day using Close-only features.
    This mirrors the features from train_model.py but only for the final row.
    """
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # Normalise adjusted close if present
    df.rename(
        columns={
            "Adj Close": "AdjClose",
            "Adj_Close": "AdjClose",
        },
        inplace=True,
    )

    # Ensure numeric
    for col in ["Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Require valid Close
    df = df.dropna(subset=["Close"])
    if df.empty:
        raise ValueError("No valid Close prices available to build features.")

    # Need at least 21 days to compute ret_20 and ma_20
    if len(df) < 21:
        raise ValueError(
            f"Need at least 21 trading days to compute features, found only {len(df)}."
        )

    closes = df["Close"].values
    n = len(closes)

    # Last day index t = n-1
    c_t = float(closes[-1])

    # Returns
    ret_1 = (closes[-1] / closes[-2]) - 1.0
    ret_5 = (closes[-1] / closes[-6]) - 1.0
    ret_10 = (closes[-1] / closes[-11]) - 1.0
    ret_20 = (closes[-1] / closes[-21]) - 1.0

    # Moving averages
    ma_5 = float(closes[-5:].mean())
    ma_10 = float(closes[-10:].mean())
    ma_20 = float(closes[-20:].mean())

    # Map of all available features
    feat_dict = {
        "ret_1": ret_1,
        "ret_5": ret_5,
        "ret_10": ret_10,
        "ret_20": ret_20,
        "ma_5": ma_5,
        "ma_10": ma_10,
        "ma_20": ma_20,
    }

    # Build feature vector in the same order as training
    try:
        x_vec = [feat_dict[name] for name in feature_cols]
    except KeyError as e:
        raise KeyError(
            f"Feature {e.args[0]} not found in feat_dict. "
            f"Training feature_cols must match predict-time features."
        )

    latest_date = pd.to_datetime(df.iloc[-1]["Date"]).date()
    return pd.np.array(x_vec, dtype="float64").reshape(1, -1), latest_date


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
    proba_down = float(1.0 - proba_up)

    pred_label = classify_with_confidence(proba_up)

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
        "prob_down": proba_down,
    }

    with open(PRED_JSON_PATH, "w") as f:
        json.dump(result, f, indent=2)

    row = {
        "generated_at_utc": now_utc,
        "last_data_date": last_data_date.isoformat(),
        "predicted_for": predicted_for.isoformat(),
        "prediction": pred_label,
        "prob_up": proba_up,
        "prob_down": proba_down,
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

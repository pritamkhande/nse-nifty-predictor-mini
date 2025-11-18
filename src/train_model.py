# src/train_model.py

from pathlib import Path

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

DATA_PATH = Path("data") / "raw" / "nifty_daily.csv"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "nifty_rf_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # Ensure numeric types (important if CSV has strings)
    for col in ["Open", "High", "Low", "Close", "AdjClose", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where Close or Volume is missing
    df = df.dropna(subset=["Close", "Volume"])

    # Basic features
    df["ret_1"] = df["Close"].pct_change()
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"].shift(1)

    # Moving averages & momentum
    for win in [5, 10, 20]:
        df[f"ma_{win}"] = df["Close"].rolling(win).mean()
        df[f"ret_{win}"] = df["Close"].pct_change(win)

    # Normalized volume
    df["vol_mean_20"] = df["Volume"].rolling(20).mean()
    df["vol_norm"] = df["Volume"] / df["vol_mean_20"]

    # Target: next-day direction
    df["target_up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Clean NaNs and remove last row (no target because of shift(-1))
    df = df.dropna().reset_index(drop=True)
    df = df.iloc[:-1, :]

    return df


def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run download_nifty.py first.")

    # Read data
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    # Normalise column names just in case
    df.rename(
        columns={
            "Adj Close": "AdjClose",
            "Adj_Close": "AdjClose",
        },
        inplace=True,
    )

    df_feat = make_features(df)

    feature_cols = [
        c
        for c in df_feat.columns
        if c
        not in [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "AdjClose",
            "Volume",
            "vol_mean_20",
            "target_up",
        ]
    ]

    X = df_feat[feature_cols].values
    y = df_feat["target_up"].values

    # Time-based split: last ~252 days as test if enough data
    if len(df_feat) > 300:
        X_train, X_test = X[:-252], X[-252:]
        y_train, y_test = y[:-252], y[-252:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    metrics = {
        "accuracy": float(acc),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_cols": feature_cols,
    }

    joblib.dump({"model": clf, "feature_cols": feature_cols}, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    train_model()

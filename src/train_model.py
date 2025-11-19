# src/train_model.py
#
# Train RandomForest model for Nifty 50 direction (next-day up/down)
# using the same feature engineering as calc_winratio.py.

from pathlib import Path
import json

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/raw/nifty_daily.csv")
MODEL_PATH = Path("models/nifty_rf_model.pkl")
OUTPUT_DIR = Path("outputs")
REPORT_PATH = OUTPUT_DIR / "model_report.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: must match calc_winratio.py logic."""
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # Normalize adjusted close name if present
    df.rename(
        columns={
            "Adj Close": "AdjClose",
            "Adj_Close": "AdjClose",
        },
        inplace=True,
    )

    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "AdjClose", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep rows with valid Close; Volume may be zero but not NaN
    df = df.dropna(subset=["Close"])

    # Features
    df["ret_1"] = df["Close"].pct_change()
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"].shift(1)

    for win in [5, 10, 20]:
        df[f"ma_{win}"] = df["Close"].rolling(win).mean()
        df[f"ret_{win}"] = df["Close"].pct_change(win)

    df["vol_mean_20"] = df["Volume"].rolling(20).mean()
    df["vol_norm"] = df["Volume"] / df["vol_mean_20"]

    # Target: next-day direction
    df["target_up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Drop rows with NaNs from rolling/pct_change/vol_mean_20
    df = df.dropna().reset_index(drop=True)

    # Last row has target referring to next day which may not exist; drop it
    if len(df) > 0:
        df = df.iloc[:-1, :].reset_index(drop=True)

    return df


def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run download_nifty.py first to create the CSV."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df_feat = make_features(df)

    n_samples = len(df_feat)
    if n_samples == 0:
        raise ValueError(
            "After feature engineering there are 0 rows.\n"
            "Check that nifty_daily.csv has valid 'Date', 'Open', 'High', 'Low', "
            "'Close', 'Volume' columns and enough history."
        )

    # Define input features (exclude raw price columns and target)
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

    # Time-aware split: first 80% train, last 20% test (no shuffle)
    if n_samples < 20:
        # Very small dataset: train on all, no test
        X_train, y_train = X, y
        X_test, y_test = None, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=False,
        )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    train_acc = float(clf.score(X_train, y_train))
    if X_test is not None:
        test_acc = float(clf.score(X_test, y_test))
    else:
        test_acc = None

    # Save model + feature list
    bundle = {
        "model": clf,
        "feature_cols": feature_cols,
    }
    joblib.dump(bundle, MODEL_PATH)

    report = {
        "n_samples": int(n_samples),
        "n_features": len(feature_cols),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Trained RandomForest on {n_samples} samples, {len(feature_cols)} features.")
    print(f"Train accuracy: {train_acc:.4f}")
    if test_acc is not None:
        print(f"Test accuracy:  {test_acc:.4f}")
    else:
        print("Test accuracy:  (not computed, dataset too small)")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    train_model()

# src/train_model.py
#
# Train RandomForest model for Nifty 50 direction (next-day up/down)
# using only Close-based features (no dependence on Volume / High / Low).

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
    """Feature engineering: Close-only features (returns, moving averages)."""
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # normalise adjusted close if present
    df.rename(
        columns={
            "Adj Close": "AdjClose",
            "Adj_Close": "AdjClose",
        },
        inplace=True,
    )

    # ensure numeric for price columns (if they exist)
    for col in ["Open", "High", "Low", "Close", "AdjClose"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # require valid Close only
    df = df.dropna(subset=["Close"])

    # if Open/High/Low missing, approximate by Close (for later display/backtest)
    for col in ["Open", "High", "Low"]:
        if col in df.columns:
            df[col] = df[col].fillna(df["Close"])

    # -------- features (Close-based only) --------
    df["ret_1"] = df["Close"].pct_change()

    for win in [5, 10, 20]:
        df[f"ma_{win}"] = df["Close"].rolling(win).mean()
        df[f"ret_{win}"] = df["Close"].pct_change(win)

    # target: next-day direction
    df["target_up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # drop rows with NaNs in features or target
    feature_cols = ["ret_1", "ma_5", "ma_10", "ma_20", "ret_5", "ret_10", "ret_20", "target_up"]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

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
            "Check that nifty_daily.csv has valid 'Date' and 'Close' columns "
            "and enough history (at least ~30-40 trading days)."
        )

    # input features (exclude raw prices and target)
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
            "target_up",
        ]
    ]

    X = df_feat[feature_cols].values
    y = df_feat["target_up"].values

    # time-aware split: first 80% train, last 20% test
    if n_samples < 20:
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

    train_acc = float(clf.score(X_train, y_train))
    if X_test is not None:
        test_acc = float(clf.score(X_test, y_test))
    else:
        test_acc = None

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

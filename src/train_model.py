# src/train_model.py
#
# Train GradientBoosting model for Nifty 50 direction (next-day up/down)
# using a fixed set of Close-based features.

from pathlib import Path
import json

import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/raw/nifty_daily.csv")
MODEL_PATH = Path("models/nifty_rf_model.pkl")  # keep same filename for compatibility
OUTPUT_DIR = Path("outputs")
REPORT_PATH = OUTPUT_DIR / "model_report.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Fixed feature set used everywhere (train, predict, backtest)
FEATURE_COLS = ["ret_1", "ret_5", "ret_10", "ret_20", "ma_5", "ma_10", "ma_20"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: Close-only, fixed feature set."""
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # Normalise adjusted close if present
    df.rename(
        columns={"Adj Close": "AdjClose", "Adj_Close": "AdjClose"},
        inplace=True,
    )

    # Ensure numeric for price columns
    for col in ["Open", "High", "Low", "Close", "AdjClose"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Require valid Close only
    df = df.dropna(subset=["Close"])

    # Optional: approximate missing OHLC by Close (useful for reporting)
    for col in ["Open", "High", "Low"]:
        if col in df.columns:
            df[col] = df[col].fillna(df["Close"])

    # Close-only features
    df["ret_1"] = df["Close"].pct_change()

    for win in [5, 10, 20]:
        df[f"ma_{win}"] = df["Close"].rolling(win).mean()
        df[f"ret_{win}"] = df["Close"].pct_change(win)

    # Target: next-day direction
    df["target_up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Drop rows with NaNs in any feature or target
    df = df.dropna(subset=FEATURE_COLS + ["target_up"]).reset_index(drop=True)

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

    X = df_feat[FEATURE_COLS].values
    y = df_feat["target_up"].values

    # Time-aware split: first 80% train, last 20% test
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

    # Gradient Boosting: sequential trees that focus on previous errors
    clf = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    train_acc = float(clf.score(X_train, y_train))
    if X_test is not None:
        test_acc = float(clf.score(X_test, y_test))
    else:
        test_acc = None

    bundle = {
        "model": clf,
        "feature_cols": FEATURE_COLS,  # saved for reference
    }
    joblib.dump(bundle, MODEL_PATH)

    report = {
        "n_samples": int(n_samples),
        "n_features": len(FEATURE_COLS),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Trained GradientBoosting on {n_samples} samples, {len(FEATURE_COLS)} features.")
    print(f"Train accuracy: {train_acc:.4f}")
    if test_acc is not None:
        print(f"Test accuracy:  {test_acc:.4f}")
    else:
        print("Test accuracy:  (not computed, dataset too small)")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    train_model()

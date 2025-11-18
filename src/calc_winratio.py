# src/calc_winratio.py
#
# Rolling backtest for the last 30 trading days.
# For each evaluated index i, we train on all data up to i-1,
# predict direction for move from day i to i+1,
# and compare with the actual next-day move.
# Stores OHLC and Win/Loss for each of those 30 trades.

from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = Path("data/raw/nifty_daily.csv")
OUT_PATH = Path("outputs/winratio_last_30.json")


def make_features(df: pd.DataFrame) -> pd.DataFrame:
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

    df["target_up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna().reset_index(drop=True)

    return df


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("nifty_daily.csv not found. Run download_nifty.py first.")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
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

    X_all = df_feat[feature_cols].values
    y_all = df_feat["target_up"].values
    dates_all = df_feat["Date"].dt.date.values

    open_all = df_feat["Open"].values
    high_all = df_feat["High"].values
    low_all = df_feat["Low"].values
    close_all = df_feat["Close"].values

    n = len(df_feat)
    if n < 300:
        raise ValueError("Not enough history to do a 30-day rolling backtest.")

    min_train = 252
    last_valid_idx = n - 2  # we will access idx+1
    if last_valid_idx <= min_train:
        raise ValueError("Not enough data after training window to backtest.")

    valid_indices = list(range(min_train, last_valid_idx + 1))
    eval_indices = valid_indices[-30:]

    results = []
    win_count = 0

    for idx in eval_indices:
        # train on up to idx-1 (strictly past)
        X_train = X_all[:idx, :]
        y_train = y_all[:idx]

        X_test = X_all[idx, :].reshape(1, -1)
        y_test = int(y_all[idx])

        as_of_date = dates_all[idx]
        pred_for_date = dates_all[idx + 1]

        o_asof = float(open_all[idx])
        h_asof = float(high_all[idx])
        l_asof = float(low_all[idx])
        c_asof = float(close_all[idx])
        c_next = float(close_all[idx + 1])

        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        proba_up = float(clf.predict_proba(X_test)[0, 1])
        pred_up = 1 if proba_up >= 0.5 else 0

        win = (pred_up == y_test)
        win_count += int(win)

        results.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "predicted_for": pred_for_date.isoformat(),
                "ai_prediction": "UP" if pred_up == 1 else "DOWN",
                "prob_up": round(proba_up * 100.0, 1),
                "actual_up": int(y_test),  # 1 = market went UP, 0 = DOWN
                "open_as_of": round(o_asof, 2),
                "high_as_of": round(h_asof, 2),
                "low_as_of": round(l_asof, 2),
                "close_as_of": round(c_asof, 2),
                "close_next": round(c_next, 2),
                "result": "WIN" if win else "LOSS",
            }
        )

    total = len(results)
    win_ratio = (win_count / total * 100.0) if total > 0 else 0.0

    output = {
        "mode": "rolling_backtest_last_30",
        "min_train_size": min_train,
        "total_predictions": total,
        "wins": win_count,
        "loss": total - win_count,
        "win_ratio_percent": round(win_ratio, 2),
        "details": results,
    }

    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

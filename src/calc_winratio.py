# src/calc_winratio.py
#
# Rolling backtest for the last 30 trading days.
# Uses Close-only features (same as train_model.py).

from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = Path("data/raw/nifty_daily.csv")
OUT_PATH = Path("outputs/winratio_last_30.json")

THRESH = 0.70
SEPARATION = 0.20


def classify_with_confidence(prob_up: float) -> str:
    prob_down = 1.0 - prob_up

    if prob_up >= THRESH and (prob_up - prob_down) >= SEPARATION:
        return "UP"
    if prob_down >= THRESH and (prob_down - prob_up) >= SEPARATION:
        return "DOWN"
    return "NO TRADE"


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

    for col in ["Open", "High", "Low", "Close", "AdjClose"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"])

    # approximate OHLC if missing (for reporting)
    for col in ["Open", "High", "Low"]:
        if col in df.columns:
            df[col] = df[col].fillna(df["Close"])

    # Close-only features
    df["ret_1"] = df["Close"].pct_change()
    for win in [5, 10, 20]:
        df[f"ma_{win}"] = df["Close"].rolling(win).mean()
        df[f"ret_{win}"] = df["Close"].pct_change(win)

    df["target_up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    feature_cols = ["ret_1", "ma_5", "ma_10", "ma_20", "ret_5", "ret_10", "ret_20", "target_up"]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

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
            "target_up",
        ]
    ]

    X_all = df_feat[feature_cols].values
    y_all = df_feat["target_up"].values
    dates_all = df_feat["Date"].dt.date.values

    open_all = df_feat.get("Open", df_feat["Close"]).values
    high_all = df_feat.get("High", df_feat["Close"]).values
    low_all = df_feat.get("Low", df_feat["Close"]).values
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
    loss_count = 0
    trade_count = 0

    for idx in eval_indices:
        X_train = X_all[:idx, :]
        y_train = y_all[:idx]

        X_test = X_all[idx, :].reshape(1, -1)
        y_test = int(y_all[idx])  # 1 = UP, 0 = DOWN

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
        proba_up_pct = round(proba_up * 100.0, 1)
        proba_down_pct = round(100.0 - proba_up_pct, 1)

        ai_label = classify_with_confidence(proba_up)

        if ai_label in ("UP", "DOWN"):
            trade_count += 1
            if (ai_label == "UP" and y_test == 1) or (ai_label == "DOWN" and y_test == 0):
                result_label = "WIN"
                win_count += 1
            else:
                result_label = "LOSS"
                loss_count += 1
        else:
            result_label = "NO TRADE"

        results.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "predicted_for": pred_for_date.isoformat(),
                "ai_prediction": ai_label,
                "prob_up": proba_up_pct,
                "prob_down": proba_down_pct,
                "actual_up": int(y_test),
                "open_as_of": round(o_asof, 2),
                "high_as_of": round(h_asof, 2),
                "low_as_of": round(l_asof, 2),
                "close_as_of": round(c_asof, 2),
                "close_next": round(c_next, 2),
                "result": result_label,
            }
        )

    total_predictions = len(results)
    trades = trade_count
    if trades > 0:
        win_ratio = win_count / trades * 100.0
    else:
        win_ratio = 0.0

    output = {
        "mode": "rolling_backtest_last_30",
        "min_train_size": min_train,
        "total_predictions": total_predictions,
        "effective_trades": trades,
        "wins": win_count,
        "loss": loss_count,
        "no_trade": total_predictions - trades,
        "win_ratio_percent": round(win_ratio, 2),
        "details": results,
    }

    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

# src/calc_winratio.py
#
# Rolling backtest using GradientBoosting and Close-only features.
# Optimised to run quickly on GitHub Actions:
#   - Uses a sliding 252-day training window (1 trading year).
#   - Only evaluates on the last 600 trading days (max).
#   - Uses a smaller model (200 trees) for backtest.
#
# Steps:
#   1) Run rolling backtest over recent history, recording probabilities + actuals.
#   2) Tune (THRESH, SEPARATION) grid to maximise win% with minimum trades.
#   3) Using best thresholds, compute summary for LAST 30 TRADING DAYS.
#   4) Save:
#        - outputs/winratio_last_30.json  (for website)
#        - outputs/best_thresholds.json   (used by predict_next_day.py)

from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

DATA_PATH = Path("data/raw/nifty_daily.csv")
OUT_PATH = Path("outputs/winratio_last_30.json")
BEST_THRESH_PATH = Path("outputs/best_thresholds.json")

FEATURE_COLS = ["ret_1", "ret_5", "ret_10", "ret_20", "ma_5", "ma_10", "ma_20"]

# Grid for tuning
THRESH_GRID = [0.60, 0.65, 0.70, 0.75]
SEP_GRID = [0.10, 0.15, 0.20, 0.25]
MIN_TRADES_FOR_TUNING = 30  # require at least this many trades on full history

# Performance parameters
TRAIN_WINDOW = 252       # days in each training window (approx 1 trading year)
MAX_EVAL_DAYS = 600      # evaluate at most this many days (≈ last 2–3 years)


def classify_with_confidence(prob_up: float, thresh: float, separation: float) -> str:
    prob_down = 1.0 - prob_up

    if prob_up >= thresh and (prob_up - prob_down) >= separation:
        return "UP"
    if prob_down >= thresh and (prob_down - prob_up) >= separation:
        return "DOWN"
    return "NO TRADE"


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df.rename(
        columns={"Adj Close": "AdjClose", "Adj_Close": "AdjClose"},
        inplace=True,
    )

    for col in ["Open", "High", "Low", "Close", "AdjClose"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"])

    # approximate OHLC if missing, for reporting
    for col in ["Open", "High", "Low"]:
        if col in df.columns:
            df[col] = df[col].fillna(df["Close"])

    # Close-only features
    df["ret_1"] = df["Close"].pct_change()
    for win in [5, 10, 20]:
        df[f"ma_{win}"] = df["Close"].rolling(win).mean()
        df[f"ret_{win}"] = df["Close"].pct_change(win)

    df["target_up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna(subset=FEATURE_COLS + ["target_up"]).reset_index(drop=True)

    return df


def compute_backtest_records(df_feat: pd.DataFrame):
    """
    Run rolling backtest with sliding 252-day training window.
    Only evaluate the last MAX_EVAL_DAYS days for speed.
    """
    X_all = df_feat[FEATURE_COLS].values
    y_all = df_feat["target_up"].values
    dates_all = df_feat["Date"].dt.date.values

    open_all = df_feat.get("Open", df_feat["Close"]).values
    high_all = df_feat.get("High", df_feat["Close"]).values
    low_all = df_feat.get("Low", df_feat["Close"]).values
    close_all = df_feat["Close"].values

    n = len(df_feat)
    if n < TRAIN_WINDOW + 50:
        raise ValueError(
            f"Not enough history to do a rolling backtest (need >= {TRAIN_WINDOW + 50} rows)."
        )

    # We will evaluate indices from start_eval_idx up to n-2 (because we look at idx+1).
    first_possible_idx = TRAIN_WINDOW
    last_eval_idx = n - 2  # idx+1 must exist

    # Limit to last MAX_EVAL_DAYS evaluation points
    if last_eval_idx - first_possible_idx + 1 > MAX_EVAL_DAYS:
        start_eval_idx = last_eval_idx - MAX_EVAL_DAYS + 1
        if start_eval_idx < first_possible_idx:
            start_eval_idx = first_possible_idx
    else:
        start_eval_idx = first_possible_idx

    valid_indices = list(range(start_eval_idx, last_eval_idx + 1))

    records = []

    for idx in valid_indices:
        # Sliding window: train only on the last TRAIN_WINDOW days before idx
        train_start = idx - TRAIN_WINDOW
        train_end = idx  # not included

        X_train = X_all[train_start:train_end, :]
        y_train = y_all[train_start:train_end]

        X_test = X_all[idx, :].reshape(1, -1)
        y_test = int(y_all[idx])

        as_of_date = dates_all[idx]
        pred_for_date = dates_all[idx + 1]

        o_asof = float(open_all[idx])
        h_asof = float(high_all[idx])
        l_asof = float(low_all[idx])
        c_asof = float(close_all[idx])
        c_next = float(close_all[idx + 1])

        clf = GradientBoostingClassifier(
            n_estimators=200,      # reduced for speed
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        prob_up = float(clf.predict_proba(X_test)[0, 1])  # 0–1
        prob_up_pct = round(prob_up * 100.0, 1)
        prob_down_pct = round(100.0 - prob_up_pct, 1)

        records.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "predicted_for": pred_for_date.isoformat(),
                "prob_up": prob_up_pct,        # percentage for display
                "prob_down": prob_down_pct,
                "actual_up": int(y_test),      # 1 = market UP next day, 0 = DOWN
                "open_as_of": round(o_asof, 2),
                "high_as_of": round(h_asof, 2),
                "low_as_of": round(l_asof, 2),
                "close_as_of": round(c_asof, 2),
                "close_next": round(c_next, 2),
            }
        )

    return records, TRAIN_WINDOW


def tune_thresholds(records):
    """Scan grid of (THRESH, SEPARATION) and pick combination with best win% and enough trades."""
    best = None

    for thresh in THRESH_GRID:
        for sep in SEP_GRID:
            wins = 0
            losses = 0
            trades = 0

            for r in records:
                prob_up = r["prob_up"] / 100.0  # percentage -> 0–1
                label = classify_with_confidence(prob_up, thresh, sep)

                if label in ("UP", "DOWN"):
                    trades += 1
                    if (label == "UP" and r["actual_up"] == 1) or (
                        label == "DOWN" and r["actual_up"] == 0
                    ):
                        wins += 1
                    else:
                        losses += 1

            if trades < MIN_TRADES_FOR_TUNING:
                continue

            win_ratio = (wins / trades * 100.0) if trades > 0 else 0.0

            if (
                best is None
                or win_ratio > best["win_ratio_percent"]
                or (
                    win_ratio == best["win_ratio_percent"]
                    and trades > best["trades"]
                )
            ):
                best = {
                    "thresh": thresh,
                    "separation": sep,
                    "wins": wins,
                    "loss": losses,
                    "trades": trades,
                    "win_ratio_percent": round(win_ratio, 2),
                }

    return best


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("nifty_daily.csv not found. Run download_nifty.py first.")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df_feat = make_features(df)

    records, min_train = compute_backtest_records(df_feat)
    total_rec = len(records)

    # ---- 1) Threshold tuning on all backtest records ----
    best_params = tune_thresholds(records)

    if best_params is None:
        # Fallback: default strict settings
        default_thresh = 0.70
        default_sep = 0.20
        wins = losses = trades = 0
        for r in records:
            prob_up = r["prob_up"] / 100.0
            label = classify_with_confidence(prob_up, default_thresh, default_sep)
            if label in ("UP", "DOWN"):
                trades += 1
                if (label == "UP" and r["actual_up"] == 1) or (
                    label == "DOWN" and r["actual_up"] == 0
                ):
                    wins += 1
                else:
                    losses += 1

        win_ratio = (wins / trades * 100.0) if trades > 0 else 0.0
        best_params = {
            "thresh": default_thresh,
            "separation": default_sep,
            "wins": wins,
            "loss": losses,
            "trades": trades,
            "win_ratio_percent": round(win_ratio, 2),
            "fallback": True,
        }
    else:
        best_params["fallback"] = False

    # Save best thresholds for predict_next_day.py
    BEST_THRESH_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_THRESH_PATH.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    # ---- 2) Last 30 trading days summary using tuned thresholds ----
    # records are chronological; take last 30
    records_last30 = records[-30:]
    wins30 = losses30 = trades30 = 0
    details_last30 = []

    for r in records_last30:
        prob_up = r["prob_up"] / 100.0
        ai_label = classify_with_confidence(
            prob_up, best_params["thresh"], best_params["separation"]
        )
        actual_label = "UP" if r["actual_up"] == 1 else "DOWN"

        if ai_label in ("UP", "DOWN"):
            trades30 += 1
            if (ai_label == "UP" and actual_label == "UP") or (
                ai_label == "DOWN" and actual_label == "DOWN"
            ):
                res_label = "WIN"
                wins30 += 1
            else:
                res_label = "LOSS"
                losses30 += 1
        else:
            res_label = "NO TRADE"

        details_last30.append(
            {
                "as_of_date": r["as_of_date"],
                "predicted_for": r["predicted_for"],
                "ai_prediction": ai_label,
                "prob_up": r["prob_up"],
                "prob_down": r["prob_down"],
                "actual_up": r["actual_up"],
                "open_as_of": r["open_as_of"],
                "high_as_of": r["high_as_of"],
                "low_as_of": r["low_as_of"],
                "close_as_of": r["close_as_of"],
                "close_next": r["close_next"],
                "result": res_label,
            }
        )

    total_predictions_30 = len(records_last30)
    if trades30 > 0:
        win_ratio_30 = wins30 / trades30 * 100.0
    else:
        win_ratio_30 = 0.0

    output = {
        "mode": "rolling_backtest_last_30",
        "feature_set": FEATURE_COLS,
        "train_window": TRAIN_WINDOW,
        "max_eval_days": MAX_EVAL_DAYS,
        "total_records_used_for_tuning": total_rec,
        "tuned_thresholds": best_params,
        "min_trades_for_tuning": MIN_TRADES_FOR_TUNING,
        # last 30 trading days summary:
        "total_predictions": total_predictions_30,
        "effective_trades": trades30,
        "wins": wins30,
        "loss": losses30,
        "no_trade": total_predictions_30 - trades30,
        "win_ratio_percent": round(win_ratio_30, 2),
        "details": details_last30,
    }

    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

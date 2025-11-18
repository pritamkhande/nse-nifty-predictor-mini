# src/calc_winratio.py
#
# NSE Prediction Ext-2
# - Read Nifty prediction history
# - Use Prob_UP / Prob_DOWN + Actual_UP
# - Self-correcting threshold over rolling 30 days
# - Export last-30-day detailed table to JSON for the site

import json
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# CONFIG – detect the history file from your repo structure
# -------------------------------------------------------------------

POSSIBLE_INPUTS = [
    Path("outputs/nifty_predictions_history.csv"),
    Path("nifty_predictions_history.csv"),
]

INPUT_CSV = None
for p in POSSIBLE_INPUTS:
    if p.exists():
        INPUT_CSV = p
        break

if INPUT_CSV is None:
    raise FileNotFoundError(
        "Could not find prediction history file. Checked:\n"
        + "\n".join(str(x) for x in POSSIBLE_INPUTS)
    )

OUTPUT_JSON = Path("outputs/winratio_last_30.json")
WINDOW_DAYS = 30
LAST_N_DAYS = 30

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _first_existing(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common variations to a standard schema, except Actual_UP."""
    df = df.copy()

    # ---------------- Date column detection ----------------
    date_col = _first_existing(df, ["Date", "date", "DATE"])
    if date_col is None:
        # heuristic: any column whose name contains "date", "day", "time", "timestamp"
        for c in df.columns:
            cl = str(c).lower()
            if "date" in cl or "day" in cl or "time" in cl or "timestamp" in cl:
                date_col = c
                break
    if date_col is None:
        # final fallback: treat first column as Date
        date_col = df.columns[0]

    df.rename(columns={date_col: "Date"}, inplace=True)

    # ---------------- OHLC (optional) ----------------
    col_map = {}
    for target, candidates in {
        "Open": ["Open", "open", "OPEN", "o"],
        "High": ["High", "high", "HIGH", "h"],
        "Low": ["Low", "low", "LOW", "l"],
        "Close": ["Close", "close", "CLOSE", "Adj Close", "adj_close", "c"],
    }.items():
        c = _first_existing(df, candidates)
        if c is not None:
            col_map[c] = target
    df.rename(columns=col_map, inplace=True)

    # ---------------- Prob_UP / Prob_DOWN ----------------
    prob_up_col = _first_existing(
        df,
        ["Prob_UP", "prob_up", "PROB_UP", "p_up", "prob_up_next", "prob_long"],
    )
    if prob_up_col is None:
        raise ValueError(
            "No Prob_UP column found. Please add one named "
            "Prob_UP/prob_up/p_up etc into nifty_predictions_history.csv."
        )
    df.rename(columns={prob_up_col: "Prob_UP"}, inplace=True)

    prob_down_col = _first_existing(
        df,
        ["Prob_DOWN", "prob_down", "PROB_DOWN", "p_down", "prob_short"]
    )
    if prob_down_col is not None:
        df.rename(columns={prob_down_col: "Prob_DOWN"}, inplace=True)
    else:
        df["Prob_DOWN"] = 1.0 - df["Prob_UP"]

    return df


def ensure_actual_up(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has Actual_UP (0/1).
    Priority:
      1) Existing explicit columns: Actual_UP/Actual/Target/Direction...
      2) Derive from Close_(t+1) > Close_t if Close exists.
    """
    df = df.copy()

    # Explicit label columns
    label_col = _first_existing(
        df,
        [
            "Actual_UP", "actual_up", "ACTUAL_UP",
            "Actual", "actual", "Target", "target",
            "Label", "label", "Direction", "direction",
        ],
    )

    if label_col is not None:
        series = df[label_col]
        if series.dtype == object:
            df["Actual_UP"] = series.astype(str).str.upper().map(
                {"UP": 1, "DOWN": 0, "1": 1, "0": 0, "TRUE": 1, "FALSE": 0}
            )
        else:
            df["Actual_UP"] = series.astype(int)

        if df["Actual_UP"].isna().any():
            raise ValueError(
                "Could not cleanly map Actual column to 0/1. "
                "Make sure values are UP/DOWN or 0/1."
            )
        return df

    # Derive from price if possible
    if "Close" not in df.columns:
        raise ValueError(
            "No Actual/Target/Direction column found, and no Close column "
            "to derive actual move from. Cannot compute Actual_UP."
        )

    df = df.sort_values("Date").reset_index(drop=True)
    df["Next_Close"] = df["Close"].shift(-1)
    df["Actual_UP"] = (df["Next_Close"] > df["Close"]).astype(int)
    # Drop last row where Next_Close is NaN (no next day price yet)
    df = df[df["Next_Close"].notna()].reset_index(drop=True)
    return df


# -------------------------------------------------------------------
# Dynamic threshold backtest
# -------------------------------------------------------------------

def dynamic_threshold_backtest(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Assumes df has columns:
      - Date (datetime-like)
      - Prob_UP (float 0–1)
      - Prob_DOWN (float 0–1)
      - Actual_UP (0/1)
    Returns df with added:
      - Best_Threshold
      - Pred_UP
      - Win
      - WinRatio_30d
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    thresholds = np.linspace(0.50, 0.90, 41)  # 0.50, 0.51, ..., 0.90

    best_thr_list = []
    pred_list = []
    win_list = []
    winratio_list = []

    for i in range(len(df)):
        if i == 0:
            thr = 0.5
        else:
            start_idx = max(0, i - window)
            hist = df.iloc[start_idx:i]

            if len(hist) < 3:
                thr = 0.5
            else:
                best_thr = 0.5
                best_score = -1.0
                for t in thresholds:
                    hist_pred = (hist["Prob_UP"] >= t).astype(int)
                    score = (hist_pred == hist["Actual_UP"]).mean()
                    if score > best_score:
                        best_score = score
                        best_thr = t
                thr = best_thr

        best_thr_list.append(thr)

        p_up = df.loc[i, "Prob_UP"]
        pred_up = int(p_up >= thr)
        pred_list.append(pred_up)

        actual_up = df.loc[i, "Actual_UP"]
        win = int(pred_up == actual_up)
        win_list.append(win)

        start_win = max(0, i - window + 1)
        winratio_30 = float(np.mean(win_list[start_win : i + 1]))
        winratio_list.append(winratio_30)

    df["Best_Threshold"] = best_thr_list
    df["Pred_UP"] = pred_list
    df["Win"] = win_list
    df["WinRatio_30d"] = winratio_list

    return df


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    print(f"Using prediction history file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    df = normalise_columns(df)
    df = ensure_actual_up(df)

    result_df = dynamic_threshold_backtest(df, window=WINDOW_DAYS)

    total_trades = len(result_df)
    total_wins = int(result_df["Win"].sum())
    overall_winratio = total_wins / total_trades if total_trades > 0 else 0.0

    last_30 = result_df.tail(LAST_N_DAYS).copy()
    last_30_trades = len(last_30)
    last_30_wins = int(last_30["Win"].sum())
    last_30_winratio = last_30_wins / last_30_trades if last_30_trades > 0 else 0.0

    # Prepare JSON payload for the site
    rows = []
    for _, r in last_30.iterrows():
        rows.append(
            {
                "date": r["Date"].strftime("%Y-%m-%d"),
                "open": float(r["Open"]) if "Open" in r else None,
                "high": float(r["High"]) if "High" in r else None,
                "low": float(r["Low"]) if "Low" in r else None,
                "close": float(r["Close"]) if "Close" in r else None,
                "prob_up": float(r["Prob_UP"]),
                "prob_down": float(r["Prob_DOWN"]),
                "best_threshold": float(r["Best_Threshold"]),
                "pred_up": int(r["Pred_UP"]),
                "actual_up": int(r["Actual_UP"]),
                "win": int(r["Win"]),
                "win_ratio_30d": float(r["WinRatio_30d"]),
            }
        )

    payload = {
        "summary": {
            "total_trades": int(total_trades),
            "total_wins": int(total_wins),
            "overall_winratio": overall_winratio,
            "last_30_trades": int(last_30_trades),
            "last_30_wins": int(last_30_wins),
            "last_30_winratio": last_30_winratio,
        },
        "rows": rows,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Console output for debugging
    print("\n===== NSE Prediction Ext-2 – Dynamic Win Ratio =====")
    print(f"Total trades      : {total_trades}")
    print(f"Total wins        : {total_wins}")
    print(f"Overall win ratio : {overall_winratio*100:.2f}%")
    print(f"Last {LAST_N_DAYS} trades: {last_30_trades}")
    print(f"Last {LAST_N_DAYS} wins  : {last_30_wins}")
    print(f"Last {LAST_N_DAYS} win % : {last_30_winratio*100:.2f}%")

    # Show compact table in logs
    display_cols = [
        c for c in [
            "Date", "Open", "High", "Low", "Close",
            "Prob_UP", "Prob_DOWN",
            "Best_Threshold", "Pred_UP", "Actual_UP",
            "Win", "WinRatio_30d"
        ] if c in last_30.columns
    ]
    print("\nLast 30 rows:")
    print(last_30[display_cols].to_string(index=False))

    print(f"\nSaved detailed JSON to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

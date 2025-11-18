# src/calc_winratio.py
#
# NSE Prediction Ext-2
# - Reads model predictions with Prob_UP (and optionally Prob_DOWN)
# - Computes Actual_UP if not already present
# - Uses a rolling 30-day window to find the best probability threshold
#   that maximizes past win ratio ("self-correcting" threshold)
# - Produces a last-30-days table with OHLC, probabilities, prediction,
#   actual, win/loss and rolling win ratio.
#
# Make sure your input CSV has at least:
#   Date, Close, Prob_UP
# and either:
#   Actual_UP (0/1)  OR
#   Next_Close (for computing Actual_UP via Next_Close > Close)
#
# You can extend this to include Open, High, Low, etc. for display.

import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------------------------
# CONFIG – adjust these as per your project structure
# -------------------------------------------------------------------
INPUT_CSV = Path("data/model_predictions_with_probs.csv")  # change to your file
OUTPUT_LAST30_CSV = Path("data/winratio_last30_days.csv")  # output summary
WINDOW_DAYS = 30  # rolling window length (days/trades)
LAST_N_DAYS = 30  # how many rows to show/save at the end


# -------------------------------------------------------------------
# Utility: infer Actual_UP if not present
# -------------------------------------------------------------------
def ensure_actual_up(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures df has an 'Actual_UP' column (0/1).
    Priority:
      1) Use existing 'Actual_UP' if present.
      2) Use 'Next_Close' and 'Close' if both exist: Actual_UP = Next_Close > Close.
      3) Use 'Target' (0/1) if exists.
      4) Use 'Is_Up' (0/1) if exists.
    Raises ValueError if none of these are available.
    """
    cols = set(df.columns)

    if "Actual_UP" in cols:
        return df

    if {"Close", "Next_Close"}.issubset(cols):
        df["Actual_UP"] = (df["Next_Close"] > df["Close"]).astype(int)
        return df

    if "Target" in cols:
        df["Actual_UP"] = df["Target"].astype(int)
        return df

    if "Is_Up" in cols:
        df["Actual_UP"] = df["Is_Up"].astype(int)
        return df

    raise ValueError(
        "Unable to infer 'Actual_UP'. Provide either: "
        "'Actual_UP' or ('Close' & 'Next_Close') or 'Target' or 'Is_Up' columns."
    )


# -------------------------------------------------------------------
# Core logic: rolling self-correcting threshold
# -------------------------------------------------------------------
def dynamic_threshold_backtest(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    df must contain:
      - 'Date' (datetime or string)
      - 'Prob_UP' (float 0–1)
      - 'Actual_UP' (0/1)

    This function:
      - For each row i, looks back up to `window` past rows [start_idx:i)
      - Searches a grid of thresholds to find the one giving highest accuracy
        on those past rows only (no future leakage)
      - Uses that threshold for row i
      - Records:
          Best_Threshold, Pred_UP, Win, WinRatio_30d (rolling)
    """
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # If Prob_DOWN not present, compute from Prob_UP
    if "Prob_DOWN" not in df.columns:
        df["Prob_DOWN"] = 1.0 - df["Prob_UP"]

    # Threshold grid – you can change the range/step
    thresholds = np.linspace(0.50, 0.90, 41)  # 0.50, 0.51, ..., 0.90

    best_thr_list = []
    pred_list = []
    win_list = []
    roll_winratio = []

    for i in range(len(df)):
        # Select threshold for this day using past data only
        if i == 0:
            # No history yet – default threshold
            thr = 0.5
        else:
            start_idx = max(0, i - window)
            hist = df.iloc[start_idx:i]

            if len(hist) < 3:
                # Not enough past data – use default
                thr = 0.5
            else:
                best_thr = 0.5
                best_score = -1.0

                # Grid-search best threshold on past data
                for t in thresholds:
                    hist_pred = (hist["Prob_UP"] >= t).astype(int)
                    score = (hist_pred == hist["Actual_UP"]).mean()

                    if score > best_score:
                        best_score = score
                        best_thr = t

                thr = best_thr

        best_thr_list.append(thr)

        # Prediction for this day using chosen threshold
        p_up = df.loc[i, "Prob_UP"]
        pred_up = int(p_up >= thr)
        pred_list.append(pred_up)

        # Win/loss
        actual_up = df.loc[i, "Actual_UP"]
        win = int(pred_up == actual_up)
        win_list.append(win)

        # Rolling win ratio over last `window` rows
        start_win = max(0, i - window + 1)
        winratio_30 = float(np.mean(win_list[start_win : i + 1]))
        roll_winratio.append(winratio_30)

    df["Best_Threshold"] = best_thr_list
    df["Pred_UP"] = pred_list
    df["Win"] = win_list
    df["WinRatio_30d"] = roll_winratio

    return df


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    # Load data
    df = pd.read_csv(INPUT_CSV)

    # Parse Date column
    if "Date" not in df.columns:
        raise ValueError("Input CSV must contain a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"])

    # Ensure Prob_UP exists
    if "Prob_UP" not in df.columns:
        raise ValueError("Input CSV must contain a 'Prob_UP' column (0–1).")

    # Ensure Actual_UP exists (or is inferred)
    df = ensure_actual_up(df)

    # Run dynamic threshold backtest
    result_df = dynamic_threshold_backtest(df, window=WINDOW_DAYS)

    # Compute overall stats
    total_trades = len(result_df)
    total_wins = int(result_df["Win"].sum())
    overall_winratio = total_wins / total_trades if total_trades > 0 else 0.0

    # Last N days summary table
    last_30 = result_df.tail(LAST_N_DAYS).copy()

    # Derive a simple Win/Loss text column
    last_30["Win_Loss"] = last_30["Win"].map({1: "WIN", 0: "LOSS"})

    # Compute stats for last N days
    last_30_trades = len(last_30)
    last_30_wins = int(last_30["Win"].sum())
    last_30_winratio = last_30_wins / last_30_trades if last_30_trades > 0 else 0.0

    # Select display columns, keep only ones that exist
    display_cols = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Prob_UP",
        "Prob_DOWN",
        "Best_Threshold",
        "Pred_UP",
        "Actual_UP",
        "Win_Loss",
        "WinRatio_30d",
    ]
    display_cols = [c for c in display_cols if c in last_30.columns]

    last_30_display = last_30[display_cols]

    # Save last-30-days summary
    OUTPUT_LAST30_CSV.parent.mkdir(parents=True, exist_ok=True)
    last_30_display.to_csv(OUTPUT_LAST30_CSV, index=False)

    # Print stats and table to console
    print("\n===== NSE Prediction Ext-2 – Dynamic 30-Day Win Ratio =====")
    print(f"Input file       : {INPUT_CSV}")
    print(f"Total trades     : {total_trades}")
    print(f"Total wins       : {total_wins}")
    print(f"Overall win ratio: {overall_winratio * 100:.2f}%")

    print(f"\nLast {LAST_N_DAYS} days:")
    print(f"Trades           : {last_30_trades}")
    print(f"Wins             : {last_30_wins}")
    print(f"Win ratio        : {last_30_winratio * 100:.2f}%")

    print(f"\nLast {LAST_N_DAYS} days detailed table (saved to {OUTPUT_LAST30_CSV}):\n")
    print(last_30_display.to_string(index=False))


if __name__ == "__main__":
    main()

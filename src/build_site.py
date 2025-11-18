from pathlib import Path
import pandas as pd
import html
import json

TEMPLATES_DIR = Path("templates")
SITE_DIR = Path("site")
DATA_PATH = Path("data") / "raw" / "nifty_daily.csv"
PRED_JSON_PATH = Path("outputs") / "nifty_prediction.json"
HIST_CSV_PATH = Path("outputs") / "nifty_predictions_history.csv"


def format_float(x, decimals=2):
    if x is None:
        return "-"
    return f"{x:.{decimals}f}"


def build_index():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run download_nifty.py.")
    if not PRED_JSON_PATH.exists():
        raise FileNotFoundError(f"{PRED_JSON_PATH} not found. Run predict_next_day.py.")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date")

    latest = df.iloc[-1]
    last_date = latest["Date"].date()
    latest_close = float(latest["Close"])
    latest_volume = int(latest["Volume"])

    if len(df) >= 2:
        prev_close = float(df.iloc[-2]["Close"])
    else:
        prev_close = latest_close

    change_abs = latest_close - prev_close
    change_pct = (change_abs / prev_close * 100.0) if prev_close != 0 else 0.0
    change_class = "up" if change_abs >= 0 else "down"
    sign = "+" if change_abs >= 0 else ""

    change_text = f"{sign}{format_float(change_abs)} ({sign}{format_float(change_pct)}%)"

    with open(PRED_JSON_PATH, "r") as f:
        pred = json.load(f)

    prediction = pred["prediction"]
    pred_badge_class = "up" if prediction.upper() == "UP" else "down"

    prob_up = float(pred["prob_up"])
    prob_down = float(pred["prob_down"])
    prob_up_percent = round(prob_up * 100.0, 1)

    generated_at_utc = pred["generated_at_utc"]
    predicted_for = pred["predicted_for"]

    if HIST_CSV_PATH.exists():
        hist_df = pd.read_csv(HIST_CSV_PATH)
        hist_df = hist_df.sort_values("generated_at_utc", ascending=False)
        recent = hist_df.head(7)
    else:
        recent = pd.DataFrame(columns=["generated_at_utc", "predicted_for", "prediction", "prob_up", "prob_down"])

    recent_rows_html = []
    for _, row in recent.iterrows():
        label = str(row["prediction"]).upper()
        pill_class = "pill-up" if label == "UP" else "pill-down"
        prob_up_r = round(float(row["prob_up"]) * 100.0, 1)
        prob_down_r = round(float(row["prob_down"]) * 100.0, 1)
        recent_rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row['generated_at_utc']))}</td>"
            f"<td>{html.escape(str(row['predicted_for']))}</td>"
            f"<td class=\"{pill_class}\">{html.escape(label)}</td>"
            f"<td>{prob_up_r}%</td>"
            f"<td>{prob_down_r}%</td>"
            "</tr>"
        )
    recent_rows_str = "\n".join(recent_rows_html) if recent_rows_html else "<tr><td colspan=\"5\">No history yet.</td></tr>"

    template = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")

    html_out = (
        template
        .replace("{{LATEST_CLOSE}}", format_float(latest_close))
        .replace("{{PREV_CLOSE}}", format_float(prev_close))
        .replace("{{LATEST_VOLUME}}", f"{latest_volume:,}")
        .replace("{{LAST_DATE}}", last_date.isoformat())
        .replace("{{CHANGE_TEXT}}", change_text)
        .replace("{{CHANGE_CLASS}}", change_class)
        .replace("{{PREDICTION}}", prediction.upper())
        .replace("{{PRED_BADGE_CLASS}}", pred_badge_class)
        .replace("{{PROB_UP_PERCENT}}", str(prob_up_percent))
        .replace("{{GENERATED_AT_UTC}}", generated_at_utc)
        .replace("{{PREDICTED_FOR}}", predicted_for)
        .replace("{{RECENT_ROWS}}", recent_rows_str)
    )

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "index.html").write_text(html_out, encoding="utf-8")
    print("Wrote site/index.html")


def build_history():
    if not HIST_CSV_PATH.exists():
        raise FileNotFoundError(f"{HIST_CSV_PATH} not found. Run predict_next_day.py at least once.")

    hist_df = pd.read_csv(HIST_CSV_PATH)
    hist_df = hist_df.sort_values("generated_at_utc", ascending=False).reset_index(drop=True)

    rows_html = []
    for idx, row in hist_df.iterrows():
        label = str(row["prediction"]).upper()
        pill_class = "pill-up" if label == "UP" else "pill-down"
        prob_up_r = round(float(row["prob_up"]) * 100.0, 1)
        prob_down_r = round(float(row["prob_down"]) * 100.0, 1)

        rows_html.append(
            "<tr>"
            f"<td>{idx + 1}</td>"
            f"<td>{html.escape(str(row['generated_at_utc']))}</td>"
            f"<td>{html.escape(str(row['last_data_date']))}</td>"
            f"<td>{html.escape(str(row['predicted_for']))}</td>"
            f"<td class=\"{pill_class}\">{html.escape(label)}</td>"
            f"<td>{prob_up_r}%</td>"
            f"<td>{prob_down_r}%</td>"
            "</tr>"
        )

    rows_str = "\n".join(rows_html) if rows_html else "<tr><td colspan=\"7\">No predictions yet.</td></tr>"

    template = (TEMPLATES_DIR / "history.html").read_text(encoding="utf-8")
    html_out = template.replace("{{HISTORY_ROWS}}", rows_str)

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "history.html").write_text(html_out, encoding="utf-8")
    print("Wrote site/history.html")


def main():
    build_index()
    build_history()


if __name__ == "__main__":
    main()

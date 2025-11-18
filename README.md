# Nifty 50 – AI Prediction Mini Site

This project:

- Downloads daily OHLCV data for the Nifty 50 index (`^NSEI`, via Yahoo Finance)
- Trains a simple RandomForest-based directional model (UP / DOWN for next trading day)
- Generates a JSON prediction + CSV history
- Builds a static mini-website with:
  - `site/index.html` – latest snapshot + next-day prediction
  - `site/history.html` – full prediction history

## Basic usage

```bash
pip install -r requirements.txt

python src/download_nifty.py
python src/train_model.py
python src/predict_next_day.py
python src/build_site.py
```

Open `site/index.html` in your browser.

To deploy via GitHub Pages, enable Pages in your repo and keep the provided
GitHub Actions workflow (`.github/workflows/pages.yml`).

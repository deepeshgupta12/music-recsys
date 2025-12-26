# Music Recommendation System (V0–V3)

This repo implements a local-first Music Recommendation System using:
- Data Science feature engineering
- Content-based + hybrid ranking
- LLM layer for tags/explanations (later)
- Synthetic users/events for personalization pipeline (later)
- Real telemetry integration (later)

## Dataset
Place the raw catalog CSV here (not committed):
`data/raw/spotify_2015_2025_85k.csv`

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
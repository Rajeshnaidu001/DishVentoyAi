# 🍕 DishVentory AI

> Intelligent restaurant management tool that predicts dish sales and calculates inventory needs using **Prophet** time-series forecasting.

[![GitHub Pages](https://img.shields.io/badge/Frontend-Live-brightgreen)](https://rajeshnaidu001.github.io/DishVentoyAi/)

## What it does

DishVentory AI helps restaurants:
- **Predict** which dishes will sell faster in the next 7 days using past order data
- **Map** predicted dishes to ingredients to calculate exact inventory needed
- **Reduce** food waste, avoid stockouts, and optimize kitchen operations

## Architecture

```
DishVentoyAi/
├── backend/          # Flask API + Prophet ML model
│   ├── app.py        # Main API server
│   ├── Procfile      # Render deployment config
│   └── requirements.txt
├── frontend/         # Static HTML/CSS/JS dashboard
│   ├── index.html
│   ├── script.js
│   └── styles.css
└── scripts/          # Offline analysis scripts
    └── forecast.py   # Full multi-pizza forecasting pipeline
```

## Quick Start

### Backend (Flask API)

```bash
cd backend
pip install -r requirements.txt
python app.py
# Server runs on http://127.0.0.1:5000
```

### Frontend

Open `frontend/index.html` in your browser, or use a local server:
```bash
cd frontend
python -m http.server 5500
# Open http://127.0.0.1:5500
```

> **Note:** For local development, update `BACKEND_URL` in `script.js` to `http://127.0.0.1:5000`

## Dataset

The model is trained on 1 year of pizza restaurant sales data. The dataset must include columns:
- `order_date` — date of order
- `pizza_id` — pizza identifier (e.g., `pepperoni_m`)
- `quantity` — number of units sold

## Tech Stack

- **ML Model:** Facebook Prophet (time-series forecasting)
- **Backend:** Python, Flask, Pandas, Matplotlib
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Render (backend), GitHub Pages (frontend)

## Deployment

- **Frontend:** Auto-deploys to GitHub Pages via GitHub Actions on push to `main`
- **Backend:** Deploy to [Render](https://render.com) — connect this repo, set root directory to `backend/`, and it will use the Procfile automatically

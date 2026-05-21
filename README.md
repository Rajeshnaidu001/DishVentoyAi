# 🍕 DishVentory AI

> Next-generation predictive restaurant operations dashboard. Predicts pepperoni pizza sales and scales raw ingredient inventory automatically using a high-fidelity **Prophet** forecasting engine.

[![GitHub Pages](https://img.shields.io/badge/Frontend-Live-blueviolet?style=for-the-badge&logo=githubpages)](https://rajeshnaidu001.github.io/DishVentoyAi/)
[![Render Backend](https://img.shields.io/badge/Backend-Active-brightgreen?style=for-the-badge&logo=render)](https://dishventory-ai-backend.onrender.com)
[![License](https://img.shields.io/badge/License-MIT-orange?style=for-the-badge)](LICENSE)

---

## ✨ Features

- **Prophet Time-Series Engine**: Deep seasonal prediction model calibrated for restaurant-grade weekly and holiday sales trends.
- **Dual-Mode Hybrid Resilience**:
  1. **Primary Layer**: Live Python backend (Flask API + Prophet model).
  2. **Smart Serverless Fallback**: A custom 100ms JavaScript forecasting client with an active **8-second timeout** to bypass Render free tier cold-starts (no hangs or freezes).
  3. **Local JSON cache**: Offline demo serving as Layer 3 fallback.
- **Premium SaaS Dashboard**: Beautiful responsive desktop layout locked to `100vh` viewports (Stripe/Datadog style). Cards scroll independently, ensuring the entire system fits **at once** without vertical scrollbars.
- **Automatic Recipe Propagation**: Seamlessly calculates and scales baking ingredients (flour, water, yeast, pepperoni, mozzarella, etc.) based on predicted pizza volume.
- **Dynamic Charts**: Powered by high-fidelity responsive Chart.js engines with neon gradients and custom dark glassmorphism.

---

## 🛠️ Architecture

```
DishVentoyAi/
├── backend/          # Python Flask Webserver + Prophet ML Pipeline
│   ├── app.py        # Flask API routing, Prophet training, and plotting
│   ├── Procfile      # Production WSGI server (Gunicorn) binding
│   └── requirements.txt
├── frontend/         # Premium Static Single-Page App (SPA)
│   ├── index.html    # Core markup with CSS/JS cache-busters
│   ├── script.js     # Progressive state managers, parsers, and charts
│   ├── styles.css    # Premium glassmorphic system & viewport constraints
│   └── demo_forecast.json # Offline simulated forecasting models
└── scripts/          # Offline Jupyter/CLI analysis utility
    └── forecast.py   # Multi-pizza batch forecasting scripts
```

---

## 🚀 Quick Start

### 1. Backend Server (Local Development)

Ensure you have Python 3.9+ installed, then set up the virtual environment:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
*The local Prophet API will automatically bind to `http://127.0.0.1:5000`.*

### 2. Frontend Dashboard

Simply run a local dev server inside the `frontend` folder:

```bash
cd frontend
python -m http.server 5500
```
Open **`http://127.0.0.1:5500`** in your browser. 
*Note: The frontend automatically detects local environments and routes requests to the local Flask server instead of the Render server.*

---

## 📊 Sales Dataset Format

DishVentory AI parses order records automatically (CSV or Excel `.xlsx`). The files must contain the following case-insensitive header column keys:
*   `order_date`: ISO timestamp or calendar date (e.g. `2015-01-01`).
*   `pizza_id`: Code name of the menu item (must match `pepperoni_m` for the default model).
*   `quantity`: Volume of units sold per order line (numeric).

---

## ⚙️ Premium Optimizations

### Viewport Height Lock & Scroll Limits
To achieve the premium, non-scrolling desktop experience, `.dashboard-grid` uses a locked layout height:
```css
@media (min-width: 1024px) {
  .dashboard-grid {
    height: calc(100vh - 180px);
    overflow: hidden;
  }
  .results-box {
    flex: 1;
    overflow-y: auto;
  }
}
```

### Chart Sizing Loop Resolution
To solve the infinite canvas resizing behavior of Chart.js in flexible structures, the canvas is isolated inside a styled bounding container:
```javascript
chartDiv.style.position = 'relative';
chartDiv.style.width = '100%';
chartDiv.style.height = '280px';
chartDiv.style.maxHeight = '280px';
```

---

## 📦 Deployment

### Frontend (GitHub Pages)
The client-side app auto-deploys to GitHub Pages on every push to `main` via the GitHub Actions CI pipeline.

### Backend (Render)
Deployed on Render using Python 3.10 and Gunicorn. 
*   **Root Directory**: `backend`
*   **Start Command**: `gunicorn app:app` (defined automatically via the `Procfile`)

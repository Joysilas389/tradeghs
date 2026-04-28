# GhanaQuant — EUR/USD Pair Trading Bot

A secure, modular, production-grade **statistical arbitrage (pair trading) system**
built for a Ghana-based investor earning in GHS. Trades EUR/USD and correlated
forex pairs using free data (yfinance), Kalman filter dynamic hedging,
HMM regime detection, and full walk-forward backtesting.

---

## Architecture

```
pair_trading/
├── backend/
│   ├── api/           Flask API + security middleware
│   ├── data/          DataHandler (yfinance) + SimulatedFeed + MT5 stub
│   ├── execution/     Paper trade execution + DB logging
│   ├── models/        SQLAlchemy ORM (trades, signals, backtests, metrics)
│   ├── risk/          RiskEngine (fixed fractional, Kelly, drawdown)
│   └── strategies/
│       ├── kalman_filter.py     Dynamic hedge ratio
│       ├── cointegration.py     Engle-Granger + ADF + Hurst
│       ├── regime_detector.py   HMM (mean-reverting vs trending)
│       ├── signal_engine.py     Z-score LONG/SHORT/EXIT/HOLD
│       └── pair_strategy.py     Master orchestrator
├── frontend/
│   ├── templates/index.html     Bootstrap 5 PWA dashboard
│   └── static/
│       ├── css/main.css         Dark trading terminal theme
│       └── js/
│           ├── api.js           Backend API calls
│           ├── charts.js        Chart.js equity + z-score gauge
│           └── dashboard.js     UI controller
├── config/settings.py           Central config + env vars
├── tests/test_strategy.py       31 unit tests (100% pass)
├── run.py                       Entry point
├── vercel.json                  Vercel deployment config
└── requirements.txt
```

---

## Signal Logic

| Z-Score | Position | Signal |
|---------|----------|--------|
| z > +2.0 | None | **SHORT** spread |
| z < -2.0 | None | **LONG** spread |
| z → 0 | Open | **EXIT** |
| \|z\| > 3.5 | Open | **STOP LOSS** (EXIT) |
| HMM = TRENDING | Any | **BLOCKED** |

---

## Quick Start (Local)

### 1. Clone / create project
```bash
git clone https://github.com/YOUR_USERNAME/pair-trading-bot.git
cd pair-trading-bot
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env — set FLASK_SECRET_KEY to a random string
# INITIAL_CAPITAL_GHS = your starting capital in GHS
```

### 5. Run
```bash
python run.py
```

Open **http://localhost:5000** in your browser.

---

## Run Tests
```bash
pytest tests/ -v
# Expected: 31 passed
```

---

## Usage

1. **Start Bot** — Enter symbol pair (e.g. `EURUSD=X` / `USDCHF=X`), choose period, click **Start Bot**
   - Downloads 5 years of free data from Yahoo Finance
   - Validates cointegration (Engle-Granger + ADF + Hurst + half-life)
   - Trains HMM on historical regimes
   - Runs Kalman filter for dynamic hedge ratio
   - Displays current signal, z-score, regime

2. **Run Backtest** — Tests strategy on out-of-sample data with transaction costs + slippage
   - Shows: Sharpe ratio, max drawdown, win rate, profit factor, equity curve
   - Toggle **Walk-Forward** for 5-window rolling validation

3. **Trade Log** — All paper trades logged to SQLite with P&L in both USD and GHS

---

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: GhanaQuant pair trading bot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/pair-trading-bot.git
git push -u origin main
```

---

## Deploy to Vercel

### Prerequisites
```bash
npm install -g vercel
```

### Deploy
```bash
vercel login
vercel --prod
```

### Set environment variables in Vercel dashboard
```
FLASK_SECRET_KEY   = your-random-secret-key
FLASK_ENV          = production
DATABASE_URL       = sqlite:////tmp/pair_trading.db
INITIAL_CAPITAL_GHS = 50000
MAX_DRAWDOWN_LIMIT  = 0.15
RISK_PER_TRADE      = 0.02
```

> **Note:** Vercel's serverless functions reset on each request — SQLite data is
> ephemeral on Vercel. For persistent storage, upgrade to PostgreSQL (swap
> `DATABASE_URL` to a Postgres connection string — SQLAlchemy handles it with
> zero code changes).

---

## Configuration (config/settings.py)

| Variable | Default | Description |
|----------|---------|-------------|
| `INITIAL_CAPITAL_GHS` | 50,000 | Starting capital in Ghanaian Cedi |
| `RISK_PER_TRADE` | 0.02 | 2% of capital risked per trade |
| `MAX_DRAWDOWN_LIMIT` | 0.15 | Bot stops at 15% drawdown |
| `ZSCORE_ENTRY` | 2.0 | Z-score threshold to enter trade |
| `ZSCORE_EXIT` | 0.0 | Z-score target for exit |
| `ZSCORE_STOP` | 3.5 | Stop-loss z-score |
| `HALFLIFE_MAX_DAYS` | 30 | Reject pairs with slow mean reversion |
| `TRANSACTION_COST_PCT` | 0.0002 | 2 bps per side |
| `SLIPPAGE_PCT` | 0.0001 | 1 bp slippage |

---

## Live Trading Upgrade Path

1. Open a forex account with **XM**, **Exness**, or **HotForex**
   (all accept MTN MoMo / GHS deposits and support MetaTrader 5)
2. Fill in `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER` in `.env`
3. The `MT5DataFeed` class in `backend/data/data_handler.py` is ready to connect
4. Replace `SimulatedFeed` with live `MT5DataFeed` in your strategy loop
5. Run **paper trading** (ExecutionService) for 30+ days before going live

---

## Security

- All secrets via environment variables (never hardcoded)
- Input sanitization on all API endpoints (symbol regex, period whitelist)
- Rate limiting: 200 req/min general, 5/min start-bot, 3/min backtest
- Security headers: `X-Content-Type-Options`, `X-Frame-Options`, `CSP`
- Internal errors never exposed to client
- SQL Injection: prevented by SQLAlchemy ORM parameterized queries
- XSS: all user content HTML-escaped in frontend JS (`escHtml()`)

---

## License

MIT — build on it, make it yours.

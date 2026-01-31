# NeuroQuant

**NeuroQuant — Personal Quant Research & Backtesting Platform**

A self-hosted algorithmic trading and backtesting platform with event-driven simulation, portfolio analytics, and a lightweight web UI.

## Key features
- Event-driven backtester with realistic fills, slippage & commissions  
- Built-in strategies: MA crossover, RSI, momentum, buy & hold  
- Portfolio analytics: Sharpe, Sortino, max drawdown, VaR, win rate  
- Data: yfinance + CSV import; local cache (Redis optional)  
- API + dashboard (FastAPI) and Docker-ready deployment

## Quick start
```bash
git clone https://github.com/MDhruv03/NeuroQuant.git
cd NeuroQuant
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
# or with Docker:
docker-compose up -d

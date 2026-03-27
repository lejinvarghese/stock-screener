# Quick Run Guide - Stock Screener

## First Time Setup

```bash
# 1. Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment (Python 3.11 required)
uv venv --python 3.11

# 3. Activate environment
source .venv/bin/activate

# 4. Install all dependencies
uv pip install -r requirements.txt

# 5. Setup environment variables
cp .env.example .env

# 6. Edit .env file with your credentials
nano .env  # or use your preferred editor
```

Required in `.env`:
- `TELEGRAM_TOKEN` and `TELEGRAM_ID` (optional, for Telegram notifications)
- `WEALTHSIMPLE_USERNAME` and `WEALTHSIMPLE_PASSWORD` (optional, for Wealthsimple integration)

## Running the App

### Standard Run
```bash
source .venv/bin/activate
python app.py
```

Opens on: http://localhost:5004

### Custom Port
```bash
source .venv/bin/activate
python app.py --port 8080
```

### Background Mode
```bash
source .venv/bin/activate
nohup python app.py --port 5004 > app.log 2>&1 &
echo $! > app.pid  # Save process ID
```

### Stop Background Process
```bash
# Using saved PID
kill $(cat app.pid)

# Or find and kill by name
pkill -f "python.*app.py"
```

### View Logs (Background Mode)
```bash
tail -f app.log
```

## Testing the App

### Check if Running
```bash
curl http://localhost:5004/
# Should return HTML
```

### Test Watchlist API
```bash
# View current watchlist
curl http://localhost:5004/watchlist/

# Add stocks
curl -X POST http://localhost:5004/watchlist/add \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

### Run Portfolio Analysis
```bash
curl -X POST http://localhost:5004/recommend_stocks/ \
  -H "Content-Type: application/json" \
  -d '{
    "threshold": 0.05,
    "budget": 10000,
    "method": "max_sharpe"
  }'
```

## Using Docker

### Build & Run
```bash
docker-compose up --build
```

Opens on: http://localhost:5000 (note: port 5000, not 5004)

### Stop Docker Container
```bash
docker-compose down
```

## Troubleshooting

### Python Version Error
```bash
# Check your Python version
python --version  # Should be 3.11+

# If not, specify explicitly
uv venv --python 3.11
```

### Module Not Found
```bash
# Reinstall dependencies
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Port Already in Use
```bash
# Use different port
python app.py --port 8080

# Or kill existing process
lsof -ti:5004 | xargs kill
```

### Can't Connect to Yahoo Finance
- This is normal - app will retry with fallback methods
- Rate limiting happens, app handles it automatically
- Just wait a few seconds between large batch requests

## Directory Structure

```
.
├── app.py                  # Main Flask application
├── core/                   # Core business logic
│   ├── analyzer.py        # Technical analysis
│   ├── optimizer.py       # Portfolio optimization
│   ├── watchlist.py       # Watchlist management
│   └── wealthsimple.py    # Wealthsimple integration
├── templates/             # HTML templates
│   └── index.html
├── data/                  # Data storage
│   ├── outputs/           # Generated charts
│   └── watchlist.db       # SQLite database
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables (create this)
```

## Next Steps

1. **Add Stocks**: Use web UI or curl to add stocks to watchlist
2. **Run Analysis**: Click "Analyze Portfolio" button or use API
3. **View Charts**: Generated charts saved to `data/outputs/`
4. **Adjust Parameters**:
   - `threshold`: Minimum position weight (default: 0.05 = 5%)
   - `budget`: Investment amount in dollars
   - `method`: "max_sharpe" or "semivariance"

## Performance Notes

- **Startup**: 2-3 seconds
- **Small watchlist** (5-10 stocks): 10-20 seconds analysis
- **Large watchlist** (50+ stocks): 2-3 minutes analysis
- Rate limiting from Yahoo Finance is normal and handled automatically

## Getting Help

- Check `SETUP_NOTES.md` for detailed testing results
- Check `README.md` for comprehensive documentation
- Issues? Check logs: `tail -f app.log`

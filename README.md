# 📊 AI-Powered Fundamental Analyst Agent

An intelligent investment analysis system that combines traditional fundamental analysis (DCF & multiples valuation) with AI-powered narrative generation to produce professional equity research reports.

## 🎯 Features

### Core Analysis
- **Discounted Cash Flow (DCF) Valuation**
  - Free Cash Flow to Firm (FCFF) methodology
  - Automatic CAGR calculation from historical data
  - Terminal growth sensitivity analysis (3%, 4%, 5%)
  - WACC calculation with market-based inputs

- **Multiples Valuation**
  - P/E, P/B, P/S, and EV/EBITDA comparisons
  - Peer company analysis (MSFT, GOOGL, META, NVDA)
  - Implied price calculations across multiple metrics

- **Blended Recommendation Engine**
  - 60% DCF / 40% Multiples weighting
  - BUY/HOLD/SELL recommendations with confidence levels
  - Threshold-based decision logic

### AI Integration
- **Claude AI Analysis**
  - Professional investment narrative generation
  - Executive summary with clear recommendations
  - Risk assessment and investment thesis
  - Data-driven insights from quantitative analysis

### Report Generation
- **JSON Export** - Structured data for system integration
- **PDF Reports** - Professional, client-ready investment reports with:
  - Executive summary with recommendation
  - DCF analysis tables
  - Terminal growth sensitivity analysis
  - Multiples comparison tables
  - AI-generated narrative analysis
  - Professional formatting and styling

### Data Management
- **Alpha Vantage Integration** - 5 years of financial statements
- **Smart Caching System** - 72-hour cache to avoid API rate limits
- **Automatic Fallbacks** - yfinance backup for real-time market data

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- Alpha Vantage API key (free tier: https://www.alphavantage.co/support/#api-key)
- Anthropic API key (https://console.anthropic.com/)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fundamental-analyst-agent.git
cd fundamental-analyst-agent
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
```

---

## 📖 Usage

### 1. Collect Financial Data

First, fetch historical financial statements from Alpha Vantage:

```bash
python src/Data_collector.py
```

This downloads 5 years of:
- Income statements
- Balance sheets
- Cash flow statements

**Note:** Takes ~30 seconds due to API rate limits.

### 2. Run Fundamental Analysis

Execute the complete analysis pipeline:

```bash
python src/fundamental_analyzer.py
```

**Output:**
- DCF valuation with intrinsic price
- Multiples analysis vs. peers
- Terminal growth sensitivity table
- BUY/HOLD/SELL recommendation

### 3. Generate AI-Powered Report

Create professional PDF and JSON reports:

```bash
python src/ai_agent.py
```

**Generated files:**
- `reports/analysis_AAPL_YYYYMMDD_HHMMSS.json` - Structured data
- `reports/report_AAPL_YYYYMMDD_HHMMSS.pdf` - Professional PDF report

---

## 📁 Project Structure

```
fundamental-analyst-agent/
├── src/
│   ├── Data_collector.py          # Fetch financial data from Alpha Vantage
│   ├── fundamental_analyzer.py     # Core valuation engine (DCF + multiples)
│   ├── ai_agent.py                 # AI narrative generation with Claude
│   └── report_generator.py         # JSON/PDF report generation
├── data/
│   ├── raw/                        # Financial statements (CSV)
│   └── cache/                      # API response cache
├── reports/                        # Generated JSON/PDF reports
├── .env                            # API keys (not in git)
├── .env.example                    # Template for environment variables
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🔧 Configuration

### Change Target Stock

Edit the ticker symbol in any script:

```python
analyzer = FundamentalAnalyzer("TSLA")  # Change from AAPL to TSLA
```

### Adjust DCF Parameters

Modify valuation assumptions in `fundamental_analyzer.py`:

```python
dcf_result = analyzer.dcf_valuation(
    forecast_years=5,          # Projection period
    terminal_growth=0.04,      # Long-term growth rate (4%)
    cap_forecast_growth=0.20,  # Maximum forecast growth cap (20%)
    default_erp=0.055,         # Equity risk premium (5.5%)
)
```

### Customize Peer Comparisons

Change peer companies in `multiples_valuation()`:

```python
multiples = analyzer.multiples_valuation(
    peers=["MSFT", "GOOGL", "META", "NVDA"]  # Customize peer list
)
```

### Adjust Recommendation Thresholds

Modify buy/sell thresholds in the recommendation logic:

```python
if blended_upside > 20:        # BUY threshold
    rec = "BUY"
elif blended_upside < -15:     # SELL threshold
    rec = "SELL"
else:
    rec = "HOLD"
```

---

## 📊 Methodology

### DCF Valuation Process

1. **Free Cash Flow Calculation**
   ```
   FCFF = EBIT(1 - Tax Rate) + D&A - CapEx - ΔNWC
   ```

2. **Growth Rate Estimation**
   - Uses 2-year CAGR of historical FCFF
   - Caps growth at 20% for realism

3. **Discount Rate (WACC)**
   ```
   WACC = (E/(D+E)) × Re + (D/(D+E)) × Rd × (1-T)
   Re = Rf + β × ERP (CAPM)
   ```

4. **Terminal Value**
   ```
   TV = FCFF_final × (1 + g) / (WACC - g)
   ```

5. **Enterprise to Equity Value**
   ```
   Equity Value = EV - Net Debt
   Intrinsic Price = Equity Value / Shares Outstanding
   ```

### Multiples Valuation

Calculates implied prices using peer median multiples:
- **P/E:** Price = Peer P/E × EPS
- **P/B:** Price = Peer P/B × Book Value per Share
- **P/S:** Price = Peer P/S × Sales per Share
- **EV/EBITDA:** Price = (Peer EV/EBITDA × EBITDA - Net Debt) / Shares

### Blended Recommendation

```
Blended Target = (DCF × 0.60) + (Multiples × 0.40)
Upside = (Target - Current Price) / Current Price
```

**Decision Rules:**
- **BUY:** Upside > 20%
- **HOLD:** -15% < Upside < 20%
- **SELL:** Upside < -15%

---

## 🔍 Example Output

### Terminal Output
```
======================================================================
DCF VALUATION
======================================================================
Forecast Growth (CAGR):        8.92%
WACC:                          10.10%
DCF Intrinsic Price:           $145.23
Current Price:                 $255.53
DCF Upside:                    -43.16%

======================================================================
MULTIPLES VALUATION
======================================================================
Current Multiples (AAPL):
  P/E: 33.11
  P/B: 50.94
  P/S: 8.76
  EV/EBITDA: 26.25

Implied Prices by Multiple:
  P/E: $248.67
  P/B: $49.41
  P/S: $311.46
  EV/EBITDA: $205.12

Multiples Average Price:       $203.66

======================================================================
BLENDED RECOMMENDATION (60% DCF / 40% Multiples)
======================================================================
Blended Target Price:          $168.60
Current Price:                 $255.53
Blended Upside:                -34.03%

======================================================================
FINAL RECOMMENDATION: SELL (High Confidence)
======================================================================
```

### JSON Structure
```json
{
  "ticker": "AAPL",
  "report_date": "2024-01-15T14:30:00",
  "current_price": 255.53,
  "recommendation": "SELL",
  "confidence": "High",
  "blended_target": 168.60,
  "upside_potential": -34.03,
  "dcf_valuation": { ... },
  "multiples_valuation": { ... },
  "sensitivity": [ ... ],
  "ai_report": "..."
}
```

---

## ⚙️ API Rate Limits

### Alpha Vantage (Free Tier)
- **5 calls per minute**
- **25 calls per day**
- **Solution:** 72-hour caching system

### Anthropic Claude
- Pay-per-use pricing
- ~$0.01-0.05 per report
- **Solution:** Cache API responses when possible

### Yahoo Finance (yfinance)
- Strict rate limiting
- **Solution:** 
  - Exponential backoff retry logic
  - 72-hour market data cache
  - Alpha Vantage primary, yfinance fallback

---

## 🛠️ Troubleshooting

### "Too Many Requests" Error
**Problem:** API rate limit exceeded

**Solution:**
```bash
# Wait 12-24 hours for limits to reset
# Or use cached data
python src/fundamental_analyzer.py  # Uses cache automatically
```

### "FCFF Data Insufficient"
**Problem:** Not enough historical data

**Solution:**
```bash
# Re-run data collector for fresh data
python src/Data_collector.py

# Or adjust CAGR years in code
cagr_years = 2  # Use 2 years instead of 5
```

### PDF Generation Fails
**Problem:** ReportLab formatting error

**Solution:**
```bash
# Check ReportLab is installed
pip install reportlab pillow --upgrade

# Check reports/ folder permissions
mkdir -p reports
chmod 755 reports
```

### Claude API Error
**Problem:** Invalid API key or insufficient credits

**Solution:**
1. Verify API key in `.env`
2. Check credits at https://console.anthropic.com/
3. Add payment method if needed

---

## 📈 Limitations & Assumptions

### Data Quality
- Relies on Alpha Vantage and yfinance data accuracy
- Historical data may have gaps or errors
- Real-time prices have 15-minute delay on free tier

### Valuation Assumptions
- CAPM holds and beta is stable
- Terminal growth < GDP growth
- Market efficiency in peer comparisons
- Historical CAGR predicts future growth

### Model Limitations
- No consideration of qualitative factors
- Binary recommendation system (no partial positions)
- Assumes rational market behavior
- No scenario analysis beyond terminal growth sensitivity

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## ⚠️ Disclaimer

**This tool is for educational and informational purposes only.**

- NOT financial advice
- NOT a recommendation to buy, sell, or hold securities
- Past performance does not guarantee future results
- All investments involve risk, including loss of principal
- Consult a qualified financial advisor before making investment decisions

The authors and contributors are not responsible for any financial losses incurred from using this tool.

---

## 🙏 Acknowledgments

- **Alpha Vantage** - Financial data API
- **Anthropic Claude** - AI analysis engine
- **yfinance** - Market data fallback
- **ReportLab** - PDF generation

---

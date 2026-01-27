import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime, timedelta
import yfinance as yf
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Cache configuration
CACHE_PATH = Path("data/cache")
CACHE_PATH.mkdir(parents=True, exist_ok=True)
MARKET_DATA_CACHE = CACHE_PATH / "market_data_cache.csv"
PEER_MULTIPLES_CACHE = CACHE_PATH / "peer_multiples_cache.csv"
CACHE_EXPIRY_HOURS = 72


def load_market_data_cache(ticker_symbol):
    """Load cached market data if it exists and is fresh (< 72 hours old)."""
    if not MARKET_DATA_CACHE.exists():
        return None
    
    try:
        df = pd.read_csv(MARKET_DATA_CACHE)
        ticker_data = df[df['ticker'] == ticker_symbol]
        
        if ticker_data.empty:
            return None
        
        cached_entry = ticker_data.iloc[-1]
        timestamp = pd.to_datetime(cached_entry['timestamp'])
        age = datetime.now() - timestamp
        
        if age < timedelta(hours=CACHE_EXPIRY_HOURS):
            print(f"✓ Using cached market data for {ticker_symbol} (age: {age.days}d {age.seconds//3600}h)")
            return {
                'currentPrice': cached_entry['currentPrice'],
                'beta': cached_entry['beta'],
                'riskFreeRate': cached_entry['riskFreeRate'],
                'sharesOutstanding': cached_entry['sharesOutstanding'],
            }
        else:
            print(f"⟳ Market data cache expired. Refreshing {ticker_symbol}...")
            return None
    except Exception as e:
        print(f"Error reading cache: {e}")
        return None


def save_market_data_cache(ticker_symbol, current_price, beta, risk_free_rate, shares_outstanding):
    """Save market data to cache with timestamp."""
    try:
        # Load existing cache or create new
        if MARKET_DATA_CACHE.exists():
            df = pd.read_csv(MARKET_DATA_CACHE)
            # Remove old entry for this ticker if exists
            df = df[df['ticker'] != ticker_symbol]
        else:
            df = pd.DataFrame()
        
        # Add new entry
        new_entry = pd.DataFrame([{
            'ticker': ticker_symbol,
            'currentPrice': current_price,
            'beta': beta,
            'riskFreeRate': risk_free_rate,
            'sharesOutstanding': shares_outstanding,
            'timestamp': datetime.now().isoformat(),
        }])
        
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(MARKET_DATA_CACHE, index=False)
        print(f"✓ Cached market data for {ticker_symbol}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")


def load_peer_multiples_cache():
    """Load cached peer multiples if fresh."""
    if not PEER_MULTIPLES_CACHE.exists():
        return None

    try:
        df = pd.read_csv(PEER_MULTIPLES_CACHE)
        if 'timestamp' not in df.columns:
            return None

        timestamp = pd.to_datetime(df['timestamp'].iloc[0])
        age = datetime.now() - timestamp

        if age < timedelta(hours=CACHE_EXPIRY_HOURS):
            print(f"✓ Using cached peer multiples (age: {age.days}d {age.seconds//3600}h)")
            return df.drop(columns=['timestamp'])
        else:
            print(f"⟳ Peer multiples cache expired. Will refresh...")
            return None
    except Exception as e:
        print(f"Error reading peer cache: {e}")
        return None


def save_peer_multiples_cache(df):
    """Save peer multiples to cache with timestamp."""
    try:
        df = df.copy()
        df['timestamp'] = datetime.now().isoformat()
        df.to_csv(PEER_MULTIPLES_CACHE, index=False)
        print(f"✓ Cached peer multiples")
    except Exception as e:
        print(f"Warning: Could not save peer cache: {e}")


def safe_yfinance_call(func, max_retries=3, delay=5):
    """Retry yfinance calls with exponential backoff to handle rate limits."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)
                print(f"Yahoo Finance rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e


def fetch_market_data_alpha_vantage(ticker_symbol):
    """
    Fetch current price, beta, and shares outstanding from Alpha Vantage.
    Uses GLOBAL_QUOTE for price and OVERVIEW for beta/shares.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY not found in .env file")

    result = {
        'currentPrice': None,
        'beta': None,
        'sharesOutstanding': None,
    }

    # Fetch current price from GLOBAL_QUOTE
    try:
        quote_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker_symbol}&apikey={api_key}"
        response = requests.get(quote_url)
        data = response.json()

        if "Global Quote" in data and "05. price" in data["Global Quote"]:
            result['currentPrice'] = float(data["Global Quote"]["05. price"])
    except Exception as e:
        print(f"Alpha Vantage GLOBAL_QUOTE failed: {e}")

    time.sleep(12)  # Rate limit: 5 calls per minute for free tier

    # Fetch beta and shares from OVERVIEW
    try:
        overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker_symbol}&apikey={api_key}"
        response = requests.get(overview_url)
        data = response.json()

        if "Beta" in data and data["Beta"] != "None":
            result['beta'] = float(data["Beta"])
        else:
            result['beta'] = 1.0  # Default beta

        if "SharesOutstanding" in data and data["SharesOutstanding"] != "None":
            result['sharesOutstanding'] = float(data["SharesOutstanding"])
    except Exception as e:
        print(f"Alpha Vantage OVERVIEW failed: {e}")

    return result


class FundamentalAnalyzer:

    def __init__(self, ticker_symbol="AAPL", data_path="data/raw"):
        self.ticker_symbol = ticker_symbol
        self._ticker = None  # Lazy-load yfinance ticker only when needed

        self.data_path = Path(data_path)
        self.income = pd.read_csv(self.data_path / "income_statement.csv", index_col=0)
        self.balance = pd.read_csv(self.data_path / "balance_sheet.csv", index_col=0)
        self.cashflow = pd.read_csv(self.data_path / "cash_flow.csv", index_col=0)

        # Load cached market data (no auto-refresh on init)
        self.market_data_cache = load_market_data_cache(ticker_symbol)

    @property
    def ticker(self):
        """Lazy-load yfinance ticker only when actually needed."""
        if self._ticker is None:
            self._ticker = yf.Ticker(self.ticker_symbol)
        return self._ticker

    def _get_first_available_row(self, df, labels):
        for label in labels:
            if label in df.index:
                return df.loc[label]
        raise KeyError(f"Missing rows: {labels}")

    def _equity(self):
        total_assets = self._get_first_available_row(
            self.balance, ["Total Assets"]
        )

        total_liab = self._get_first_available_row(
            self.balance,
            [
                "Total Liabilities Net Minority Interest",
                "Total Liabilities",
                "Total Liab",
            ],
        )

        equity = total_assets - total_liab
        return equity.replace(0, pd.NA)

    def profitability_ratios(self):
        net_income = self.income.loc["Net Income"]
        revenue = self.income.loc["Total Revenue"]
        equity = self._equity()

        return pd.DataFrame({
            "Net Margin": net_income / revenue,
            "ROE": net_income / equity,
        })

    def leverage_ratios(self):
        total_liab = self._get_first_available_row(
            self.balance,
            [
                "Total Liabilities Net Minority Interest",
                "Total Liabilities",
                "Total Liab",
            ],
        )

        equity = self._equity()

        return pd.DataFrame({
            "Debt to Equity": total_liab / equity,
        })

    def growth_rates(self):
        revenue = self.income.loc["Total Revenue"]
        net_income = self.income.loc["Net Income"]

        return pd.DataFrame({
            "Revenue Growth": revenue.pct_change(),
            "Net Income Growth": net_income.pct_change(),
        })

    def free_cash_flow_fcff(self):
        """
        Unlevered Free Cash Flow (FCFF):
        FCFF = EBIT(1 - TaxRate) + D&A - Capex - ΔNWC
        """
        ebit = self._get_first_available_row(
            self.income, ["EBIT", "Operating Income", "Ebit"]
        )

        tax_expense = self._get_first_available_row(
            self.income, ["Tax Provision", "Income Tax Expense"]
        )
        pretax_income = self._get_first_available_row(
            self.income, ["Pretax Income", "Income Before Tax"]
        )
        tax_rate = tax_expense.iloc[-1] / pretax_income.iloc[-1]
        tax_rate = float(max(0.0, min(0.5, tax_rate)))

        da = self._get_first_available_row(
            self.cashflow,
            [
                "Depreciation And Amortization",
                "Depreciation",
                "Depreciation & Amortization",
            ],
        )

        capex = self._get_first_available_row(
            self.cashflow, ["Capital Expenditures", "Capital Expenditure"]
        )

        current_assets = self._get_first_available_row(
            self.balance, ["Total Current Assets", "Current Assets"]
        )
        cash = self._get_first_available_row(
            self.balance,
            [
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Short Term Investments",
            ],
        )
        current_liabilities = self._get_first_available_row(
            self.balance, ["Total Current Liabilities", "Current Liabilities"]
        )
        short_term_debt = self._get_first_available_row(
            self.balance,
            [
                "Short Long Term Debt",
                "Short Term Debt",
                "Current Debt",
                "Current Debt And Capital Lease Obligation",
            ],
        )

        nwc = (current_assets - cash) - (current_liabilities - short_term_debt)
        delta_nwc = nwc.diff()

        fcff = (ebit * (1 - tax_rate)) + da - capex - delta_nwc
        return fcff

    def fcf_cagr(self, years=5):
        """Compute CAGR of Free Cash Flow over the last `years`."""
        fcf = self.free_cash_flow_fcff().dropna()

        if len(fcf) < years + 1:
            raise ValueError("Not enough historical FCF data to compute CAGR")

        fcf_start = fcf.iloc[-(years + 1)]
        fcf_end = fcf.iloc[-1]

        if fcf_start <= 0:
            raise ValueError("FCF start value is non-positive; CAGR not meaningful")

        cagr = (fcf_end / fcf_start) ** (1 / years) - 1
        return float(cagr)
    
    def risk_free_rate_alpha_vantage(self):
        """Fetch 10-year US Treasury yield from Alpha Vantage."""
        import requests
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not found in .env file")
        
        url = f"https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=monthly&maturity=10year&apikey={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if "data" in data and len(data["data"]) > 0:
            # Get the most recent yield (already in percentage, convert to decimal)
            latest_yield = float(data["data"][0]["value"]) / 100
            return latest_yield
        else:
            raise ValueError("Could not fetch Treasury rate from Alpha Vantage")
    
    def risk_free_rate_yfinance(self):
        """Fallback: Fetch 10-year US Treasury yield from Yahoo Finance (^TNX)."""
        def fetch_rate():
            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="10d")
            
            if hist.empty:
                raise ValueError("Could not fetch ^TNX data from Yahoo Finance")
            
            x = float(hist["Close"].dropna().iloc[-1])
            
            if x > 20:
                return x / 1000
            elif x > 1:
                return x / 100
            else:
                return x
        
        return safe_yfinance_call(fetch_rate)
    
    def risk_free_rate(self):
        """Get risk-free rate from cache or fetch fresh. Falls back to default if APIs fail."""
        if self.market_data_cache and self.market_data_cache.get('riskFreeRate'):
            return self.market_data_cache['riskFreeRate']

        # Try Alpha Vantage first
        try:
            return self.risk_free_rate_alpha_vantage()
        except Exception as e:
            print(f"Alpha Vantage Treasury fetch failed ({e}), trying yfinance...")

        # Try yfinance
        try:
            return self.risk_free_rate_yfinance()
        except Exception as e:
            print(f"yfinance Treasury fetch failed ({e}), using default rate...")

        # Default to a reasonable 10-year Treasury rate
        default_rate = 0.0425  # ~4.25% as of 2024
        print(f"Using default risk-free rate: {default_rate:.2%}")
        return default_rate

    def equity_risk_premium(self, default_erp=0.055):
        """Equity Risk Premium (market assumption, e.g., 5.5%)."""
        return float(default_erp)

    def cost_of_equity_capm(self, default_erp=0.055):
        """CAPM: Re = Rf + beta * ERP"""
        if self.market_data_cache and 'beta' in self.market_data_cache:
            beta = self.market_data_cache['beta']
        else:
            beta = safe_yfinance_call(lambda: self.ticker.info.get("beta", 1.0))
        rf = self.risk_free_rate()
        erp = self.equity_risk_premium(default_erp=default_erp)
        return float(rf + beta * erp)

    def effective_tax_rate_latest(self):
        """Effective tax rate = Tax expense / Pretax income (latest year)."""
        tax_expense = self._get_first_available_row(
            self.income, ["Tax Provision", "Income Tax Expense"]
        )
        pretax_income = self._get_first_available_row(
            self.income, ["Pretax Income", "Income Before Tax"]
        )

        tax = float(tax_expense.iloc[-1])
        ebt = float(pretax_income.iloc[-1])

        if ebt == 0:
            return 0.21

        t = tax / ebt
        return float(max(0.0, min(0.5, t)))

    def total_debt_latest(self):
        """Latest total debt."""
        if "Total Debt" in self.balance.index:
            return float(self.balance.loc["Total Debt"].iloc[-1])

        lt_debt = self._get_first_available_row(
            self.balance, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]
        ).iloc[-1]

        st_debt = self._get_first_available_row(
            self.balance,
            [
                "Short Long Term Debt",
                "Short Term Debt",
                "Current Debt",
                "Current Debt And Capital Lease Obligation",
            ],
        ).iloc[-1]

        return float(lt_debt + st_debt)
    
    def cost_of_debt(self, use_market_approach=True, verbose=False):
        """
        Calculate cost of debt using multiple approaches with fallbacks.
        
        Approach 1: Interest Expense / Average Debt
        Approach 2: Market-based estimate using D/E ratio
        Approach 3: Risk-free rate + credit spread
        """
        # Approach 1: Direct interest expense
        try:
            interest_series = self._get_first_available_row(
                self.income,
                [
                    "Interest Expense",
                    "Interest Expense Non Operating",
                    "Net Non Operating Interest Income Expense",
                    "Net Interest Income",
                ],
            )

            ie = interest_series.iloc[-1]

            if not pd.isna(ie) and ie != 0:
                ie = float(abs(ie))

                if "Total Debt" in self.balance.index:
                    debt_series = self.balance.loc["Total Debt"]
                else:
                    lt = self._get_first_available_row(
                        self.balance,
                        ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
                    )
                    st = self._get_first_available_row(
                        self.balance,
                        [
                            "Short Long Term Debt",
                            "Short Term Debt",
                            "Current Debt",
                            "Current Debt And Capital Lease Obligation",
                        ],
                    )
                    debt_series = lt + st

                if len(debt_series) >= 2:
                    avg_debt = float((debt_series.iloc[-1] + debt_series.iloc[-2]) / 2)
                else:
                    avg_debt = float(debt_series.iloc[-1])

                if avg_debt > 0:
                    rd = ie / avg_debt
                    if 0.005 <= rd <= 0.15:
                        if verbose:
                            print(f"Cost of Debt (Approach 1 - Direct): {rd:.4f}")
                        return float(rd)
        except (KeyError, IndexError, ValueError):
            pass
        
        # Approach 2: Market-based estimate
        if use_market_approach:
            try:
                debt = self.total_debt_latest()
                # Calculate market cap from cached data
                if (self.market_data_cache
                    and self.market_data_cache.get('currentPrice')
                    and self.market_data_cache.get('sharesOutstanding')):
                    equity_val = float(self.market_data_cache['currentPrice'] * self.market_data_cache['sharesOutstanding'])
                else:
                    equity_val = float(self._equity().iloc[-1])

                de_ratio = debt / equity_val if equity_val > 0 else 0
                
                if de_ratio < 0.3:
                    credit_spread = 0.008
                    rating = "AAA/AA"
                elif de_ratio < 0.5:
                    credit_spread = 0.012
                    rating = "A"
                elif de_ratio < 1.0:
                    credit_spread = 0.018
                    rating = "BBB"
                else:
                    credit_spread = 0.025
                    rating = "BB or lower"
                
                rf = self.risk_free_rate()
                rd = rf + credit_spread
                
                if verbose:
                    print(f"Cost of Debt (Approach 2 - Market-based):")
                    print(f"  D/E Ratio: {de_ratio:.2f}, Implied Rating: {rating}")
                    print(f"  Rf: {rf:.4f}, Credit Spread: {credit_spread:.4f}, Rd: {rd:.4f}")
                
                return float(rd)
                
            except (KeyError, ValueError) as e:
                if verbose:
                    print(f"Approach 2 failed: {e}")
        
        # Approach 3: Conservative estimate
        try:
            rf = self.risk_free_rate()
            rd = rf + 0.010
            if verbose:
                print(f"Cost of Debt (Approach 3 - Conservative): Rf {rf:.4f} + 100bps = {rd:.4f}")
            return float(rd)
        except:
            if verbose:
                print("Cost of Debt (Fallback): 5.0%")
            return 0.05

    def wacc(self, default_erp=0.055):
        """Weighted Average Cost of Capital."""
        re = self.cost_of_equity_capm(default_erp=default_erp)
        rd = self.cost_of_debt()
        tax = self.effective_tax_rate_latest()

        debt = float(self.total_debt_latest())

        # Calculate market cap from cached price * shares, or use book equity as fallback
        if (self.market_data_cache
            and self.market_data_cache.get('currentPrice')
            and self.market_data_cache.get('sharesOutstanding')):
            equity = float(self.market_data_cache['currentPrice'] * self.market_data_cache['sharesOutstanding'])
        else:
            equity = float(self._equity().iloc[-1])

        if (debt + equity) == 0:
            raise ValueError("Debt + equity is zero; cannot compute WACC")

        return float((equity / (debt + equity)) * re + (debt / (debt + equity)) * rd * (1 - tax))

    def cash_and_equivalents_latest(self):
        """Latest cash and cash equivalents."""
        cash = self._get_first_available_row(
            self.balance,
            [
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Short Term Investments",
            ],
        ).iloc[-1]
        return float(cash)

    def net_debt_latest(self):
        """Net debt = Total debt - Cash."""
        debt = float(self.total_debt_latest())
        cash = float(self.cash_and_equivalents_latest())
        return float(debt - cash)

    def update_market_data_cache(self):
        """Fetch fresh market data and update cache using Alpha Vantage (primary) or yfinance (fallback)."""
        print("Fetching fresh market data from Alpha Vantage...")

        try:
            av_data = fetch_market_data_alpha_vantage(self.ticker_symbol)
            current_price = av_data['currentPrice']
            beta = av_data['beta']
            shares_outstanding = av_data['sharesOutstanding']

            # Validate we got the critical data
            if current_price is None or shares_outstanding is None:
                raise ValueError("Alpha Vantage returned incomplete data")

        except Exception as e:
            print(f"Alpha Vantage failed ({e}), falling back to yfinance...")
            current_price = safe_yfinance_call(lambda: self.ticker.info.get("currentPrice", None))
            beta = safe_yfinance_call(lambda: self.ticker.info.get("beta", 1.0))
            shares_outstanding = safe_yfinance_call(lambda: self.ticker.info.get("sharesOutstanding", None))

        risk_free_rate = self.risk_free_rate()

        save_market_data_cache(
            self.ticker_symbol,
            current_price,
            beta,
            risk_free_rate,
            shares_outstanding
        )

        self.market_data_cache = {
            'currentPrice': current_price,
            'beta': beta,
            'riskFreeRate': risk_free_rate,
            'sharesOutstanding': shares_outstanding,
        }

    def dcf_valuation(
        self,
        forecast_years=5,
        terminal_growth=0.04,  # ← Changed from 0.025 (2.5%) to 0.04 (4%)
        cap_forecast_growth=0.20,
        default_erp=0.055,
    ):
        """Discounted Cash Flow (DCF) valuation using FCFF."""
        fcff_series = self.free_cash_flow_fcff().dropna()
        if fcff_series.empty:
            raise ValueError("FCFF series is empty")
        
        fcff_0 = float(fcff_series.iloc[-1])
        
        # Use 2-year CAGR for recent trend (more stable, excludes outliers)
        available_years = len(fcff_series) - 1
        
        if available_years >= 2:
            cagr_years = 2  # Use most recent 2 years for growth forecast
        else:
            raise ValueError(f"Need at least 3 years of data, but only have {len(fcff_series)}")
        
        g = float(self.fcf_cagr(years=cagr_years))
        g = max(-0.50, min(g, cap_forecast_growth))
        w = float(self.wacc(default_erp=default_erp))
        
        if terminal_growth >= w:
            terminal_growth = w - 0.005
        
        pv_fcffs = []
        fcff_t = fcff_0
        
        for t in range(1, forecast_years + 1):
            fcff_t = fcff_t * (1 + g)
            pv_fcffs.append(fcff_t / ((1 + w) ** t))
        
        terminal_value = (fcff_t * (1 + terminal_growth)) / (w - terminal_growth)
        pv_terminal = terminal_value / ((1 + w) ** forecast_years)
        
        enterprise_value = sum(pv_fcffs) + pv_terminal
        net_debt = self.net_debt_latest()
        equity_value = enterprise_value - net_debt
        
        if self.market_data_cache and 'sharesOutstanding' in self.market_data_cache:
            shares = self.market_data_cache['sharesOutstanding']
        else:
            shares = safe_yfinance_call(lambda: self.ticker.info.get("sharesOutstanding", None))
        
        intrinsic_price = equity_value / shares if shares else None
        
        return {
            "fcff_base": fcff_0,
            "forecast_growth": g,
            "terminal_growth": terminal_growth,
            "wacc": w,
            "enterprise_value": enterprise_value,
            "net_debt": net_debt,
            "equity_value": equity_value,
            "shares_outstanding": shares,
            "intrinsic_price": intrinsic_price,
            "current_price": (self.market_data_cache.get('currentPrice') if self.market_data_cache else None) or safe_yfinance_call(lambda: self.ticker.info.get("currentPrice", None)),
        }
    
    def dcf_terminal_growth_sensitivity(
        self,
        terminal_growth_rates=(0.03, 0.04, 0.05),  # Changed default to 3%, 4%, 5%
        forecast_years=5,
        cap_forecast_growth=0.20,
        default_erp=0.055,
    ):
        """Run DCF valuation across multiple terminal growth rates."""
        results = []
        
        for tg in terminal_growth_rates:
            dcf = self.dcf_valuation(
                forecast_years=forecast_years,
                terminal_growth=tg,
                cap_forecast_growth=cap_forecast_growth,
                default_erp=default_erp,
            )
            
            results.append({
                "terminal_growth": tg,
                "wacc": dcf["wacc"],
                "intrinsic_price": dcf["intrinsic_price"],
                "current_price": dcf["current_price"],
            })
        
        return pd.DataFrame(results)

    def get_current_multiples(self):
        """Get current valuation multiples for the stock using Alpha Vantage."""
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

        try:
            url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={self.ticker_symbol}&apikey={api_key}"
            response = requests.get(url)
            data = response.json()

            return {
                "P/E": float(data.get("TrailingPE")) if data.get("TrailingPE") and data.get("TrailingPE") != "None" else None,
                "Forward P/E": float(data.get("ForwardPE")) if data.get("ForwardPE") and data.get("ForwardPE") != "None" else None,
                "P/B": float(data.get("PriceToBookRatio")) if data.get("PriceToBookRatio") and data.get("PriceToBookRatio") != "None" else None,
                "P/S": float(data.get("PriceToSalesRatioTTM")) if data.get("PriceToSalesRatioTTM") and data.get("PriceToSalesRatioTTM") != "None" else None,
                "EV/EBITDA": float(data.get("EVToEBITDA")) if data.get("EVToEBITDA") and data.get("EVToEBITDA") != "None" else None,
            }
        except Exception as e:
            print(f"Alpha Vantage multiples failed ({e}), trying yfinance...")
            def fetch_multiples():
                info = self.ticker.info
                return {
                    "P/E": info.get("trailingPE", None),
                    "Forward P/E": info.get("forwardPE", None),
                    "P/B": info.get("priceToBook", None),
                    "P/S": info.get("priceToSalesTrailing12Months", None),
                    "EV/EBITDA": info.get("enterpriseToEbitda", None),
                }
            return safe_yfinance_call(fetch_multiples)
    
    def get_peer_multiples(self, peers=None, use_cache=True):
        """Fetch valuation multiples for peer companies with caching."""
        if peers is None:
            peers = ["MSFT", "GOOGL", "META", "NVDA"]

        # Try to load from cache first
        if use_cache:
            cached = load_peer_multiples_cache()
            if cached is not None:
                return cached

        print("Fetching peer multiples from Alpha Vantage...")
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        peer_data = []

        for peer in peers:
            try:
                # Use Alpha Vantage OVERVIEW for peer data
                url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={peer}&apikey={api_key}"
                response = requests.get(url)
                data = response.json()

                peer_entry = {
                    "Ticker": peer,
                    "P/E": float(data.get("TrailingPE")) if data.get("TrailingPE") and data.get("TrailingPE") != "None" else None,
                    "Forward P/E": float(data.get("ForwardPE")) if data.get("ForwardPE") and data.get("ForwardPE") != "None" else None,
                    "P/B": float(data.get("PriceToBookRatio")) if data.get("PriceToBookRatio") and data.get("PriceToBookRatio") != "None" else None,
                    "P/S": float(data.get("PriceToSalesRatioTTM")) if data.get("PriceToSalesRatioTTM") and data.get("PriceToSalesRatioTTM") != "None" else None,
                    "EV/EBITDA": float(data.get("EVToEBITDA")) if data.get("EVToEBITDA") and data.get("EVToEBITDA") != "None" else None,
                }
                peer_data.append(peer_entry)
                print(f"  ✓ {peer}")
                time.sleep(12)  # Alpha Vantage rate limit
            except Exception as e:
                print(f"  ✗ Error fetching {peer}: {e}")

        df = pd.DataFrame(peer_data)

        # Save to cache
        if not df.empty:
            save_peer_multiples_cache(df)

        return df
    
    def multiples_valuation(self, peers=None):
        """Value the stock using peer multiples approach."""
        peer_df = self.get_peer_multiples(peers)
        current_multiples = self.get_current_multiples()

        # Calculate median peer multiples
        peer_medians = {}
        for col in ["P/E", "Forward P/E", "P/B", "P/S", "EV/EBITDA"]:
            values = peer_df[col].dropna()
            if not values.empty:
                peer_medians[col] = float(values.median())

        # Get fundamentals from cache or fetch
        net_income = self.income.loc["Net Income"].iloc[-1]
        revenue = self.income.loc["Total Revenue"].iloc[-1]
        book_value = self._equity().iloc[-1]

        # Use cached shares outstanding
        if self.market_data_cache and self.market_data_cache.get('sharesOutstanding'):
            shares = self.market_data_cache['sharesOutstanding']
        else:
            shares = safe_yfinance_call(lambda: self.ticker.info.get("sharesOutstanding", 1))
        
        # Calculate EBITDA
        ebit = self._get_first_available_row(
            self.income, ["EBIT", "Operating Income", "Ebit"]
        ).iloc[-1]
        da = self._get_first_available_row(
            self.cashflow,
            ["Depreciation And Amortization", "Depreciation", "Depreciation & Amortization"],
        ).iloc[-1]
        ebitda = ebit + da
        
        # Implied prices
        implied_prices = {}
        
        if "P/E" in peer_medians:
            eps = net_income / shares
            implied_prices["P/E"] = peer_medians["P/E"] * eps
        
        if "P/B" in peer_medians:
            book_per_share = book_value / shares
            implied_prices["P/B"] = peer_medians["P/B"] * book_per_share
        
        if "P/S" in peer_medians:
            sales_per_share = revenue / shares
            implied_prices["P/S"] = peer_medians["P/S"] * sales_per_share
        
        if "EV/EBITDA" in peer_medians:
            implied_ev = peer_medians["EV/EBITDA"] * ebitda
            implied_equity = implied_ev - self.net_debt_latest()
            implied_prices["EV/EBITDA"] = implied_equity / shares

        # Use cached current price
        if self.market_data_cache and self.market_data_cache.get('currentPrice'):
            current_price = self.market_data_cache['currentPrice']
        else:
            current_price = safe_yfinance_call(lambda: self.ticker.info.get("currentPrice", None))
        
        return {
            "current_price": current_price,
            "current_multiples": current_multiples,
            "peer_median_multiples": peer_medians,
            "implied_prices": implied_prices,
            "average_implied_price": np.mean(list(implied_prices.values())) if implied_prices else None,
        }


if __name__ == "__main__":
    analyzer = FundamentalAnalyzer("AAPL")

    # Check if cache is valid and has required data
    cache_valid = (
        analyzer.market_data_cache is not None
        and analyzer.market_data_cache.get('currentPrice') is not None
        and analyzer.market_data_cache.get('sharesOutstanding') is not None
    )

    if not cache_valid:
        print("Cache is missing or incomplete. Refreshing market data...")
        analyzer.update_market_data_cache()

    print("="*70)
    print("DCF VALUATION")
    print("="*70)

    try:
        dcf_result = analyzer.dcf_valuation()
        
        if dcf_result['intrinsic_price'] is not None:
            print(f"\nForecast Growth (CAGR):        {dcf_result['forecast_growth']:.2%}")
            print(f"WACC:                          {dcf_result['wacc']:.2%}")
            print(f"DCF Intrinsic Price:           ${dcf_result['intrinsic_price']:.2f}")
            print(f"Current Price:                 ${dcf_result['current_price']:.2f}")
            dcf_upside = ((dcf_result['intrinsic_price'] - dcf_result['current_price']) / dcf_result['current_price']) * 100
            print(f"DCF Upside:                    {dcf_upside:+.2f}%")
        else:
            print("\nError: Could not calculate DCF intrinsic price (missing shares outstanding)")
            print(f"Forecast Growth (CAGR):        {dcf_result['forecast_growth']:.2%}")
            print(f"WACC:                          {dcf_result['wacc']:.2%}")
    except Exception as e:
        print(f"\nDCF Valuation failed: {e}")
        print("\nThis is likely due to Yahoo Finance rate limiting.")
        print("Please wait 10-15 minutes and try again.")
        exit(1)
    
    print("\n" + "="*70)
    print("MULTIPLES VALUATION")
    print("="*70)
    
    try:
        multiples = analyzer.multiples_valuation()
        
        print(f"\nCurrent Multiples (AAPL):")
        for k, v in multiples['current_multiples'].items():
            if v:
                print(f"  {k}: {v:.2f}")
        
        print(f"\nPeer Median Multiples:")
        for k, v in multiples['peer_median_multiples'].items():
            print(f"  {k}: {v:.2f}")
        
        print(f"\nImplied Prices by Multiple:")
        for k, v in multiples['implied_prices'].items():
            print(f"  {k}: ${v:.2f}")
        
        print(f"\nMultiples Average Price:       ${multiples['average_implied_price']:.2f}")
        multiples_upside = ((multiples['average_implied_price'] - multiples['current_price']) / multiples['current_price']) * 100
        print(f"Multiples Upside:              {multiples_upside:+.2f}%")
        
        print("\n" + "="*70)
        print("BLENDED RECOMMENDATION (60% DCF / 40% Multiples)")
        print("="*70)
        
        blended = dcf_result['intrinsic_price'] * 0.6 + multiples['average_implied_price'] * 0.4
        blended_upside = ((blended - dcf_result['current_price']) / dcf_result['current_price']) * 100
        
        print(f"\nDCF Target (60%):              ${dcf_result['intrinsic_price']:.2f}")
        print(f"Multiples Target (40%):        ${multiples['average_implied_price']:.2f}")
        print(f"Blended Target Price:          ${blended:.2f}")
        print(f"Current Price:                 ${dcf_result['current_price']:.2f}")
        print(f"Blended Upside:                {blended_upside:+.2f}%")
        
        if blended_upside > 20:
            rec = "BUY"
            confidence = "High" if blended_upside > 30 else "Medium"
        elif blended_upside < -15:
            rec = "SELL"
            confidence = "High" if blended_upside < -25 else "Medium"
        else:
            rec = "HOLD"
            confidence = "Medium"
        
        print(f"\n{'='*70}")
        print(f"FINAL RECOMMENDATION: {rec} ({confidence} Confidence)")
        print(f"{'='*70}")

        print("\n" + "="*70)
        
        print("TERMINAL GROWTH SENSITIVITY (3%, 4%, 5%)")
        print("="*70)

        sensitivity = analyzer.dcf_terminal_growth_sensitivity()

        print(f"\n{'Terminal Growth':<20} {'Target Price':<15} {'Upside':<15}")
        print("-" * 50)
        for _, row in sensitivity.iterrows():
            tg = row['terminal_growth']
            price = row['intrinsic_price']
            current = row['current_price']
            upside = ((price - current) / current) * 100 if current else 0
            print(f"{tg:>18.1%}  ${price:>12.2f}  {upside:>12.2f}%")
        
    except Exception as e:
        print(f"\nMultiples Valuation failed: {e}")
        print("This is likely due to Yahoo Finance rate limiting.")
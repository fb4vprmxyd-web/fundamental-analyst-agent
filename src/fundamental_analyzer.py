import pandas as pd
import numpy as np
from pathlib import Path


class FundamentalAnalyzer:

    def __init__(self, ticker_symbol="AAPL", data_path="data/raw"):
        import yfinance as yf

        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol)

        self.data_path = Path(data_path)
        self.income = pd.read_csv(self.data_path / "income_statement.csv", index_col=0)
        self.balance = pd.read_csv(self.data_path / "balance_sheet.csv", index_col=0)
        self.cashflow = pd.read_csv(self.data_path / "cash_flow.csv", index_col=0)

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
    
    def risk_free_rate(self):
        """Fetch 10-year US Treasury yield from Yahoo Finance (^TNX)."""
        import yfinance as yf

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

    def equity_risk_premium(self, default_erp=0.055):
        """Equity Risk Premium (market assumption, e.g., 5.5%)."""
        return float(default_erp)

    def cost_of_equity_capm(self, default_erp=0.055):
        """CAPM: Re = Rf + beta * ERP"""
        beta = self.ticker.info.get("beta", 1.0)
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
                market_cap = self.ticker.info.get("marketCap", None)
                equity_val = float(market_cap) if market_cap else self._equity().iloc[-1]

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

        market_cap = self.ticker.info.get("marketCap", None)
        if market_cap is None:
            equity = float(self._equity().iloc[-1])
        else:
            equity = float(market_cap)

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

    def dcf_valuation(
        self,
        forecast_years=5,
        terminal_growth=0.025,
        cap_forecast_growth=0.20,
        default_erp=0.055,
    ):
        """Discounted Cash Flow (DCF) valuation using FCFF."""
        fcff_series = self.free_cash_flow_fcff().dropna()
        if fcff_series.empty:
            raise ValueError("FCFF series is empty")

        fcff_0 = float(fcff_series.iloc[-1])
        
        # Auto-adjust CAGR years based on available data
        available_years = len(fcff_series) - 1
        cagr_years = min(available_years, 5)  # Use up to 5 years if available
        
        if cagr_years < 2:
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

        shares = self.ticker.info.get("sharesOutstanding", None)
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
            "current_price": self.ticker.info.get("currentPrice", None),
        }

    def dcf_terminal_growth_sensitivity(
        self,
        terminal_growth_rates=(0.025, 0.03, 0.035),
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
        """Get current valuation multiples for the stock."""
        info = self.ticker.info
        
        return {
            "P/E": info.get("trailingPE", None),
            "Forward P/E": info.get("forwardPE", None),
            "P/B": info.get("priceToBook", None),
            "P/S": info.get("priceToSalesTrailing12Months", None),
            "EV/EBITDA": info.get("enterpriseToEbitda", None),
        }
    
    def get_peer_multiples(self, peers=None):
        """Fetch valuation multiples for peer companies."""
        import yfinance as yf
        
        if peers is None:
            peers = ["MSFT", "GOOGL", "META", "NVDA"]
        
        peer_data = []
        
        for peer in peers:
            try:
                ticker = yf.Ticker(peer)
                info = ticker.info
                
                peer_data.append({
                    "Ticker": peer,
                    "P/E": info.get("trailingPE", None),
                    "Forward P/E": info.get("forwardPE", None),
                    "P/B": info.get("priceToBook", None),
                    "P/S": info.get("priceToSalesTrailing12Months", None),
                    "EV/EBITDA": info.get("enterpriseToEbitda", None),
                })
            except Exception as e:
                print(f"Error fetching {peer}: {e}")
        
        return pd.DataFrame(peer_data)
    
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
        
        # Get fundamentals
        info = self.ticker.info
        net_income = self.income.loc["Net Income"].iloc[-1]
        revenue = self.income.loc["Total Revenue"].iloc[-1]
        book_value = self._equity().iloc[-1]
        shares = info.get("sharesOutstanding", 1)
        
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
        
        current_price = info.get("currentPrice", None)
        
        return {
            "current_price": current_price,
            "current_multiples": current_multiples,
            "peer_median_multiples": peer_medians,
            "implied_prices": implied_prices,
            "average_implied_price": np.mean(list(implied_prices.values())) if implied_prices else None,
        }


if __name__ == "__main__":
    analyzer = FundamentalAnalyzer("AAPL")
    
    print("="*70)
    print("DCF VALUATION")
    print("="*70)
    
    dcf_result = analyzer.dcf_valuation()
    print(f"\nForecast Growth (2-yr CAGR):   {dcf_result['forecast_growth']:.2%}")
    print(f"WACC:                          {dcf_result['wacc']:.2%}")
    print(f"DCF Intrinsic Price:           ${dcf_result['intrinsic_price']:.2f}")
    print(f"Current Price:                 ${dcf_result['current_price']:.2f}")
    dcf_upside = ((dcf_result['intrinsic_price'] - dcf_result['current_price']) / dcf_result['current_price']) * 100
    print(f"DCF Upside:                    {dcf_upside:+.2f}%")
    
    print("\n" + "="*70)
    print("MULTIPLES VALUATION")
    print("="*70)
    
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
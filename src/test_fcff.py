from fundamental_analyzer import FundamentalAnalyzer
import pandas as pd

analyzer = FundamentalAnalyzer("AAPL")

print("="*70)
print("DATA AVAILABILITY CHECK")
print("="*70)

print("\n1. RAW DATA YEARS:")
print(f"   Income: {list(analyzer.income.columns)}")
print(f"   Balance: {list(analyzer.balance.columns)}")
print(f"   Cashflow: {list(analyzer.cashflow.columns)}")

print("\n2. FCFF COMPONENTS:")

# EBIT
ebit = analyzer._get_first_available_row(analyzer.income, ["EBIT", "Operating Income", "Ebit"])
print(f"\n   EBIT by year:\n{ebit}")

# D&A
da = analyzer._get_first_available_row(analyzer.cashflow, ["Depreciation And Amortization", "Depreciation"])
print(f"\n   D&A by year:\n{da}")

# Capex
capex = analyzer._get_first_available_row(analyzer.cashflow, ["Capital Expenditures", "Capital Expenditure"])
print(f"\n   CapEx by year:\n{capex}")

# NWC calculation
current_assets = analyzer._get_first_available_row(analyzer.balance, ["Total Current Assets", "Current Assets"])
cash = analyzer._get_first_available_row(analyzer.balance, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])
current_liabilities = analyzer._get_first_available_row(analyzer.balance, ["Total Current Liabilities", "Current Liabilities"])
short_term_debt = analyzer._get_first_available_row(analyzer.balance, ["Short Long Term Debt", "Short Term Debt", "Current Debt"])

nwc = (current_assets - cash) - (current_liabilities - short_term_debt)
print(f"\n   NWC by year:\n{nwc}")

delta_nwc = nwc.diff()
print(f"\n   ΔNWC by year (THIS CREATES NaN!):\n{delta_nwc}")

print("\n3. FINAL FCFF:")
fcff = analyzer.free_cash_flow_fcff()
print(f"\n   FCFF (before dropna):\n{fcff}")

fcff_clean = fcff.dropna()
print(f"\n   FCFF (after dropna):\n{fcff_clean}")
print(f"\n   >>> USABLE YEARS: {len(fcff_clean)}")
print("\n4. CAGR CALCULATION:")

fcff_clean = analyzer.free_cash_flow_fcff().dropna()
print(f"   Available FCFF data points: {len(fcff_clean)}")
print(f"   Years: {list(fcff_clean.index)}")

# Try different CAGR periods
for years in [2, 3, 4]:
    try:
        cagr = analyzer.fcf_cagr(years=years)
        fcf_start = fcff_clean.iloc[-(years + 1)]
        fcf_end = fcff_clean.iloc[-1]
        print(f"\n   {years}-year CAGR: {cagr:.2%}")
        print(f"     Start ({fcff_clean.index[-(years + 1)]}): ${fcf_start:,.0f}")
        print(f"     End ({fcff_clean.index[-1]}): ${fcf_end:,.0f}")
    except Exception as e:
        print(f"\n   {years}-year CAGR: FAILED - {e}")
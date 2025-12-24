import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw")

def load_statements():
    income = pd.read_csv(RAW_DATA_PATH / "income_statement.csv", index_col=0)
    balance = pd.read_csv(RAW_DATA_PATH / "balance_sheet.csv", index_col=0)
    cashflow = pd.read_csv(RAW_DATA_PATH / "cash_flow.csv", index_col=0)

    return income, balance, cashflow

def profitability_ratios(income, balance):
    ratios = {}

    ratios["Net Margin"] = (income.loc["Net Income"] / income.loc["Total Revenue"])

    ratios["ROE"] = (income.loc["Net Income"] / balance.loc["Total Stockholder Equity"])

    return pd.DataFrame(ratios)

def leverage_ratios(balance):
    ratios = {}

    ratios["Debt to Equity"] = (balance.loc["Total Liab"] / balance.loc["Total Stockholder Equity"])

    return pd.DataFrame(ratios)
def leverage_ratios(balance):
    ratios = {}

    ratios["Debt to Equity"] = (
        balance.loc["Total Liab"] / balance.loc["Total Stockholder Equity"]
    )

    return pd.DataFrame(ratios)
def growth_rates(income):
    growth = {}

    revenue = income.loc["Total Revenue"]
    growth["Revenue Growth"] = revenue.pct_change()

    net_income = income.loc["Net Income"]
    growth["Net Income Growth"] = net_income.pct_change()

    return pd.DataFrame(growth)
if __name__ == "__main__":
    income, balance, cashflow = load_statements()

    prof = profitability_ratios(income, balance)
    lev = leverage_ratios(balance)
    growth = growth_rates(income)

    print("Profitability:")
    print(prof.tail())

    print("\nLeverage:")
    print(lev.tail())

    print("\nGrowth:")
    print(growth.tail())

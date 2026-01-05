import pandas as pd
from pathlib import Path


class FundamentalAnalyzer:
    def __init__(self, data_path="data/raw"):
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


if __name__ == "__main__":
    analyzer = FundamentalAnalyzer()

    print(analyzer.profitability_ratios().tail())
    print(analyzer.leverage_ratios().tail())
    print(analyzer.growth_rates().tail())

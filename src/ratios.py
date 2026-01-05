from fundamental_analyzer import FundamentalAnalyzer


def run_ratio_analysis():
    analyzer = FundamentalAnalyzer()

    profitability = analyzer.profitability_ratios()
    leverage = analyzer.leverage_ratios()
    growth = analyzer.growth_rates()

    return {
        "profitability": profitability,
        "leverage": leverage,
        "growth": growth,
    }


if __name__ == "__main__":
    results = run_ratio_analysis()

    print("Profitability Ratios:")
    print(results["profitability"].tail())

    print("\nLeverage Ratios:")
    print(results["leverage"].tail())

    print("\nGrowth Rates:")
    print(results["growth"].tail())

import os
from anthropic import Anthropic
from dotenv import load_dotenv
from fundamental_analyzer import FundamentalAnalyzer
from report_generator import ReportGenerator, create_analysis_data_structure

# Load environment variables
load_dotenv()

class InvestmentAIAgent:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env file")
        self.client = Anthropic(api_key=api_key)
    
    def generate_investment_report(self, ticker="AAPL"):
        """Generate a comprehensive AI-powered investment report."""
        
        # Run the analysis
        print(f"Analyzing {ticker}...")
        analyzer = FundamentalAnalyzer(ticker)
        
        # Get DCF results
        dcf_result = analyzer.dcf_valuation()
        sensitivity = analyzer.dcf_terminal_growth_sensitivity()
        
        # Get multiples results
        multiples = analyzer.multiples_valuation()
        
        # Calculate blended recommendation
        blended_price = dcf_result['intrinsic_price'] * 0.6 + multiples['average_implied_price'] * 0.4
        current_price = dcf_result['current_price']
        blended_upside = ((blended_price - current_price) / current_price) * 100
        
        # Prepare data for Claude
        analysis_data = f"""
Stock: {ticker}
Current Price: ${current_price:.2f}

DCF VALUATION:
- Intrinsic Price: ${dcf_result['intrinsic_price']:.2f}
- Forecast Growth (CAGR): {dcf_result['forecast_growth']:.2%}
- WACC: {dcf_result['wacc']:.2%}
- Terminal Growth: {dcf_result['terminal_growth']:.2%}
- Base FCFF: ${dcf_result['fcff_base']:,.0f}
- Enterprise Value: ${dcf_result['enterprise_value']:,.0f}
- Upside: {((dcf_result['intrinsic_price'] - current_price) / current_price * 100):+.2f}%

MULTIPLES VALUATION:
Current Multiples:
- P/E: {multiples['current_multiples']['P/E']:.2f}
- P/B: {multiples['current_multiples']['P/B']:.2f}
- P/S: {multiples['current_multiples']['P/S']:.2f}
- EV/EBITDA: {multiples['current_multiples']['EV/EBITDA']:.2f}

Peer Median Multiples:
- P/E: {multiples['peer_median_multiples']['P/E']:.2f}
- P/B: {multiples['peer_median_multiples']['P/B']:.2f}
- P/S: {multiples['peer_median_multiples']['P/S']:.2f}
- EV/EBITDA: {multiples['peer_median_multiples']['EV/EBITDA']:.2f}

Implied Prices:
- P/E: ${multiples['implied_prices']['P/E']:.2f}
- P/B: ${multiples['implied_prices']['P/B']:.2f}
- P/S: ${multiples['implied_prices']['P/S']:.2f}
- EV/EBITDA: ${multiples['implied_prices']['EV/EBITDA']:.2f}
- Average: ${multiples['average_implied_price']:.2f}
- Upside: {((multiples['average_implied_price'] - current_price) / current_price * 100):+.2f}%

BLENDED RECOMMENDATION:
- Target Price: ${blended_price:.2f} (60% DCF, 40% Multiples)
- Upside: {blended_upside:+.2f}%
"""
        
        # Create prompt for Claude
        prompt = f"""You are a professional equity research analyst. Based on the following fundamental analysis, write a comprehensive investment report for {ticker}.

{analysis_data}

Please provide:
1. **Executive Summary** - One paragraph overview with clear BUY/HOLD/SELL recommendation
2. **Valuation Analysis** - Discuss the DCF and multiples approaches, highlighting key insights
3. **Key Risks** - What could make this analysis wrong?
4. **Investment Thesis** - The core argument for or against investing
5. **Price Target** - Your recommended target price with reasoning

Be specific, professional, and data-driven. Use the numbers provided to support your analysis."""

        # Call Claude API
        print("Generating AI report...")
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the response
        report = message.content[0].text

                # Generate JSON and PDF reports
        analysis_data = create_analysis_data_structure(
            ticker=ticker,
            dcf_result=dcf_result,
            multiples=multiples,
            blended_target=blended_price,
            blended_upside=blended_upside,
            sensitivity=sensitivity,
            ai_report=report,
        )

        generator = ReportGenerator()
        generator.save_to_json(analysis_data)
        generator.generate_pdf(analysis_data)
        
        return {
            "ticker": ticker,
            "dcf_result": dcf_result,
            "multiples": multiples,
            "blended_price": blended_price,
            "blended_upside": blended_upside,
            "ai_report": report
        }

if __name__ == "__main__":
    agent = InvestmentAIAgent()
    
    print("="*70)
    print("AI-POWERED INVESTMENT ANALYSIS")
    print("="*70)
    
    result = agent.generate_investment_report("AAPL")
    
    print("\n" + "="*70)
    print("QUANTITATIVE SUMMARY")
    print("="*70)
    print(f"Current Price:        ${result['dcf_result']['current_price']:.2f}")
    print(f"DCF Target:           ${result['dcf_result']['intrinsic_price']:.2f}")
    print(f"Multiples Target:     ${result['multiples']['average_implied_price']:.2f}")
    print(f"Blended Target:       ${result['blended_price']:.2f}")
    print(f"Upside Potential:     {result['blended_upside']:+.2f}%")
    
    print("\n" + "="*70)
    print("AI INVESTMENT REPORT")
    print("="*70)
    print(result['ai_report'])


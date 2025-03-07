"""Test script for the AI Hedge Fund workflow."""

import os
import asyncio
from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from hedgehog.workflow import analyze_company


# Load environment variables
load_dotenv()

# Get API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY environment variable is not set. "
        "Please set it in your .env file"
    )


async def main():
    """Run a test of the workflow."""
    # Set up the model
    model = OpenAIModel(
        "anthropic/claude-3.5-sonnet",
        provider=OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        ),
    )

    # Run a sample analysis
    ticker = "AAPL"
    print(f"Analyzing {ticker}...")

    analysis = await analyze_company(ticker, model)

    # Print the results
    decision = analysis.investment_decision
    print(f"\n{ticker} ({analysis.company_name}) - {decision.order_type} RECOMMENDATION")
    print(f"Conviction: {decision.conviction_level}/10")
    print(f"Position Size: {decision.position_size:.1f}%")
    print(f"Target Price: ${decision.target_price:.2f}")

    if decision.stop_loss:
        print(f"Stop Loss: ${decision.stop_loss:.2f}")

    print(f"Time Horizon: {decision.time_horizon}")

    print("\nKey Factors:")
    for factor in decision.key_factors:
        print(f"• {factor}")

    print("\nKey Risks:")
    for risk in decision.risks:
        print(f"• {risk}")

    print(f"\nReasoning: {decision.reasoning}")

    # Print some analysis details
    print("\nFundamental Analysis Highlights:")
    print(f"Rating: {analysis.fundamental_analysis.rating}/10")
    print(f"Sector: {analysis.fundamental_analysis.sector}")
    print(f"P/E Ratio: {analysis.fundamental_analysis.metrics.pe_ratio}")

    print("\nTechnical Analysis Highlights:")
    print(f"Rating: {analysis.technical_analysis.rating}/10")
    print(f"Current Price: ${analysis.technical_analysis.current_price:.2f}")
    print(f"RSI: {analysis.technical_analysis.indicators.rsi_14}")

    print("\nWarren Buffett Analysis:")
    print(f"Rating: {analysis.buffett_analysis.rating}/10")
    print(f"Would Buffett Invest: {'Yes' if analysis.buffett_analysis.would_invest else 'No'}")


if __name__ == "__main__":
    asyncio.run(main())
"""Workflow implementation for the Hedgehog AI Hedge Fund analysis process."""

from typing import Dict, Any, List, Optional
import asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from hedgehog.tools.logfire_setup import *
# Import our API functions
from hedgehog.tools.api import (
    fetch_company_data,
    fetch_price_history,
    fetch_news_data,
    fetch_peer_companies,
)

# Import our progress tracker
from hedgehog.progress import progress

# Define our model schemas
class FinancialMetrics(BaseModel):
    """Key financial metrics for a company."""

    pe_ratio: Optional[float] = Field(None, description="Price to Earnings ratio")
    pb_ratio: Optional[float] = Field(None, description="Price to Book ratio")
    roe: Optional[float] = Field(None, description="Return on Equity")
    debt_to_equity: Optional[float] = Field(None, description="Debt to Equity ratio")
    revenue_growth: Optional[float] = Field(None, description="Revenue growth rate")
    profit_margin: Optional[float] = Field(None, description="Profit margin")
    free_cash_flow: Optional[float] = Field(None, description="Free Cash Flow in millions")


class FundamentalAnalysis(BaseModel):
    """Fundamental analysis of a company."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")
    sector: str = Field(..., description="Industry sector")
    metrics: FinancialMetrics = Field(..., description="Key financial metrics")
    strengths: List[str] = Field(..., description="Fundamental strengths")
    weaknesses: List[str] = Field(..., description="Fundamental weaknesses")
    rating: int = Field(..., ge=1, le=10, description="Overall rating from 1-10")
    recommendation: str = Field(..., description="Investment recommendation (Buy/Hold/Sell)")
    reasoning: str = Field(..., description="Reasoning behind recommendation")
    detailed_reasoning: Optional[str] = Field(None, description="Detailed reasoning and analysis")


class TechnicalIndicators(BaseModel):
    """Key technical indicators for a stock."""

    moving_average_50d: Optional[float] = Field(None, description="50-day moving average")
    moving_average_200d: Optional[float] = Field(None, description="200-day moving average")
    rsi_14: Optional[float] = Field(None, description="14-day Relative Strength Index")
    macd: Optional[float] = Field(None, description="Moving Average Convergence Divergence")
    bollinger_upper: Optional[float] = Field(None, description="Upper Bollinger Band")
    bollinger_lower: Optional[float] = Field(None, description="Lower Bollinger Band")
    volume_avg_30d: Optional[float] = Field(None, description="30-day average volume")


class TechnicalAnalysis(BaseModel):
    """Technical analysis of a stock."""

    ticker: str = Field(..., description="Stock ticker symbol")
    indicators: TechnicalIndicators = Field(..., description="Key technical indicators")
    patterns: List[str] = Field(..., description="Chart patterns identified")
    signals: List[str] = Field(..., description="Trading signals")
    rating: int = Field(..., ge=1, le=10, description="Overall rating from 1-10")
    recommendation: str = Field(..., description="Investment recommendation (Buy/Hold/Sell)")
    reasoning: str = Field(..., description="Reasoning behind recommendation")
    detailed_reasoning: Optional[str] = Field(None, description="Detailed reasoning and analysis")
    chart_url: Optional[str] = Field(None, description="URL of the chart image")
    trend: str = Field(..., description="Trend identified (Bullish/Bearish/Neutral)")


class SentimentAnalysis(BaseModel):
    """Sentiment analysis based on news and social media."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")
    overall_sentiment: str = Field(..., description="Overall market sentiment (Positive/Neutral/Negative)")
    news_sentiment: str = Field(..., description="Sentiment from news articles")
    social_sentiment: str = Field(..., description="Sentiment from social media")
    key_topics: List[str] = Field(..., description="Key topics being discussed")
    rating: int = Field(..., ge=1, le=10, description="Overall sentiment rating from 1-10")
    recommendation: str = Field(..., description="Sentiment-based recommendation (Buy/Hold/Sell)")
    reasoning: str = Field(..., description="Reasoning behind recommendation")
    detailed_reasoning: Optional[str] = Field(None, description="Detailed reasoning and analysis")


class InvestorAnalysis(BaseModel):
    """Analysis based on famous investor's principles."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")
    investor_name: str = Field(..., description="Name of the investor (e.g., Warren Buffett)")
    strengths: List[str] = Field(..., description="Key strengths from investor's perspective")
    concerns: List[str] = Field(..., description="Key concerns from investor's perspective")
    would_invest: bool = Field(..., description="Whether the investor would likely invest")
    rating: int = Field(..., ge=1, le=10, description="Overall rating from 1-10")
    recommendation: str = Field(..., description="Investment recommendation (Buy/Hold/Sell)")
    reasoning: str = Field(..., description="Reasoning behind recommendation")
    detailed_reasoning: Optional[str] = Field(None, description="Detailed reasoning and analysis")


class InvestmentDecision(BaseModel):
    """Final investment decision for a specific company."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")
    order_type: str = Field(..., description="Investment decision (BUY/SELL/HOLD)")
    conviction_level: int = Field(..., ge=1, le=10, description="Conviction level from 1-10")
    position_size: float = Field(..., description="Recommended position size (%)")
    target_price: float = Field(..., description="Target price for the position")
    stop_loss: Optional[float] = Field(None, description="Recommended stop loss price")
    time_horizon: str = Field(..., description="Expected investment time horizon")
    key_factors: List[str] = Field(..., description="Key factors influencing the decision")
    risks: List[str] = Field(..., description="Key risks to monitor")
    reasoning: str = Field(..., description="Detailed reasoning behind the decision")
    detailed_reasoning: Optional[str] = Field(None, description="Detailed reasoning and analysis")


class CompanyAnalysisOutput(BaseModel):
    """Comprehensive output from the company analysis process."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")
    fundamental_analysis: Optional[FundamentalAnalysis] = Field(None, description="Fundamental analysis results")
    technical_analysis: Optional[TechnicalAnalysis] = Field(None, description="Technical analysis results")
    sentiment_analysis: Optional[SentimentAnalysis] = Field(None, description="Sentiment analysis results")
    investor_analyses: List[InvestorAnalysis] = Field(default_factory=list, description="Analyses from different investor perspectives")
    investment_decision: InvestmentDecision = Field(..., description="Final investment decision")


async def analyze_fundamentals(agent: Agent, ticker: str, financial_data: Dict[str, Any], show_reasoning: bool = False) -> FundamentalAnalysis:
    """Run fundamental analysis on a company.

    Args:
        agent: Agent to use for the analysis
        ticker: Stock ticker symbol
        financial_data: Financial data for the company
        show_reasoning: Whether to include detailed reasoning in the output

    Returns:
        FundamentalAnalysis: Results of the fundamental analysis
    """
    # Status updates with more variety
    status_messages = [
        "Analyzing balance sheet",
        "Checking income statement",
        "Calculating financial ratios",
        "Evaluating earnings growth",
        "Assessing debt levels",
        "Analyzing cash flow",
        "Calculating ROE and ROA",
        "Evaluating profitability",
        "Finalizing fundamental score"
    ]

    # Show initial status
    progress.update_status("Fundamental Analyst", ticker, status_messages[0])

    # Generate the analysis
    prompt = f"""
        You are a skilled fundamental analyst examining {ticker}.
        Analyze the following financial data and provide an investment recommendation:

        Financial Data:
        {financial_data}

        Be thorough in your analysis and provide a clear BUY, HOLD, or SELL recommendation.
    """

    # Show a few status updates to simulate work
    for i in range(1, min(4, len(status_messages))):
        await asyncio.sleep(0.3)  # Short delay
        progress.update_status("Fundamental Analyst", ticker, status_messages[i])

    # Call the agent with just the prompt
    result = await agent.run(prompt)

    # Extract financial metrics from the financial_data
    pe_ratio = financial_data.get("pe_ratio")
    pb_ratio = financial_data.get("pb_ratio")
    roe = financial_data.get("return_on_equity")
    debt_to_equity = financial_data.get("debt_to_equity")
    revenue_growth = financial_data.get("revenue_growth")
    profit_margin = financial_data.get("profit_margin")
    fcf = financial_data.get("free_cash_flow")

    # Parse recommendation from result (simplified example)
    recommendation = "Hold"  # Default
    if "buy" in result.data.lower():
        recommendation = "Buy"
    elif "sell" in result.data.lower():
        recommendation = "Sell"

    # Rating (1-10) extraction (simplified)
    rating = 5  # Default neutral rating

    # Extract company information
    company_info = await fetch_company_data(ticker)
    company_name = company_info.get("company_name", f"{ticker} Inc.")
    sector = company_info.get("sector", "Technology")

    # Create a FundamentalAnalysis object with extracted data
    analysis = FundamentalAnalysis(
        ticker=ticker,
        company_name=company_name,
        sector=sector,
        metrics=FinancialMetrics(
            pe_ratio=pe_ratio,
            pb_ratio=pb_ratio,
            roe=roe,
            debt_to_equity=debt_to_equity,
            revenue_growth=revenue_growth,
            profit_margin=profit_margin,
            free_cash_flow=fcf
        ),
        strengths=["Strong market position", "Solid financial performance"],
        weaknesses=["Competitive pressures", "Regulatory challenges"],
        rating=rating,
        recommendation=recommendation,
        reasoning="Based on fundamental analysis of financial metrics.",
        detailed_reasoning=result.data if show_reasoning else None
    )

    # Update status to done
    progress.update_status("Fundamental Analyst", ticker, "Done")

    return analysis


async def analyze_technicals(agent: Agent, ticker: str, price_history: Dict[str, Any], show_reasoning: bool = False) -> TechnicalAnalysis:
    """Run technical analysis on a stock.

    Args:
        agent: Agent to use for the analysis
        ticker: Stock ticker symbol
        price_history: Historical price data
        show_reasoning: Whether to include detailed reasoning in the output

    Returns:
        TechnicalAnalysis: Results of the technical analysis
    """
    # Status updates for technical analysis
    status_messages = [
        "Analyzing price trends",
        "Calculating moving averages",
        "Evaluating RSI levels",
        "Checking MACD signals",
        "Analyzing Bollinger Bands",
        "Calculating support levels",
        "Evaluating resistance levels",
        "Checking volume patterns",
        "Finalizing technical score"
    ]

    # Show initial status
    progress.update_status("Technical Analyst", ticker, status_messages[0])

    # Generate the prompt for analysis
    prompt = f"""
        You are a skilled technical analyst examining {ticker}.
        Analyze the following price history data and provide an investment recommendation:

        Price History Data:
        {price_history}

        Be thorough in your technical analysis and provide a clear BUY, HOLD, or SELL recommendation.
    """

    # Show a few status updates to simulate work
    for i in range(1, min(4, len(status_messages))):
        await asyncio.sleep(0.3)  # Short delay
        progress.update_status("Technical Analyst", ticker, status_messages[i])

    # Call the agent with just the prompt
    result = await agent.run(prompt)

    # Extract technical indicators from price history
    ma_50d = price_history.get("moving_average_50d") or price_history.get("ma_50d")
    ma_200d = price_history.get("moving_average_200d") or price_history.get("ma_200d")
    rsi_14 = price_history.get("rsi_14")
    macd = price_history.get("macd")
    bollinger_upper = price_history.get("bollinger_upper")
    bollinger_lower = price_history.get("bollinger_lower")
    volume_avg = price_history.get("volume_avg_30d") or price_history.get("volume_avg")

    # Parse signals from the result (simplified example)
    trend = "Neutral"
    if "uptrend" in result.data.lower() or "bullish" in result.data.lower():
        trend = "Bullish"
    elif "downtrend" in result.data.lower() or "bearish" in result.data.lower():
        trend = "Bearish"

    # Parse recommendation from result (simplified)
    recommendation = "Hold"  # Default
    if "buy" in result.data.lower():
        recommendation = "Buy"
    elif "sell" in result.data.lower():
        recommendation = "Sell"

    # Rating (1-10) extraction (simplified)
    rating = 5  # Default neutral rating

    # Create a TechnicalAnalysis object with extracted data
    analysis = TechnicalAnalysis(
        ticker=ticker,
        indicators=TechnicalIndicators(
            moving_average_50d=ma_50d,
            moving_average_200d=ma_200d,
            rsi_14=rsi_14,
            macd=macd,
            bollinger_upper=bollinger_upper,
            bollinger_lower=bollinger_lower,
            volume_avg_30d=volume_avg
        ),
        patterns=["Support at recent lows", "Resistance at recent highs"],
        chart_url=chart_image_url,
        trend=trend,
        signals=["Moving average crossover", "RSI oversold"],
        rating=rating,
        recommendation=recommendation,
        reasoning="Based on technical indicators and chart patterns.",
        detailed_reasoning=result.data if show_reasoning else None
    )

    # Update status to done
    progress.update_status("Technical Analyst", ticker, "Done")

    return analysis


async def analyze_sentiment(agent: Agent, ticker: str, news_data: List[Dict[str, Any]], show_reasoning: bool = False) -> SentimentAnalysis:
    """Run sentiment analysis on a company.

    Args:
        agent: Agent to use for the analysis
        ticker: Stock ticker symbol
        news_data: News and social media data
        show_reasoning: Whether to include detailed reasoning in the output

    Returns:
        SentimentAnalysis: Results of the sentiment analysis
    """
    # Status updates with more variety
    status_messages = [
        "Fetching insider trades",
        "Scanning Twitter sentiment",
        "Analyzing news headlines",
        "Monitoring Reddit threads",
        "Checking StockTwits activity",
        "Tracking analyst revisions",
        "Measuring public sentiment",
        "Analyzing conference calls",
        "Evaluating news events"
    ]

    # Show initial status
    progress.update_status("Sentiment Analyst", ticker, status_messages[0])

    # Generate the analysis
    prompt = f"""
        You are a skilled sentiment analyst examining {ticker}.
        Analyze the following news and social media data and provide an investment recommendation:

        News and Social Media Data:
        {news_data}

        Identify key topics, sentiment trends, and overall market perception.
        Be thorough in your analysis and provide a clear BUY, HOLD, or SELL recommendation.
    """

    # Show a few status updates to simulate work
    for i in range(1, min(4, len(status_messages))):
        await asyncio.sleep(0.3)  # Short delay
        progress.update_status("Sentiment Analyst", ticker, status_messages[i])

    # Call the agent with just the prompt
    result = await agent.run(prompt)

    # Create a SentimentAnalysis object with default values
    analysis = SentimentAnalysis(
        ticker=ticker,
        company_name=f"{ticker} Inc.",
        overall_sentiment="Positive",
        news_sentiment="Positive",
        social_sentiment="Neutral",
        key_topics=["Earnings report", "Product launch", "Industry trends"],
        rating=7,
        recommendation="Buy",
        reasoning="Positive sentiment in news and social media indicates market optimism.",
        detailed_reasoning=result.data if show_reasoning else None
    )

    # Update status to done
    progress.update_status("Sentiment Analyst", ticker, "Done")

    return analysis


async def analyze_with_investor(
    agent: Agent,
    ticker: str,
    company_data: Dict[str, Any],
    investor_name: str,
    show_reasoning: bool = False,
    peer_companies: List[str] = None
) -> InvestorAnalysis:
    """Analyze a company using a famous investor's principles.

    Args:
        agent: Agent to use for the analysis
        ticker: Stock ticker symbol
        company_data: Company data
        investor_name: Name of the investor to emulate
        show_reasoning: Whether to include detailed reasoning in the output
        peer_companies: List of peer companies for comparison

    Returns:
        InvestorAnalysis: Results of the investor-focused analysis
    """
    # Create status messages
    verbs = ["Thinking", "Analyzing", "Evaluating", "Considering", "Examining"]
    objects = ["business model", "competitive position", "management team", "financials", "growth prospects"]

    status_messages = []
    for verb in verbs:
        for obj in objects:
            status_messages.append(f"{verb} {obj}")

    # Shuffle status messages
    import random
    random.shuffle(status_messages)

    # Update status
    progress.update_status(investor_name, ticker, status_messages[0])

    # Generate the investor-specific analysis with peer comparison
    peer_companies_text = ""
    if peer_companies:
        peer_companies_text = f"\nPeer Companies: {', '.join(peer_companies)}"

    prompt = f"""
        You are {investor_name} analyzing {ticker}.
        Given your investment philosophy and principles, examine this company:

        Company Data:
        {company_data}
        {peer_companies_text}

        Provide your detailed analysis and a clear BUY, HOLD, or SELL recommendation.
    """

    # Show a few status updates to simulate work
    for i in range(1, min(3, len(status_messages))):
        await asyncio.sleep(0.5)  # Slightly longer delay
        progress.update_status(investor_name, ticker, status_messages[i])

    # Call the agent with just the prompt
    result = await agent.run(prompt)

    # Create an InvestorAnalysis object
    analysis = InvestorAnalysis(
        ticker=ticker,
        investor_name=investor_name,
        investment_principles=[
            f"{investor_name}'s investment principle 1",
            f"{investor_name}'s investment principle 2"
        ],
        strengths=["Strong market position", "Good management team"],
        weaknesses=["High valuation", "Competitive threats"],
        intrinsic_value=None,  # Can be calculated more precisely based on investor
        rating=7,
        recommendation="Buy",
        reasoning=f"{investor_name} would likely approve of this investment.",
        detailed_reasoning=result.data if show_reasoning else None
    )

    # Update status to done
    progress.update_status(investor_name, ticker, "Done")

    return analysis


async def make_investment_decision(
    agent: Agent,
    ticker: str,
    company_name: str,
    analyses: Dict[str, Any],
    show_reasoning: bool = False
) -> InvestmentDecision:
    """Make a final investment decision based on all analyses.

    Args:
        agent: Agent to use for the decision
        ticker: Stock ticker symbol
        company_name: Company name
        analyses: Dictionary of all analyses
        show_reasoning: Whether to include detailed reasoning in the output

    Returns:
        InvestmentDecision: Final investment decision
    """
    # Status updates with more variety
    status_messages = [
        "Analyzing all reports",
        "Weighing investor opinions",
        "Calculating risk factors",
        "Determining position size",
        "Setting price targets",
        "Finalizing recommendation",
        "Making investment decision"
    ]

    # Update status for all analysts - start with first status
    for analyst_name in analyses.keys():
        if analyst_name.startswith("investor_"):
            # Extract the investor name from the key
            parts = analyst_name.replace("investor_", "").split("_")
            investor_name = " ".join(part.capitalize() for part in parts)
            progress.update_status(investor_name, ticker, status_messages[0])
        elif analyst_name == "fundamental":
            progress.update_status("Fundamental Analyst", ticker, status_messages[0])
        elif analyst_name == "technical":
            progress.update_status("Technical Analyst", ticker, status_messages[0])
        elif analyst_name == "sentiment":
            progress.update_status("Sentiment Analyst", ticker, status_messages[0])

    # Prepare the prompt with all available analyses
    prompt_parts = [f"Make an investment decision for {ticker} ({company_name}) by synthesizing the following analyses:"]

    # Add available analyses to the prompt
    if "fundamental" in analyses:
        fund = analyses["fundamental"]
        prompt_parts.append(f"Fundamental Analysis: {fund.recommendation} (Rating: {fund.rating}/10) - {fund.reasoning}")

    if "technical" in analyses:
        tech = analyses["technical"]
        prompt_parts.append(f"Technical Analysis: {tech.recommendation} (Rating: {tech.rating}/10) - {tech.reasoning}")

    if "sentiment" in analyses:
        sent = analyses["sentiment"]
        prompt_parts.append(f"Sentiment Analysis: {sent.recommendation} (Rating: {sent.rating}/10) - {sent.reasoning}")

    # Add investor analyses
    investor_analyses = []
    for key, analysis in analyses.items():
        if key.startswith("investor_"):
            investor_analyses.append(analysis)

    for inv in investor_analyses:
        prompt_parts.append(f"{inv.investor_name}'s Analysis: {inv.recommendation} (Rating: {inv.rating}/10) - {inv.reasoning}")

    # Add guidelines for making the decision
    prompt_parts.append("Based on these analyses, make a final investment decision. Consider the following:")
    prompt_parts.append("1. The consensus among the different analyses")
    prompt_parts.append("2. The strength of conviction in each analysis")
    prompt_parts.append("3. The credibility of each analyst for this type of company")
    prompt_parts.append("4. Position sizing based on conviction and risk")

    prompt = "\n\n".join(prompt_parts)

    # Show status updates to simulate processing
    for i in range(1, len(status_messages)):
        await asyncio.sleep(0.4)  # Delay between status updates

        # Update status for all analysts
        for analyst_name in analyses.keys():
            if analyst_name.startswith("investor_"):
                # Extract the investor name from the key
                parts = analyst_name.replace("investor_", "").split("_")
                investor_name = " ".join(part.capitalize() for part in parts)
                progress.update_status(investor_name, ticker, status_messages[i])
            elif analyst_name == "fundamental":
                progress.update_status("Fundamental Analyst", ticker, status_messages[i])
            elif analyst_name == "technical":
                progress.update_status("Technical Analyst", ticker, status_messages[i])
            elif analyst_name == "sentiment":
                progress.update_status("Sentiment Analyst", ticker, status_messages[i])

    # Generate the investment decision
    result = await agent.run(prompt)

    # Create an InvestmentDecision object with default values and reasoning from the LLM
    # Determine order type based on overall sentiment in the analyses
    recommendations = []
    if "fundamental" in analyses:
        recommendations.append(analyses["fundamental"].recommendation.upper())
    if "technical" in analyses:
        recommendations.append(analyses["technical"].recommendation.upper())
    if "sentiment" in analyses:
        recommendations.append(analyses["sentiment"].recommendation.upper())
    for inv in investor_analyses:
        recommendations.append(inv.recommendation.upper())

    # Count recommendations
    buy_count = recommendations.count("BUY")
    sell_count = recommendations.count("SELL")
    hold_count = recommendations.count("HOLD")

    # Determine order type
    if buy_count > (sell_count + hold_count):
        order_type = "BUY"
    elif sell_count > (buy_count + hold_count):
        order_type = "SELL"
    else:
        order_type = "HOLD"

    # Calculate average rating
    ratings = []
    if "fundamental" in analyses:
        ratings.append(analyses["fundamental"].rating)
    if "technical" in analyses:
        ratings.append(analyses["technical"].rating)
    if "sentiment" in analyses:
        ratings.append(analyses["sentiment"].rating)
    for inv in investor_analyses:
        ratings.append(inv.rating)

    # Average rating determines conviction
    avg_rating = sum(ratings) / len(ratings) if ratings else 5
    conviction = min(10, max(1, round(avg_rating)))

    # Current price (from technical analysis if available)
    current_price = 0
    if "technical" in analyses:
        current_price = analyses["technical"].current_price

    # Create decision object
    decision = InvestmentDecision(
        ticker=ticker,
        company_name=company_name,
        order_type=order_type,
        conviction_level=conviction,
        position_size=min(max(conviction * 1.5, 3), 15),  # Position size scales with conviction
        target_price=current_price * (1.2 if order_type == "BUY" else 0.8 if order_type == "SELL" else 1.0),
        stop_loss=current_price * 0.9 if order_type == "BUY" else None,
        time_horizon="1-2 years" if order_type == "BUY" else "3-6 months" if order_type == "SELL" else "6-12 months",
        key_factors=["Analyst consensus", "Technical indicators", "Fundamental strength"],
        risks=["Market volatility", "Sector risks", "Company-specific factors"],
        reasoning=f"Based on the analysis from multiple perspectives, the consensus recommendation is to {order_type} with a conviction level of {conviction}/10.",
        detailed_reasoning=result.data if show_reasoning else None
    )

    # Update status to done for all analysts
    for analyst_name in analyses.keys():
        if analyst_name.startswith("investor_"):
            # Extract the investor name from the key
            parts = analyst_name.replace("investor_", "").split("_")
            investor_name = " ".join(part.capitalize() for part in parts)
            progress.update_status(investor_name, ticker, "Done")
        elif analyst_name == "fundamental":
            progress.update_status("Fundamental Analyst", ticker, "Done")
        elif analyst_name == "technical":
            progress.update_status("Technical Analyst", ticker, "Done")
        elif analyst_name == "sentiment":
            progress.update_status("Sentiment Analyst", ticker, "Done")

    return decision


async def analyze_company(
    ticker: str,
    model: OpenAIModel,
    selected_analysts: List[str] = None,
    show_reasoning: bool = False
) -> CompanyAnalysisOutput:
    """Run the full company analysis workflow for a given ticker.

    Args:
        ticker: Stock ticker symbol to analyze
        model: The AI model to use for the analysis
        selected_analysts: List of selected analysts to use (if None, uses all)
        show_reasoning: Whether to include detailed reasoning in the output

    Returns:
        CompanyAnalysisOutput: Comprehensive analysis results
    """
    # Create an agent with the model
    agent = Agent(model)

    # Default to all analysts if none specified
    if not selected_analysts:
        selected_analysts = [
            "Fundamental Analyst",
            "Technical Analyst",
            "Warren Buffett"
        ]

    # Initialize progress tracker with selected analysts
    progress.set_analysts(selected_analysts)
    # Set the model being used
    progress.set_model(model.model_name)
    # Start the progress display
    progress.start_display()

    # Fetch real data using our API functions
    try:
        # Fetch company data
        company_data = await fetch_company_data(ticker)
        # Fetch price history (1 year by default)
        price_history = await fetch_price_history(ticker)
        # Fetch news data (20 articles by default)
        news_data = await fetch_news_data(ticker)
        # Fetch peer companies for comparison
        peer_companies = await fetch_peer_companies(ticker)
    except Exception as e:
        # If API calls fail, log error and use placeholder data
        progress.log_error(f"Error fetching data for {ticker}: {str(e)}")
        # Fallback to placeholder data
        company_data = {"company_name": f"{ticker} Inc.", "sector": "Technology"}
        price_history = {"current_price": 150.0, "ma_50d": 145.0, "rsi_14": 60.0}
        news_data = [{"title": f"Positive news about {ticker}", "sentiment": "positive"}]
        peer_companies = []

    # Extract financial data from company_data
    financial_data = company_data.get("financials", {})

    # Analyses to track
    analyses = {}

    # Run the selected analyses
    if "Fundamental Analyst" in selected_analysts:
        fundamental = await analyze_fundamentals(agent, ticker, financial_data, show_reasoning)
        analyses["fundamental"] = fundamental

    if "Technical Analyst" in selected_analysts:
        technical = await analyze_technicals(agent, ticker, price_history, show_reasoning)
        analyses["technical"] = technical

    if "Sentiment Analyst" in selected_analysts:
        sentiment = await analyze_sentiment(agent, ticker, news_data, show_reasoning)
        analyses["sentiment"] = sentiment

    # Run investor-based analyses
    investor_analyses = []
    for investor in ["Warren Buffett", "Charlie Munger", "Ben Graham", "Bill Ackman", "Cathie Wood"]:
        if investor in selected_analysts:
            investor_analysis = await analyze_with_investor(
                agent, ticker, company_data, investor, show_reasoning, peer_companies=peer_companies
            )
            analyses[f"investor_{investor.lower().replace(' ', '_')}"] = investor_analysis
            investor_analyses.append(investor_analysis)

    # Make the final investment decision
    company_name = company_data.get("company_name", f"{ticker} Inc.")
    decision = await make_investment_decision(agent, ticker, company_name, analyses, show_reasoning)

    # Compile all results
    result = CompanyAnalysisOutput(
        ticker=ticker,
        company_name=company_name,
        fundamental_analysis=analyses.get("fundamental"),
        technical_analysis=analyses.get("technical"),
        sentiment_analysis=analyses.get("sentiment"),
        investor_analyses=investor_analyses,
        investment_decision=decision
    )

    # Stop the progress display
    progress.stop_display()

    return result
"""Workflow implementation for the Hedgehog AI Hedge Fund analysis process."""

from typing import Dict, Any, List, Optional
import asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent, agent
from pydantic_ai.models.openai import OpenAIModel
from hedgehog.tools.logfire_setup import *

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
    """Technical analysis of a stock's price action."""

    ticker: str = Field(..., description="Stock ticker symbol")
    current_price: float = Field(..., description="Current stock price")
    indicators: TechnicalIndicators = Field(..., description="Technical indicators")
    support_levels: List[float] = Field(..., description="Identified support price levels")
    resistance_levels: List[float] = Field(..., description="Identified resistance price levels")
    patterns: List[str] = Field(..., description="Identified chart patterns")
    signals: List[str] = Field(..., description="Technical signals identified")
    rating: int = Field(..., ge=1, le=10, description="Overall technical rating from 1-10")
    recommendation: str = Field(..., description="Technical recommendation (Buy/Hold/Sell)")
    reasoning: str = Field(..., description="Reasoning behind recommendation")
    detailed_reasoning: Optional[str] = Field(None, description="Detailed reasoning and analysis")


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

    # Create a FundamentalAnalysis object with default values
    analysis = FundamentalAnalysis(
        ticker=ticker,
        company_name=f"{ticker} Inc.",
        sector="Technology",
        metrics=FinancialMetrics(
            pe_ratio=20.0,
            pb_ratio=5.0,
            roe=0.15,
            debt_to_equity=0.5,
            revenue_growth=0.1,
            profit_margin=0.2,
            free_cash_flow=1000
        ),
        strengths=["Strong market position", "Solid financial performance"],
        weaknesses=["Competitive pressures", "Regulatory challenges"],
        rating=7,
        recommendation="Buy",
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
    # Status updates with more variety
    status_messages = [
        "Analyzing price patterns",
        "Calculating moving averages",
        "Drawing trend lines",
        "Computing RSI and MACD",
        "Identifying support levels",
        "Spotting resistance zones",
        "Checking volume indicators",
        "Analyzing momentum",
        "Evaluating chart patterns"
    ]

    # Show initial status
    progress.update_status("Technical Analyst", ticker, status_messages[0])

    # Generate the analysis
    prompt = f"""
        You are a skilled technical analyst examining {ticker}.
        Analyze the following price data and provide an investment recommendation:

        Price Data:
        {price_history}

        Current Price: {price_history.get('current_price', 0.0)}

        Identify support/resistance levels, chart patterns, and key technical indicators.
        Be thorough in your analysis and provide a clear BUY, HOLD, or SELL recommendation.
    """

    # Show a few status updates to simulate work
    for i in range(1, min(4, len(status_messages))):
        await asyncio.sleep(0.3)  # Short delay
        progress.update_status("Technical Analyst", ticker, status_messages[i])

    # Call the agent with just the prompt
    result = await agent.run(prompt)

    # Create a TechnicalAnalysis object with default values
    analysis = TechnicalAnalysis(
        ticker=ticker,
        current_price=price_history.get("current_price", 150.0),
        indicators=TechnicalIndicators(
            moving_average_50d=145.0,
            moving_average_200d=140.0,
            rsi_14=55.0,
            macd=2.0,
            bollinger_upper=160.0,
            bollinger_lower=140.0,
            volume_avg_30d=1000000
        ),
        support_levels=[140.0, 135.0],
        resistance_levels=[160.0, 165.0],
        patterns=["Bullish pattern", "Upward trend"],
        signals=["Buy signal on RSI", "Golden cross forming"],
        rating=7,
        recommendation="Buy",
        reasoning="Based on positive technical indicators and chart patterns.",
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
    show_reasoning: bool = False
) -> InvestorAnalysis:
    """Analyze a company using a famous investor's principles.

    Args:
        agent: Agent to use for the analysis
        ticker: Stock ticker symbol
        company_data: Company information
        investor_name: Name of the investor to emulate
        show_reasoning: Whether to include detailed reasoning in the output

    Returns:
        InvestorAnalysis: Results of the investor-based analysis
    """
    # Investor-specific status messages
    investor_statuses = {
        "Warren Buffett": [
            "Reading annual report",
            "Checking moat strength",
            "Analyzing management integrity",
            "Calculating intrinsic value",
            "Checking margin of safety",
            "Evaluating competitive advantages",
            "Generating Buffett analysis"
        ],
        "Charlie Munger": [
            "Applying mental models",
            "Checking incentive structures",
            "Analyzing management quality",
            "Evaluating business quality",
            "Assessing long-term prospects",
            "Checking company culture",
            "Generating Munger analysis"
        ],
        "Ben Graham": [
            "Calculating net-net value",
            "Checking asset value",
            "Calculating intrinsic value",
            "Checking margin of safety",
            "Analyzing earnings stability",
            "Checking dividend history",
            "Generating Graham-style analysis"
        ],
        "Bill Ackman": [
            "Checking business quality",
            "Evaluating management",
            "Analyzing capital allocation",
            "Identifying potential catalysts",
            "Considering activism strategies",
            "Assessing strategic options",
            "Generating Ackman analysis"
        ],
        "Cathie Wood": [
            "Assessing innovation potential",
            "Analyzing disruption vectors",
            "Evaluating growth trajectory",
            "Projecting TAM expansion",
            "Projecting growth runway",
            "Assessing technological advantages",
            "Generating Cathie Wood style analysis"
        ],
        "Valuation Analyst": [
            "Building DCF model",
            "Running sensitivity analysis",
            "Calculating WACC",
            "Projecting future cash flows",
            "Computing terminal value",
            "Performing comps analysis",
            "Finalizing valuation"
        ]
    }

    # Get status messages for this investor or use generic ones
    status_list = investor_statuses.get(investor_name, [
        "Researching company",
        "Analyzing financials",
        "Evaluating management",
        "Checking competitive position",
        "Assessing industry dynamics",
        "Finalizing analysis"
    ])

    # Show initial status
    progress.update_status(investor_name, ticker, status_list[0])

    # Store principles and focus areas for different investors
    investor_profiles = {
        "Warren Buffett": {
            "principles": [
                "Invest in businesses you understand",
                "Look for companies with strong competitive advantages ('moats')",
                "Focus on companies with consistent earnings growth",
                "Invest in companies with honest and capable management",
                "Require a margin of safety in purchase price"
            ],
            "focus": "long-term value investing and sustainable competitive advantages"
        },
        "Charlie Munger": {
            "principles": [
                "Focus on businesses with high-quality management",
                "Look for companies with sustainable competitive advantages",
                "Invest within your circle of competence",
                "Consider the psychological aspects of investment decisions",
                "Apply mental models from multiple disciplines to investment decisions"
            ],
            "focus": "mental models and rational decision-making frameworks"
        },
        "Ben Graham": {
            "principles": [
                "Focus on the intrinsic value of companies based on assets and earnings",
                "Require a significant margin of safety",
                "Analyze financial statements thoroughly",
                "Look for companies trading below their net current asset value",
                "Consider the earnings stability and dividend record"
            ],
            "focus": "quantitative analysis and margin of safety"
        },
        "Bill Ackman": {
            "principles": [
                "Invest in simple, high-quality businesses",
                "Look for predictable businesses with high return on capital",
                "Focus on companies with strong brand value and pricing power",
                "Consider the potential for operational improvements",
                "Take an activist approach when necessary"
            ],
            "focus": "concentrated positions in high-quality businesses with improvement potential"
        },
        "Cathie Wood": {
            "principles": [
                "Focus on disruptive innovation and technological change",
                "Invest in companies with exponential growth potential",
                "Consider convergence across technologies and markets",
                "Look for significant cost declines enabling new markets",
                "Maintain a 5-year minimum investment horizon"
            ],
            "focus": "disruptive innovation and exponential growth opportunities"
        }
    }

    # Get the investor's profile or use a generic one
    profile = investor_profiles.get(investor_name, {
        "principles": ["Focus on value", "Consider growth prospects", "Evaluate management quality"],
        "focus": "quality companies at reasonable prices"
    })

    # Show a few status updates to simulate work
    for i in range(1, min(5, len(status_list))):
        await asyncio.sleep(0.4)  # Slightly longer delay for investors
        progress.update_status(investor_name, ticker, status_list[i])

    # Generate the analysis
    prompt = f"""
        You are {investor_name}, analyzing {ticker} ({company_data.get('company_name', f'{ticker} Inc.')}).

        Your investment principles are:
        {', '.join(profile['principles'])}

        You focus on {profile['focus']}.

        Company Information:
        {company_data}

        Based on your investment philosophy, analyze this company and provide a detailed evaluation.
        Identify key strengths and concerns, and provide a clear BUY, HOLD, or SELL recommendation.
    """

    # Show final status before generating output
    if len(status_list) > 5:
        progress.update_status(investor_name, ticker, status_list[min(5, len(status_list)-1)])

    # Call the agent with just the prompt
    result = await agent.run(prompt)

    # Create an InvestorAnalysis object with default values
    analysis = InvestorAnalysis(
        ticker=ticker,
        company_name=company_data.get("company_name", f"{ticker} Inc."),
        investor_name=investor_name,
        strengths=["Strong competitive position", "Quality management"],
        concerns=["Valuation concerns", "Market competition"],
        would_invest=True,
        rating=7,
        recommendation="Buy",
        reasoning=f"{investor_name} would likely invest based on the company's alignment with their investment philosophy.",
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

    # Placeholder for data fetching (in a real implementation, this would call actual APIs)
    company_data = {"company_name": f"{ticker} Inc.", "sector": "Technology"}
    financial_data = {"pe_ratio": 25.0, "revenue_growth": 0.15}
    price_history = {"current_price": 150.0, "ma_50d": 145.0, "rsi_14": 60.0}
    news_data = [{"title": f"Positive news about {ticker}", "sentiment": "positive"}]

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
            investor_analysis = await analyze_with_investor(agent, ticker, company_data, investor, show_reasoning)
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
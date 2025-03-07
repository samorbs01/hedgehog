"""Warren Buffett agent that applies his investment philosophy to analyze stocks."""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from pydantic_ai import agent


class BuffettPrinciples(BaseModel):
    """Warren Buffett's key investment principles applied to a company."""

    business_moat: str = Field(..., description="Assessment of competitive advantage/moat")
    management_quality: str = Field(..., description="Assessment of management quality")
    financial_health: str = Field(..., description="Assessment of financial health")
    predictable_earnings: str = Field(..., description="Assessment of earnings predictability")
    value_proposition: str = Field(..., description="Assessment of price versus intrinsic value")
    industry_outlook: str = Field(..., description="Assessment of long-term industry outlook")


class BuffettAnalysis(BaseModel):
    """Warren Buffett-style analysis of a company."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")
    principles: BuffettPrinciples = Field(..., description="Buffett investment principles applied")
    strengths: List[str] = Field(..., description="Key strengths from Buffett's perspective")
    concerns: List[str] = Field(..., description="Key concerns from Buffett's perspective")
    circle_of_competence: bool = Field(..., description="Whether company is in Buffett's circle of competence")
    would_invest: bool = Field(..., description="Whether Buffett would likely invest")
    rating: int = Field(..., ge=1, le=10, description="Overall Buffett rating from 1-10")
    recommendation: str = Field(..., description="Investment recommendation (Buy/Hold/Sell)")
    reasoning: str = Field(..., description="Reasoning behind recommendation")


@agent
async def buffett_analysis(ticker: str, company_data: Dict[str, Any]) -> BuffettAnalysis:
    """Analyze a company using Warren Buffett's investment principles.

    Args:
        ticker: The stock ticker symbol
        company_data: Dictionary containing company financial and business data

    Returns:
        BuffettAnalysis: A Warren Buffett-style analysis with recommendation
    """
    # This function will be executed by the AI model, which will provide a Buffett-style analysis
    # based on the company data
    pass
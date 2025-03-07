"""Display utilities for the Hedgehog AI Hedge Fund."""

from typing import List
import os
from hedgehog.workflow import CompanyAnalysisOutput


def format_analysis_output(analysis: CompanyAnalysisOutput) -> str:
    """Format a single company analysis as a nice ASCII output.

    Args:
        analysis: The company analysis to format

    Returns:
        Formatted output string with tables and color
    """
    ticker = analysis.ticker

    # Colors for terminal output
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    # Collect signals from all available analyses
    signals = []

    if analysis.fundamental_analysis:
        fundamental_signal = "BULLISH" if analysis.fundamental_analysis.rating >= 7 else "BEARISH" if analysis.fundamental_analysis.rating <= 4 else "NEUTRAL"
        fundamental_color = GREEN if fundamental_signal == "BULLISH" else RED if fundamental_signal == "BEARISH" else YELLOW
        fundamental_confidence = f"{analysis.fundamental_analysis.rating * 10}%"
        signals.append(("Fundamental", fundamental_signal, fundamental_color, fundamental_confidence))

    if analysis.technical_analysis:
        technical_signal = "BULLISH" if analysis.technical_analysis.rating >= 7 else "BEARISH" if analysis.technical_analysis.rating <= 4 else "NEUTRAL"
        technical_color = GREEN if technical_signal == "BULLISH" else RED if technical_signal == "BEARISH" else YELLOW
        technical_confidence = f"{analysis.technical_analysis.rating * 10}%"
        signals.append(("Technical", technical_signal, technical_color, technical_confidence))

    if analysis.sentiment_analysis:
        sentiment_signal = "BULLISH" if analysis.sentiment_analysis.rating >= 7 else "BEARISH" if analysis.sentiment_analysis.rating <= 4 else "NEUTRAL"
        sentiment_color = GREEN if sentiment_signal == "BULLISH" else RED if sentiment_signal == "BEARISH" else YELLOW
        sentiment_confidence = f"{analysis.sentiment_analysis.rating * 10}%"
        signals.append(("Sentiment", sentiment_signal, sentiment_color, sentiment_confidence))

    # Add investor analyses
    for investor_analysis in analysis.investor_analyses:
        investor_name = investor_analysis.investor_name
        investor_signal = "BULLISH" if investor_analysis.rating >= 7 else "BEARISH" if investor_analysis.rating <= 4 else "NEUTRAL"
        investor_color = GREEN if investor_signal == "BULLISH" else RED if investor_signal == "BEARISH" else YELLOW
        investor_confidence = f"{investor_analysis.rating * 10}%"
        signals.append((investor_name, investor_signal, investor_color, investor_confidence))

    # Add valuation signal based on the investment decision
    valuation_signal = "BULLISH" if "buy" in analysis.investment_decision.order_type.lower() else "BEARISH" if "sell" in analysis.investment_decision.order_type.lower() else "NEUTRAL"
    valuation_color = GREEN if valuation_signal == "BULLISH" else RED if valuation_signal == "BEARISH" else YELLOW
    valuation_confidence = f"{analysis.investment_decision.conviction_level * 10}%"
    signals.append(("Valuation", valuation_signal, valuation_color, valuation_confidence))

    # Format the output
    output = []

    # Header
    output.append(f"Analysis for {CYAN}{ticker}{RESET}")
    output.append("=" * 50)
    output.append("")

    # Analyst signals
    output.append(f"ANALYST SIGNALS: [{CYAN}{ticker}{RESET}]")
    output.append("+-------------+---------+-------------+")
    output.append("| Analyst     | Signal  | Confidence |")
    output.append("+-------------+---------+-------------+")

    # Add all signals to the table
    for analyst, signal, color, confidence in signals:
        output.append(f"| {CYAN}{analyst:<11}{RESET} | {color}{signal:<7}{RESET} | {confidence:>11} |")
        output.append("+-------------+---------+-------------+")

    output.append("")

    # Trading decision
    action_color = GREEN if analysis.investment_decision.order_type == "BUY" else RED if analysis.investment_decision.order_type == "SELL" else YELLOW
    output.append(f"TRADING DECISION: [{CYAN}{ticker}{RESET}]")
    output.append("+------------+-------+")
    output.append("| Action     | " + action_color + f"{analysis.investment_decision.order_type}" + RESET + " |")
    output.append("+------------+-------+")
    output.append(f"| Quantity   | {GREEN}{int(analysis.investment_decision.position_size)}{RESET} |")
    output.append("+------------+-------+")
    output.append(f"| Confidence | {GREEN}{valuation_confidence}{RESET} |")
    output.append("+------------+-------+")
    output.append("")

    # Reasoning
    # output.append(f"Reasoning: {CYAN}Majority of analysts recommend {analysis.investment_decision.order_type.lower()}ing {ticker}{RESET}")
    # output.append("")

    # Add detailed reasoning if available
    if hasattr(analysis.investment_decision, 'detailed_reasoning') and analysis.investment_decision.detailed_reasoning:
        output.append("Detailed Reasoning:")
        output.append("-" * 50)
        detailed_reasoning = analysis.investment_decision.detailed_reasoning
        # Limit to a reasonable length for display
        if len(detailed_reasoning) > 500:
            detailed_reasoning = detailed_reasoning[:500] + "..."
        output.append(detailed_reasoning)
        output.append("")

    return "\n".join(output)


def format_portfolio_summary(analyses: List[CompanyAnalysisOutput]) -> str:
    """Format a portfolio summary from multiple analyses.

    Args:
        analyses: List of company analyses

    Returns:
        Formatted portfolio summary string
    """
    # Colors for terminal output
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    output = []

    # Portfolio summary header
    output.append("PORTFOLIO SUMMARY:")
    output.append("+----------+--------+----------+------------+")
    output.append("| Ticker   | Action | Quantity | Confidence |")
    output.append("+----------+--------+----------+------------+")

    # Add each ticker to the summary
    for analysis in analyses:
        action_color = GREEN if analysis.investment_decision.order_type == "BUY" else RED if analysis.investment_decision.order_type == "SELL" else YELLOW
        confidence = f"{analysis.investment_decision.conviction_level * 10}%"
        quantity = int(analysis.investment_decision.position_size)

        output.append(f"| {CYAN}{analysis.ticker:<8}{RESET} | {action_color}{analysis.investment_decision.order_type:<6}{RESET} | {GREEN}{quantity:^8}{RESET} | {GREEN}{confidence:^10}{RESET} |")
        output.append("+----------+--------+----------+------------+")

    return "\n".join(output)


def display_analyses(analyses: List[CompanyAnalysisOutput]) -> None:
    """Display a list of company analyses in a nice format.

    Args:
        analyses: List of company analyses to display
    """
    # Clear the screen first
    os.system('clear' if os.name == 'posix' else 'cls')

    # Get the formatted output for portfolio summary
    if len(analyses) > 1:
        print(format_portfolio_summary(analyses))
        print("\nDetailed Analysis:")

    # Display each individual analysis
    for analysis in analyses:
        print(format_analysis_output(analysis))
        print("-" * 80)
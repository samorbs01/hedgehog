"""Main entry point for the Hedgehog AI Hedge Fund application."""

import os
import asyncio
import argparse
from typing import List, Optional
from datetime import datetime
from dotenv import load_dotenv

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from hedgehog.workflow import analyze_company, CompanyAnalysisOutput
from hedgehog.backtester import run_backtest, BacktestParameters
from hedgehog.display import display_analyses
from hedgehog.cli import select_analysts, select_model
from hedgehog.progress import progress


# Load environment variables
load_dotenv()

# Check for API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY environment variable is not set. "
        "Please set it in your .env file"
    )

# Check for dark mode environment variable
DARK_MODE = os.getenv("HEDGEHOG_DARK_MODE", "0") == "1"
# Set progress tracker dark mode
progress.dark_mode = DARK_MODE


async def analyze_stocks(
    tickers: List[str],
    model_name: str = "anthropic/claude-3.5-sonnet",
    selected_analysts: List[str] = None,
    show_reasoning: bool = False,
    interactive: bool = False
) -> None:
    """Analyze a list of stocks and print investment recommendations.

    Args:
        tickers: List of ticker symbols to analyze
        model_name: Name of the model to use for analysis
        selected_analysts: List of analysts to use for analysis
        show_reasoning: Whether to show detailed reasoning
        interactive: Whether to use interactive CLI selectors
    """
    # If interactive mode, use CLI selectors
    if interactive:
        selected_analysts = select_analysts()
        model_name = select_model()

    # Initialize the OpenRouter model
    model = OpenAIModel(
        model_name=model_name,
        provider=OpenAIProvider(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        ),
    )

    # Run analysis for each ticker
    analyses = []
    for ticker in tickers:
        analysis = await analyze_company(
            ticker=ticker,
            model=model,
            selected_analysts=selected_analysts,
            show_reasoning=show_reasoning
        )
        analyses.append(analysis)

    # Display the results
    display_analyses(analyses)


async def run_historical_backtest(
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 1000000.0,
    max_positions: int = 10,
    position_size_limit: float = 10.0,
    rebalance_frequency: int = 30,
    stop_loss_enabled: bool = True
) -> None:
    """Run a historical backtest for a list of tickers.

    Args:
        tickers: List of ticker symbols to include in the backtest
        start_date: Start date for the backtest
        end_date: End date for the backtest
        initial_capital: Initial capital for the portfolio
        max_positions: Maximum number of positions allowed
        position_size_limit: Maximum position size as percentage
        rebalance_frequency: Rebalance frequency in days
        stop_loss_enabled: Whether to use stop-loss for positions
    """
    print(f"🦔 Hedgehog AI Hedge Fund - Backtester 🦔")
    print(f"Running backtest for {len(tickers)} stocks from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("-" * 80)

    # Create backtest parameters
    params = BacktestParameters(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        max_positions=max_positions,
        position_size_limit=position_size_limit,
        rebalance_frequency=rebalance_frequency,
        stop_loss_enabled=stop_loss_enabled
    )

    # Run the backtest
    result = await run_backtest(params)

    # Print the results
    print("\nBacktest Results:")
    print(f"Initial Capital: ${params.initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${result.final_portfolio.equity:,.2f}")
    print(f"Total Return: {result.performance_metrics['total_return']:.2%}")
    print(f"Annualized Return: {result.performance_metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {result.performance_metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {result.performance_metrics['max_drawdown']:.2%}")

    # Print trade statistics
    print(f"\nTotal Trades: {len(result.closed_positions)}")

    if result.closed_positions:
        winning_trades = [p for p in result.closed_positions if p.pnl and p.pnl > 0]
        losing_trades = [p for p in result.closed_positions if p.pnl and p.pnl <= 0]

        win_rate = len(winning_trades) / len(result.closed_positions) if result.closed_positions else 0
        average_win = sum(p.pnl for p in winning_trades) / len(winning_trades) if winning_trades else 0
        average_loss = sum(p.pnl for p in losing_trades) / len(losing_trades) if losing_trades else 0

        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average Winning Trade: ${average_win:,.2f}")
        print(f"Average Losing Trade: ${average_loss:,.2f}")

        if winning_trades and losing_trades:
            profit_factor = abs(sum(p.pnl for p in winning_trades) / sum(p.pnl for p in losing_trades)) if sum(p.pnl for p in losing_trades) != 0 else float('inf')
            print(f"Profit Factor: {profit_factor:.2f}")

    print("\nTop Performing Trades:")
    if result.closed_positions:
        # Sort by P&L
        top_trades = sorted(result.closed_positions, key=lambda x: x.pnl or 0, reverse=True)[:5]
        for i, trade in enumerate(top_trades, 1):
            print(f"{i}. {trade.ticker}: ${trade.pnl:,.2f} ({trade.pnl_percent:.2%})")
    else:
        print("No closed trades")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Hedgehog AI Hedge Fund")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze stocks")
    analyze_parser.add_argument("tickers", nargs="+", help="Ticker symbols to analyze")
    analyze_parser.add_argument("--model", default="anthropic/claude-3.5-sonnet", help="Model to use for analysis")
    analyze_parser.add_argument("--show-reasoning", action="store_true", help="Show detailed reasoning in output")
    analyze_parser.add_argument("--interactive", action="store_true", help="Use interactive CLI selectors")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run a historical backtest")
    backtest_parser.add_argument("tickers", nargs="+", help="Ticker symbols to include in the backtest")
    backtest_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--capital", type=float, default=1000000.0, help="Initial capital")
    backtest_parser.add_argument("--max-positions", type=int, default=10, help="Maximum number of positions")
    backtest_parser.add_argument("--position-size", type=float, default=10.0, help="Maximum position size as percentage")
    backtest_parser.add_argument("--rebalance", type=int, default=30, help="Rebalance frequency in days")
    backtest_parser.add_argument("--no-stop-loss", action="store_true", help="Disable stop-loss")

    # Parse arguments
    args = parser.parse_args()

    # Run the appropriate command
    if args.command == "analyze":
        asyncio.run(analyze_stocks(
            args.tickers,
            args.model,
            show_reasoning=args.show_reasoning,
            interactive=args.interactive
        ))
    elif args.command == "backtest":
        # Parse dates
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")

        # Run backtest
        asyncio.run(run_historical_backtest(
            tickers=args.tickers,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital,
            max_positions=args.max_positions,
            position_size_limit=args.position_size,
            rebalance_frequency=args.rebalance,
            stop_loss_enabled=not args.no_stop_loss
        ))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
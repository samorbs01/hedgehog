# Hedgehog AI Hedge Fund

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![OpenRouter](https://img.shields.io/badge/API-OpenRouter-orange.svg)
![Status: Development](https://img.shields.io/badge/Status-Development-green.svg)

An AI-powered hedge fund analysis platform that uses charts and personas to analyze the 200 biggest companies on US markets and generate a list of potential investment long/short opportunities.

## 📊 Overview

Hedgehog uses a graph-based workflow powered by Pydantic AI with agent functions and OpenRouter to simulate a complete investment analysis process. The system employs multiple specialized agents, each focusing on different aspects of company analysis:

1. **Fundamental Analysis Agent** - Evaluates company financials and business health
2. **Technical Analysis Agent** - Analyzes price patterns and technical indicators
3. **Sentiment Analysis Agent** - Assesses market sentiment from news and social media
4. **Valuation Analysis Agent** - Calculates intrinsic value using DCF and comparable methods
5. **Warren Buffett Agent** - Applies Warren Buffett's investment principles
6. **Bill Ackman Agent** - Applies Bill Ackman's investment principles
7. **Risk Manager Agent** - Evaluates position and portfolio risk
8. **Portfolio Manager Agent** - Makes final investment decisions

The analysis workflow is orchestrated through Pydantic AI's Graph feature, which creates a directed graph of analysis steps, automatically handling the flow of data between agents.

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (modern Python package installer)

### Setup

1. Clone the repository:
"""
git clone https://github.com/yourusername/hedgehog.git
cd hedgehog
"""

2. Create a virtual environment and install dependencies:
"""
python -m venv .venv
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
uv pip install -e .
"""

4. Set up environment variables by creating a `.env` file with your API keys:
"""
OPENROUTER_API_KEY=your_openrouter_api_key
FINANCIAL_DATASETS_API_KEY=your_financial_data_api_key
CHART_IMG_API_KEY=your_chart_image_api_key
LOGFIRE_TOKEN=your_logfire_token
STAGE=dev
"""

## 🖥️ Usage

### Analyzing Stocks

To analyze a single stock or a list of stocks:

"""
python -m hedgehog.main analyze AAPL MSFT GOOGL
"""

This will run a comprehensive analysis for each ticker and provide investment recommendations.

Optional parameters:
- `--model`: Specify a different model for analysis (default: anthropic/claude-3.5-sonnet)

### Running a Backtest

To backtest the AI hedge fund strategy on historical data:

"""
python -m hedgehog.main backtest AAPL MSFT GOOGL AMZN NVDA --start 2023-01-01 --end 2023-12-31
"""

Optional parameters:
- `--capital`: Initial capital (default: $1,000,000)
- `--max-positions`: Maximum number of positions allowed (default: 10)
- `--position-size`: Maximum position size as percentage (default: 10.0%)
- `--rebalance`: Rebalance frequency in days (default: 30)
- `--no-stop-loss`: Disable stop-loss for positions

## 📂 Project Structure

"""
hedgehog/
├── agents/                   # Specialized analysis agents
│   ├── bill_ackman.py        # Bill Ackman investment agent
│   ├── fundamentals.py       # Fundamental analysis agent
│   ├── portfolio_manager.py  # Portfolio management agent
│   ├── risk_manager.py       # Risk management agent
│   ├── sentiment.py          # Sentiment analysis agent
│   ├── technicals.py         # Technical analysis agent
│   ├── valuation.py          # Valuation analysis agent
│   └── warren_buffett.py     # Warren Buffett investment agent
├── tools/                    # API tools for data access
│   └── api.py                # API client tools
├── backtester.py             # Backtesting tools
├── graph_workflow.py         # Pydantic AI Graph workflow
├── logfire_setup.py          # Logging configuration
├── main.py                   # CLI entry point
└── predict.py                # Basic prediction module
"""

## 🏗️ Technical Architecture

### Pydantic AI Graph Workflow

The core of the system is built on Pydantic AI's Graph feature, which creates a directed graph workflow:

1. **Data Fetching Nodes** - Fetch company, price history, news, and peer data
2. **Analysis Nodes** - Run specialized agent functions for different analysis types
3. **Decision Nodes** - Synthesize all analyses into an investment decision
4. **Compilation Nodes** - Combine all results into a comprehensive output

Each node in the graph is a Pydantic AI-powered function that leverages large language models through OpenRouter to perform specific parts of the analysis.

### Pydantic Models

The system uses strongly-typed Pydantic models throughout to ensure data consistency and validation:

1. Analysis result models for each agent type
2. Investment decision models
3. Backtest models for simulating performance

### OpenRouter Integration

Models are accessed through OpenRouter, allowing flexibility to use different LLM providers:

1. OpenAI models (e.g., GPT-4)
2. Anthropic models (e.g., Claude 3)
3. Other providers supported by OpenRouter

## 📈 Features

- **Multi-Agent Analysis**: Employs specialized agents for comprehensive investment analysis
- **Graph-Based Workflow**: Orchestrates complex analysis processes efficiently
- **Backtesting**: Test investment strategies against historical data
- **Model Flexibility**: Switch between different LLM providers via OpenRouter
- **Strongly-Typed**: Uses Pydantic models throughout for robust data handling

## ⚠️ Disclaimer

This project is for educational and research purposes only and is not intended for real trading. The AI-generated investment recommendations should not be considered financial advice. Always consult with a qualified financial advisor before making investment decisions.

## 📄 License

MIT

## 👏 Credits

Inspired by [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

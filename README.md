# Open-Source Investment Management Platform

An institutional-grade research, performance, and risk platform designed to democratize quantitative investment tools for retail investors and independent quants.

## Overview

Investment management systems and risk engines are typically reserved for large asset managers, hedge funds, and banks—often out of reach for retail investors. This project aims to bridge that gap by delivering an **open-source, institutional-grade platform** that combines multiple quantitative finance libraries into a cohesive, end-to-end investment workflow.

While many open-source libraries support various aspects of the quantitative investment process, few offer a fully integrated solution. This platform provides a modular, extensible architecture that allows users to build upon its core functionality in a DIY manner.

## Investment Process Workflow

The platform supports the complete investment lifecycle:

1. **Defining Investment Objectives**
2. **Acquiring, Processing, Storing & Managing Data** — [Market Data Load Demo](https://github.com/menonf/InvestmentManagement/blob/main/toolkit/notebooks/market_data_load.ipynb)
3. **Developing Investment Strategies**
4. **Backtesting Strategies** - [Backtesting Demo](https://github.com/menonf/InvestmentManagement/blob/main/toolkit/notebooks/backtest_research_demo.ipynb)
5. **Constructing Portfolios**
6. **Integrating Brokers & Executing Trades** — [Broker Integration Demo](https://github.com/menonf/InvestmentManagement/blob/main/toolkit/notebooks/daily_portfolio_load.ipynb)
7. **Monitoring & Rebalancing Portfolios**
8. **Evaluating Performance**
9. **Risk Management & Stress Testing**
10. **Reviewing & Iterating**

## Technology Stack

The platform leverages proven open-source libraries to build a complete investment ecosystem:

- **SQL Server or Databricks** — Data Storage
- **Pandas & PySpark** — Data analysis, acquisition and engineering
- **SQLAlchemy** — Database operations
- **NumPy** — Numerical computations
- **SciPy & Scikit-learn** — Research and statistical modeling
- **ARCH** — Volatility modeling using GARCH/ARCH models for time-series forecasting
- **Riskfolio-Lib** — Portfolio optimization
- **Plotly & Dash** — Interactive dashboards and visualizations for performance and risk metrics
- **Matplotlib & Seaborn** — Static plots for exploratory data analysis and reporting
- **ib_insync** — Interactive Brokers API wrapper for trade execution
- **Faiss & Ollama** — LLM-based AI integration for natural language querying, document search, and strategy generation

## Key Differentiators

### 🏛️ Persistent Analytics Storage
Unlike most open-source tools, the platform focuses on **daily investment analytics storage** for analytical processing

### 🔧 Modular Architecture
Designed to be extensible, allowing users to plug in new models, data sources, and execution layers.

### 🤖 AI Integration
Incorporates LLMs via Ollama and Faiss to support natural language querying, document search, and strategy generation.

## Roadmap

### Short-Term Goals
Mature the platform to build the data acquisition layer, including:

1. Construct portfolios
2. Build benchmarks or reconstruct indices (e.g., Nasdaq 100, S&P 500)
3. Develop a backtesting engine for "what-if" analysis

### Long-Term Vision
Use the data acquisition layer to:

1. Build factor exposures such as style, countries, industries, thematic but mostly growth & momentum<br>reference books<br>
   [Quantitative Momentum](https://www.amazon.co.uk/Quantitative-Momentum-Practitioners-Momentum-Based-Selection/dp/111923719X)<br>
   [Factor Investing for Dummies](https://www.amazon.co.uk/Investing-Dummies-Business-Personal-Finance/dp/1119906741)<br>
3. Measure risk of factor exposures
4. Use value factor to identify securities trading below intrinsic value<br>reference books<br>
   [Intelligent Investor](https://en.wikipedia.org/wiki/The_Intelligent_Investor)<br>
   [Quantitative Value](https://www.amazon.co.uk/Quantitative-Value-Practitioners-Intelligent-Eliminating/dp/1118328078)<br>
   [AI investor](https://aiinvestor.gumroad.com/l/BuildYourOwnAIInvestor)<br>
6. Use all the above ingredients to backtest growth at reasonable price (GARP) investment strategy
7. Run real money using this framework

### Data Layers
![Picture1](https://github.com/user-attachments/assets/cdcd1c9a-74f4-4d92-8db6-1c11786da7a2)

## Project Repository

[GitHub Repository](https://github.com/menonf/InvestmentManagement)

## Getting Started

### Clone the Repository

```sh
cd C:\Users\menon\OneDrive\Documents\SourceCode
git clone https://github.com/menonf/InvestmentManagement.git
```

### Setup Virtual Environment

Open the cloned folder in **Visual Studio Code** and run the following commands in the terminal:

```sh
python -m venv .venv
.venv\Scripts\activate.ps1
pip install -r requirements.txt
```

### VS Code Debugging Configuration

If using **VS Code**, use the following settings in launch.json:

```json
{
    "configurations": [
        {
            "name": "Python: Current File (Integrated Terminal)",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "cwd": "${workspaceRoot}",
            "env": {"PYTHONPATH": "${workspaceRoot}"},
            "console": "integratedTerminal",
            "justMyCode": true,
            "purpose": ["debug-in-terminal"]
        }
    ]
}
```

### Navigate & Debug

Run and debug **[Backtesting Demo](https://github.com/menonf/InvestmentManagement/blob/main/toolkit/notebooks/backtest_research_demo.ipynb)**.

### Contributing

Before making changes, create a new working branch:

```sh
git checkout -b data_acquisition_layer
```

Once changes are complete, submit a pull request.

## Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results.
Please consult with a qualified financial advisor before making investment decisions.

---

*Built with ❤️ for the quantitative finance community*

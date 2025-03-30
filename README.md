# Investment Management Platform

## Types of Investors

The investment process starts with identifying the strategic goals and risk tolerance of the investor.

- **Menon** (34 years old) has a 25-year investment horizon and a high-risk tolerance. His optimal allocation should be **100% high-risk public equities**, as market risks tend to even out over long-term investing.
- **Rashid** (56 years old) is saving for his children's education. He has a comfortable income but wants to avoid market turmoil. His optimal allocation should be **60% equities and 40% bonds** to provide a safety net.
- **Wellcome Trust**, a 100-year-old foundation, has a very high-risk tolerance due to its long investment horizon. Its optimal allocation would be **30% private equity, 30% private credit, 30% equities, and 10% cash & bonds**.

## About the Python Project

Risk engines are often a luxury for large asset managers, hedge funds, and banks. This project aims to deliver an **open-source, institutional-grade research, performance, and risk platform** for retail investors like Menon and Rashid, allowing them to extend the platform’s functionality on their own.

While many open-source libraries exist for different aspects of quantitative investment processes (e.g., `pyportfolioopt` for portfolio optimization and `pyfolio` for risk metrics), none integrate them all into an **end-to-end investment process**.

Additionally, most open-source projects do not focus on **storing day-to-day investment analytics**, a function that large financial institutions spend significant resources on. This project not only integrates the investment process but also provides **an easy-to-use and scalable data platform** for retail investors.

## Vision

- **Short-Term Goal**: Reach a stage where the platform is mature enough to employ low-cost overseas developers to extend its functionality and upgrade it from a pet project to an enterprise-grade solution.
- **Grand Vision**: Make the platform **secure, reliable, and scalable** enough to integrate with broker APIs for trading. The first use case is **Menon’s ISA shares portfolio with Hargreaves Lansdown**, which he aims to manage through this platform. If successful, this could evolve into a **family office** managing assets for other investors.


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

If using **VS Code**, use the following settings:

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

Run and debug **`scripts/load_portfolio_analytics.py`**.

### Contributing

Before making changes, create a new working branch:

```sh
git checkout -b data_acquisition_layer
```

Once changes are complete, submit a pull request.

### Data Layers
![Picture1](https://github.com/user-attachments/assets/cdcd1c9a-74f4-4d92-8db6-1c11786da7a2)

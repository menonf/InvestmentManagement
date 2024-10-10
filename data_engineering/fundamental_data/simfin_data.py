"""Module for simfin fundamental data."""

import pandas as pd
import simfin as sf

sf.set_data_dir("~/simfin_data/")
sf.set_api_key(api_key="MFzWmtbTCssYk6YyGO7YSze13qmFUAWd")

income_statement = sf.load(dataset="income", variant="annual", market="us", index="Ticker")
cash_flow_statement = sf.load(dataset="cashflow", variant="annual", market="us", index="Ticker")
balance_sheet_statement = sf.load(dataset="balance", variant="annual", market="us", index="Ticker")


fin_stmt = pd.merge(
    pd.merge(
        income_statement,
        cash_flow_statement,
        on=[
            "Ticker",
            "SimFinId",
            "Currency",
            "Fiscal Year",
            "Report Date",
            "Publish Date",
        ],
    ),
    balance_sheet_statement,
    on=["Ticker", "SimFinId", "Currency", "Fiscal Year", "Report Date", "Publish Date"],
)

print(fin_stmt.loc["MSFT"])
fin_stmt.loc["MSFT"].to_excel("~/simfin_data/fin_stmt.xlsx")

"""Module for financetoolkit fundamental data."""
from financetoolkit import Toolkit

tickers = ["AAPL", "MSFT"]
api_key = "eaa83187bd18f51cf2c84830f00e89ae"


companies = Toolkit(tickers, api_key=api_key, start_date="2016-12-31")

income_statement = companies.get_income_statement()
cash_flow_statement = companies.get_cash_flow_statement()
balance_sheet_statement = companies.get_balance_sheet_statement()

operational_ratios = companies.ratios.collect_efficiency_ratios()
profitability_ratios = companies.ratios.collect_profitability_ratios()
liquidity_ratio = companies.ratios.collect_liquidity_ratios()
solvency_ratios = companies.ratios.collect_solvency_ratios()


print(operational_ratios)

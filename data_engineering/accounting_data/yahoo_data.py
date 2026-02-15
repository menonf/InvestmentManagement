"""Module for yahoo fundamental data."""

import yfinance as yf

msft = yf.Ticker("AAPL")


msft.income_stmt
msft.balance_sheet
msft.cashflow

msft.quarterly_income_stmt
msft.quarterly_balance_sheet
msft.quarterly_cashflow


print(msft.quarterly_income_stmt)

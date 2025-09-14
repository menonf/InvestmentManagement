#!/usr/bin/env python3
"""
Interactive Brokers Module.

This module contains all Interactive Brokers related classes and functions
for connecting to IB Gateway and retrieving portfolio data.
"""


from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import nest_asyncio
import pandas as pd
from ib_insync import IB, Contract, Stock

from data_engineering.database import db_functions as database


class DateUtils:
    """Utility functions for date operations."""

    @staticmethod
    def get_trading_day() -> str:
        """
        Get the most recent trading day (excluding weekends).

        Returns:
            str: Date in 'YYYY-MM-DD' format
        """
        today = datetime.today()
        while today.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            today -= timedelta(days=1)
        return today.strftime("%Y-%m-%d")


class InteractiveBroker:
    """Interactive Brokers portfolio management with integrated connection handling."""

    def __init__(self) -> None:
        """Initialize Interactive Brokers connection handler."""
        self.ib: Optional[IB] = None
        nest_asyncio.apply()

    def connect(self, host: str = "127.0.0.1", port: int = 4001, client_id: int = 1, timeout: int = 10) -> bool:
        """
        Connect to IB Gateway.

        Args:
            host: IB Gateway host
            port: IB Gateway port
            client_id: Client ID for connection
            timeout: Connection timeout in seconds

        Returns:
            bool: Connection success status
        """
        self._cleanup_existing_connection()

        self.ib = IB()
        try:
            print("Attempting to connect to IB Gateway...")
            self.ib.connect(host, port, clientId=client_id, timeout=timeout)
            print(f"Connected to IB Gateway: {self.ib.isConnected()}")

            if self.ib.isConnected():
                print("Connection successful!")
                self._test_connection()
                return True
            else:
                print("Connection failed")
                return False

        except Exception as e:
            print(f"Connection error: {e}")
            print("Make sure IB Gateway is running and API is enabled")
            return False

    def disconnect(self) -> None:
        """Disconnect from IB Gateway."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            print("Disconnected from IB Gateway")

    def is_connected(self) -> bool:
        """Check if connected to IB Gateway."""
        return self.ib is not None and self.ib.isConnected()

    def _cleanup_existing_connection(self) -> None:
        """Clean up any existing IB connections."""
        try:
            if hasattr(self, "ib") and self.ib and self.ib.isConnected():
                print("Existing connection found. Disconnecting...")
                self.ib.disconnect()
                print("Disconnected successfully")
        except Exception as e:
            print(f"Error checking/closing existing connection: {e}")

    def _test_connection(self) -> None:
        """Test the connection with a simple request."""
        if self.ib and self.ib.isConnected():
            try:
                account_summary = self.ib.accountSummary()
                print(f"Retrieved {len(account_summary)} account summary items")
            except Exception as e:
                print(f"Warning: Could not retrieve account summary: {e}")

    def get_positions(self, account_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve portfolio positions for given account.

        Args:
            account_id: IB account ID

        Returns:
            list: Portfolio positions data

        Raises:
            RuntimeError: If not connected to IB Gateway
        """
        if not self.is_connected() or self.ib is None:
            raise RuntimeError("Not connected to IB Gateway")

        try:
            positions = self.ib.positions(account_id)
            portfolio_holdings = []

            for pos in positions:
                contract = pos.contract
                portfolio_holdings.append(
                    {
                        "as_of_date": DateUtils.get_trading_day(),
                        "portfolio_short_name": pos.account,
                        "symbol": contract.symbol,
                        "ib_security_type": contract.secType,
                        "ib_exchange": contract.exchange,
                        "ib_currency": contract.currency,
                        "held_shares": pos.position,
                        "avg_cost": pos.avgCost,
                    }
                )

            return portfolio_holdings

        except Exception as e:
            print(f"Error retrieving positions: {e}")
            raise

    def get_account_summary(self, account_id: Optional[str] = None) -> List[Any]:
        """
        Get account summary information.

        Args:
            account_id: Optional account ID, if None gets all accounts

        Returns:
            list: Account summary data
        """
        if not self.is_connected() or self.ib is None:
            raise RuntimeError("Not connected to IB Gateway")

        try:
            if account_id:
                summary = self.ib.accountSummary(account=account_id)
            else:
                summary = self.ib.accountSummary()
            return summary
        except Exception as e:
            print(f"Error retrieving account summary: {e}")
            raise

    def get_contract_details(
        self, symbol: str, sec_type: str = "STK", exchange: str = "SMART", currency: str = "USD"
    ) -> Optional[Any]:
        """
        Get contract details for a security.

        Args:
            symbol: Security symbol
            sec_type: Security type (STK, OPT, FUT, etc.)
            exchange: Exchange
            currency: Currency

        Returns:
            Contract details or None if not found
        """
        if not self.is_connected() or self.ib is None:
            raise RuntimeError("Not connected to IB Gateway")

        try:
            contract = (
                Stock(symbol, exchange, currency)
                if sec_type == "STK"
                else Contract(symbol=symbol, secType=sec_type, exchange=exchange, currency=currency)
            )

            contract_details = self.ib.reqContractDetails(contract)
            return contract_details[0] if contract_details else None

        except Exception as e:
            print(f"Error getting contract details for {symbol}: {e}")
            return None

    def get_positions_by_account(self, account_ids: Union[str, List[str]]) -> pd.DataFrame:
        """
        Get complete portfolio data for one or multiple accounts.

        Args:
            account_ids: Single IB account ID (str) or list of IB account IDs

        Returns:
            pd.DataFrame: Portfolio holdings data
        """
        if not self.is_connected():
            raise RuntimeError("Must connect to IB Gateway first")

        # Handle both single account and multiple accounts
        if isinstance(account_ids, str):
            account_ids = [account_ids]
            single_account = True
        else:
            single_account = False

        all_positions = []
        successful_accounts = []

        for account_id in account_ids:
            try:
                positions = self.get_positions(account_id)
                all_positions.extend(positions)
                successful_accounts.append(account_id)
            except Exception as e:
                print(f"Error retrieving data for account {account_id}: {e}")

        df = pd.DataFrame(all_positions)

        if not df.empty:
            if single_account:
                print(f"Retrieved {len(df)} positions for account {account_ids[0]}")
                print(df.head())
            else:
                print(f"Retrieved {len(df)} positions across {len(successful_accounts)} accounts")
                print(f"Successful accounts: {successful_accounts}")
                if successful_accounts:
                    print(f"Account breakdown: {df['portfolio_short_name'].value_counts().to_dict()}")
        else:
            if single_account:
                print(f"No positions found for account {account_ids[0]}")
            else:
                print("No positions found for any accounts")

        return df


class IBDataValidator:
    """Validates IB data and handles common data issues."""

    @staticmethod
    def validate_ib_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate IB data DataFrame.

        Args:
            df: IB data DataFrame

        Returns:
            tuple: (is_valid: bool, issues: list)
        """
        issues = []

        # Check required columns
        required_columns = [
            "as_of_date",
            "portfolio_short_name",
            "symbol",
            "ib_security_type",
            "ib_exchange",
            "ib_currency",
            "held_shares",
            "avg_cost",
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")

        # Check for empty data
        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues

        # Check for null values in critical columns
        critical_columns = ["symbol", "held_shares"]
        for col in critical_columns:
            if col in df.columns and df[col].isnull().any():
                null_count = df[col].isnull().sum()
                issues.append(f"Found {null_count} null values in critical column '{col}'")

        # Check for zero positions (might be expected)
        zero_positions = df[df["held_shares"] == 0]
        if not zero_positions.empty:
            issues.append(f"Found {len(zero_positions)} positions with zero shares")

        # Validate data types
        if "held_shares" in df.columns and not pd.api.types.is_numeric_dtype(df["held_shares"]):
            issues.append("'held_shares' column is not numeric")

        if "avg_cost" in df.columns and not pd.api.types.is_numeric_dtype(df["avg_cost"]):
            issues.append("'avg_cost' column is not numeric")

        is_valid = len(issues) == 0
        return is_valid, issues

    @staticmethod
    def clean_ib_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean IB data DataFrame.

        Args:
            df: IB data DataFrame

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if df.empty:
            return df

        df_clean = df.copy()

        # Remove positions with zero shares (optional, depends on requirements)
        # df_clean = df_clean[df_clean['held_shares'] != 0]

        # Clean symbol names (remove extra spaces, convert to uppercase)
        if "symbol" in df_clean.columns:
            df_clean["symbol"] = df_clean["symbol"].str.strip().str.upper()

        # Ensure numeric columns are properly typed
        numeric_columns = ["held_shares", "avg_cost"]
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        print(f"Cleaned portfolio data: {len(df_clean)} records")
        return df_clean


def get_trading_day() -> str:
    """Get current trading day - convenience function."""
    return DateUtils.get_trading_day()


class SecurityMasterManager:
    """Manages security master data operations."""

    # Static mappings for security data standardization
    SECURITY_TYPE_MAPPING = {"STK": "Common Stock", "OPT": "Option", "FUT": "Future", "BOND": "Bond", "CASH": "Cash", "CFD": "CFD"}

    EXCHANGE_MIC_MAPPING = {
        "NASDAQ": "XNAS",
        "NYSE": "XNYS",
        "AMEX": "XASE",
        "SMART": "XNAS",
        "ISLAND": "XNAS",
        "ARCA": "ARCX",
        "BATS": "BATS",
    }

    ASSET_CLASS_MAPPING = {
        "STK": "Equity",
        "OPT": "Derivatives",
        "FUT": "Derivatives",
        "BOND": "Fixed Income",
        "CASH": "Cash",
        "CFD": "Derivatives",
    }

    def __init__(self) -> None:
        """Initialize database connection for security master operations."""
        self.engine, self.connection, self.conn_str, self.session = database.get_db_connection()

    def insert_missing_securities(self, missing_tickers: pd.DataFrame) -> bool:
        """
        Insert missing securities into the SecurityMaster table.

        Args:
            missing_tickers: DataFrame with missing ticker information

        Returns:
            bool: Success status
        """
        if missing_tickers.empty:
            print("No missing tickers found. All securities exist in SecurityMaster table.")
            return True

        print(f"Found {len(missing_tickers)} missing tickers in SecurityMaster table:")
        display_cols = ["symbol", "ib_security_type", "ib_exchange", "ib_currency"]
        available_cols = [col for col in display_cols if col in missing_tickers.columns]
        if available_cols:
            print(missing_tickers[available_cols].drop_duplicates())

        unique_missing = missing_tickers[["symbol", "ib_security_type", "ib_exchange", "ib_currency"]].drop_duplicates()
        new_records = self._create_security_records(unique_missing)

        return self._insert_records(new_records)

    def _create_security_records(self, unique_missing: pd.DataFrame) -> List[Any]:
        """Create new security records for insertion."""
        new_records = []

        for _, row in unique_missing.iterrows():
            new_record = database.SecurityMaster(
                security_id=None,  # Auto-generated
                symbol=row["symbol"],
                isin=None,
                sedol=None,
                cusip=None,
                loanxid=None,
                name=row["symbol"],  # Use symbol as name for now
                country=None,
                currency=row["ib_currency"],
                sector=None,
                industry=None,
                security_type=self.SECURITY_TYPE_MAPPING.get(row["ib_security_type"], "Common Stock"),
                asset_class=self.ASSET_CLASS_MAPPING.get(row["ib_security_type"], "Other"),
                exchange=self.EXCHANGE_MIC_MAPPING.get(row["ib_exchange"], "XNAS"),
                is_active=1,
                source_vendor="IB",
                upsert_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                upsert_by="ib_portfolio_loader",
            )
            new_records.append(new_record)

        return new_records

    def _insert_records(self, new_records: List[Any]) -> bool:
        """Insert new records with error handling."""
        try:
            self.session.add_all(new_records)
            self.session.commit()
            print(f"Successfully inserted {len(new_records)} new records into SecurityMaster.")
            return True

        except Exception as e:
            self.session.rollback()  # type: ignore
            print(f"Error inserting new records: {e}")
            return False

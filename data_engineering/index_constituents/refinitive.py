"""
Refinitiv-based index constituent reconstruction and metric generation.

This module performs the following:

1. Reconstructs historical index membership using joiner/leaver events.
2. Enriches constituents with internal security master identifiers.
3. Builds float-adjusted shares outstanding metrics.

All functionality remains equivalent to the original implementation.
"""

import lseg.data as ld
import pandas as pd

from data_engineering.database import database as database

ld.open_session()


class IndexConstituents:
    """Service for reconstructing historical index membership using Refinitiv index constituent data."""

    def get_historical_constituents(
        self,
        index: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Reconstruct historical index membership over a date range.

        Parameters
        ----------
        index : str
            Index RIC (e.g. ".NDX").
        start : str
            Start date in ISO format (YYYY-MM-DD).
        end : str
            End date in ISO format (YYYY-MM-DD).

        Returns
        -------
        pd.DataFrame
            Interval-based index membership containing:
            - Constituent RIC
            - Exchange Ticker
            - Start Date
            - End Date
        """
        initial = self.get_constituents_as_of(index, start)
        changes = self.get_constituent_changes(index, start, end)
        return self.update_constituents(start, initial, changes)

    def get_constituents_as_of(
        self,
        ric: str,
        date: str,
    ) -> pd.DataFrame:
        """
        Retrieve index constituents active on a specific date.

        Parameters
        ----------
        ric : str
            Index RIC.
        date : str
            Date in ISO format (YYYY-MM-DD).

        Returns
        -------
        pd.DataFrame
            Constituents active on the given date with initial start and end placeholders.

        """
        universe = [f"0#{ric}({date.replace('-', '')})"]

        df = ld.get_data(
            universe=universe,
            fields=["TR.PriceClose", "TR.ExchangeTicker"],
            parameters={"SDATE": date, "EDATE": date},
        )

        df = df.rename(columns={"Instrument": "Constituent RIC"})
        df["Start Date"] = date
        df["End Date"] = None

        return df[["Constituent RIC", "Exchange Ticker", "Start Date", "End Date"]]

    def get_constituent_changes(
        self,
        ric: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Retrieve joiner and leaver events for an index.

        Parameters
        ----------
        ric : str
            Index RIC.
        start : str
            Start date (YYYY-MM-DD).
        end : str
            End date (YYYY-MM-DD).

        Returns
        -------
        pd.DataFrame
            Constituent change events including:
            - Constituent RIC
            - Exchange Ticker
            - Date
            - Change type

        """
        const_changes = ld.get_data(
            universe=[ric],
            fields=[
                "TR.IndexJLConstituentChangeDate",
                "TR.IndexJLConstituentRIC",
                "TR.IndexJLConstituentName",
                "TR.IndexJLConstituentituentChange",
            ],
            parameters={"SDATE": start, "EDATE": end, "IC": "B"},
        )

        tickers = ld.get_data(
            universe=const_changes["Constituent RIC"].unique(),
            fields=["TR.TickerSymbol", "TR.RIC"],
        )

        const_changes = const_changes.merge(
            tickers,
            left_on="Constituent RIC",
            right_on="RIC",
            how="left",
        )

        const_changes = const_changes.rename(columns={"Ticker Symbol": "Exchange Ticker"})

        return const_changes

    def update_constituents(
        self,
        start: str,
        constituents: pd.DataFrame,
        constituent_changes: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construct continuous membership intervals from seed constituents and join/leave events.

        Parameters
        ----------
        start : str
            Reconstruction window start date.
        constituents : pd.DataFrame
            Constituents active at the start date.
        constituent_changes : pd.DataFrame
            Chronologically ordered join and leave events.

        Returns
        -------
        pd.DataFrame
            Interval-based membership history withadjusted start and end dates.

        """
        constituent_changes = constituent_changes.copy()
        constituent_changes["Date"] = pd.to_datetime(constituent_changes["Date"])

        start_dt = pd.to_datetime(start)

        df = pd.DataFrame(
            {
                "Constituent RIC": constituents["Constituent RIC"],
                "Exchange Ticker": constituents["Exchange Ticker"],
                "Start Date": start_dt,
                "End Date": pd.NaT,
            }
        )

        for _, change in constituent_changes.sort_values("Date").iterrows():
            ric = change["Constituent RIC"]
            change_date = change["Date"]
            change_type = change["Change"]
            ticker = change.get("Exchange Ticker", "")

            if change_type == "Joiner":
                seed_mask = (df["Constituent RIC"] == ric) & (df["Start Date"] == start_dt) & (df["End Date"].isna())

                if seed_mask.any():
                    idx = df[seed_mask].index[0]
                    df.loc[idx, "Start Date"] = change_date
                    df.loc[idx, "Exchange Ticker"] = ticker
                else:
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "Constituent RIC": [ric],
                                    "Exchange Ticker": [ticker],
                                    "Start Date": [change_date],
                                    "End Date": [pd.NaT],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

            elif change_type == "Leaver":
                mask = (df["Constituent RIC"] == ric) & (df["End Date"].isna()) & (df["Start Date"] <= change_date)

                if mask.any():
                    idx = df[mask].sort_values("Start Date").index[-1]
                    df.loc[idx, "End Date"] = change_date
                else:
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "Constituent RIC": [ric],
                                    "Exchange Ticker": [ticker],
                                    "Start Date": [start_dt],
                                    "End Date": [change_date],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

        return df


def build_index_constituents(index: str, start: str, end: str) -> pd.DataFrame:
    """
    High-level wrapper to reconstruct index membership.

    Parameters
    ----------
    index : str
        Index RIC.
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Historical index membership intervals.

    """
    ic = IndexConstituents()
    return ic.get_historical_constituents(index=index, start=start, end=end)


def enrich_with_security_master(df: pd.DataFrame) -> pd.DataFrame:
    """Map index constituents to internal security identifiers."""
    engine, connection, conn_str, session = database.get_db_connection()
    security_master = database.read_security_master(session, engine)

    df = df.merge(
        security_master,
        left_on="Exchange Ticker",
        right_on="symbol",
        how="left",
    )

    df["security_id"] = df["security_id"].astype("Int64")
    df["index_id"] = 1
    df["source_vendor"] = "refinitive"
    df["upsert_date"] = pd.Timestamp.now().floor("s")
    df["upsert_by"] = "data_engineering.index_constituents.refinitive.py"

    df = df.rename(
        columns={
            "Exchange Ticker": "exchange_ticker",
            "Start Date": "start_date",
            "End Date": "end_date",
        }
    )

    df["end_date"] = df["end_date"].replace({pd.NaT: None})

    return df[
        [
            "index_id",
            "security_id",
            "exchange_ticker",
            "start_date",
            "end_date",
            "source_vendor",
            "upsert_date",
            "upsert_by",
        ]
    ]


def build_float_adjusted_shares(df_constituents: pd.DataFrame) -> pd.DataFrame:
    """Build float-adjusted shares outstanding metric used for reconstructing historical index weights."""
    universe = df_constituents["Constituent RIC"].drop_duplicates().tolist()

    shares = ld.get_data(
        universe=universe,
        fields=["TR.SharesOutstanding", "TR.SharesOutstanding.Date"],
        parameters={"SDate": "2025-01-01", "EDate": "2025-12-31", "Frq": "D"},
    )

    free_float = ld.get_data(
        universe=universe,
        fields=["TR.FreeFloatPct", "TR.FreeFloatPct.Date"],
        parameters={"SDate": "2025-01-01", "EDate": "2026-01-05", "Frq": "M"},
    )

    free_float = (
        free_float.dropna(subset=["Free Float (Percent)"])
        .drop_duplicates(subset=["Instrument", "Date"], keep="last")
        .assign(Date=lambda x: pd.to_datetime(x["Date"]))
        .sort_values(["Instrument", "Date"])
    )

    free_float = free_float.set_index("Date").groupby("Instrument")["Free Float (Percent)"].resample("D").ffill().reset_index()

    merged = shares.merge(free_float, how="left")

    joined = (
        df_constituents[["Constituent RIC", "exchange_ticker", "security_id"]]
        .drop_duplicates()
        .merge(
            merged,
            left_on="Constituent RIC",
            right_on="Instrument",
            how="right",
        )
    )

    joined = joined.ffill()

    joined["Shares Outstanding"] = joined["Outstanding Shares"] * (joined["Free Float (Percent)"].round(0).clip(0, 100) / 100)

    metrics = joined.rename(
        columns={
            "Shares Outstanding": "metric_value",
            "Date": "effective_date",
        }
    )

    metrics["metric_type"] = "Shares Outstanding"
    metrics["source_vendor"] = "refinitive"
    metrics["end_date"] = None

    return metrics[
        [
            "security_id",
            "metric_type",
            "metric_value",
            "source_vendor",
            "effective_date",
            "end_date",
        ]
    ]


if __name__ == "__main__":
    nasdaq100 = build_index_constituents(
        index=".NDX",
        start="2025-01-01",
        end="2025-12-31",
    )

    df_to_write = enrich_with_security_master(nasdaq100)

    metrics_df = build_float_adjusted_shares(df_to_write)

    print(metrics_df.head())

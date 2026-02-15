import os
import sys

from brokers.interactive_broker import *
from data_engineering.database import database as database


def main():
    """Main execution function"""
    print(f"=== Portfolio Data Loader Started ===")
    print(f"Current trading day: {DateUtils.get_trading_day()}")
    
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
        
    print(f"Current directory: {current_dir}")
    print(f"Added to path: {parent_dir}")
    
    ib = None
    
    try:
        print("\n=== Connecting to IB Gateway ===")
        ib = InteractiveBroker()
        sm_manager = SecurityMasterManager()
        
        # Connect to IB Gateway
        if not ib.connect():
            print("Failed to connect to IB Gateway. Exiting.")
            return
        
        print("\n=== Retrieving Interactive Broker Data ===")
        ib_account_positions = ib.get_positions_by_account('U20761295')
        
        if ib_account_positions.empty:
            print("No portfolio data retrieved. Exiting.")
            return
            
        # Validate the data
        is_valid, issues = IBDataValidator.validate_ib_data(ib_account_positions)
        if not is_valid:
            print("Data validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
            print("Proceeding with data cleaning...")
            
        # Clean the data
        ib_account_positions = IBDataValidator.clean_ib_data(ib_account_positions)

        engine, connection, session = database.get_db_connection()
        security_master = database.read_security_master(session, engine)

        merged_securities = pd.merge(ib_account_positions,
            security_master, 
            on='symbol', 
            how='left'
        )

        missing_tickers = merged_securities[merged_securities['security_id'].isna()].copy()

        # Insert missing securities if any
        if sm_manager.insert_missing_securities(missing_tickers):
            if not missing_tickers.empty:
                # Re-read security master data to include newly inserted records
                security_master = database.read_security_master(session, engine)
                
                # Re-merge the data with updated securities table
                merged_securities = pd.merge(ib_account_positions,
                    security_master, 
                    on='symbol', 
                    how='left'
                )
                print("Re-merged portfolio data with updated SecurityMaster.")

        # Final verification
        remaining_missing = merged_securities[merged_securities['security_id'].isna()]
        if not remaining_missing.empty:
            print(f"Warning: {len(remaining_missing)} records still have missing security data:")
            print(remaining_missing[['symbol', 'security_type', 'exchange']])
        else:
            print("All positions have corresponding SecurityMaster records.")
            
        
        df_portfolio_data = database.read_portfolio(
            session, 
            engine, 
            merged_securities['portfolio_short_name'].unique().tolist()
        )
            
        df_portfolio_market_data = pd.merge(df_portfolio_data, merged_securities)
            
        # Select final columns for database storage
        df_portfolio_market_data = df_portfolio_market_data[[
            'as_of_date',
            'port_id', 
            'security_id', 
            'held_shares'
        ]]
        
        database.write_portfolio_holdings(df_portfolio_market_data, session)
        print("Portfolio holdings successfully written to database")
        
    except Exception as e:
        print(f"\n=== Error in main execution ===")
        print(f"Error: {e}")
        raise
        
    finally:
        # Cleanup connections
        print("\n=== Cleaning up connections ===")
        try:
            if ib and ib.is_connected():
                ib.disconnect()
                print("IB Gateway connection closed successfully")
            else:
                print("No active IB connection to close")
        except Exception as e:
            print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()
"""Data module for pulling S&P 500 returns from WRDS CRSP database."""

import wrds
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime


class CRSPDataLoader:
    """Loads daily S&P 500 stock returns from WRDS CRSP database."""

    def __init__(self, username: Optional[str] = None):
        """Initialize WRDS connection.
        
        Args:
            username: WRDS username. If None, will use environment credentials.
        """
        self.username = username
        self.db = None

    def connect(self) -> None:
        """Establish connection to WRDS database."""
        if self.username:
            self.db = wrds.Connection(wrds_username=self.username)
        else:
            self.db = wrds.Connection()

    def disconnect(self) -> None:
        """Close WRDS database connection."""
        if self.db:
            self.db.close()

    def get_sp500_constituents(self, date: Optional[str] = None) -> pd.DataFrame:
        """Get S&P 500 constituent tickers.
        
        Args:
            date: Date in 'YYYY-MM-DD' format. If None, uses most recent.
            
        Returns:
            DataFrame with S&P 500 constituent information.
        """
        # Query S&P 500 constituents from CRSP index membership
        query = """
            SELECT DISTINCT permno, ticker
            FROM crsp.dsp500list
            WHERE ending >= '{date}'
            ORDER BY permno
        """
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        return self.db.raw_sql(query.format(date=date))

    def get_daily_returns(
        self, 
        start_date: str, 
        end_date: str,
        permnos: Optional[list] = None
    ) -> pd.DataFrame:
        """Pull daily returns for S&P 500 stocks from CRSP.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            permnos: List of PERMNOs to query. If None, queries S&P 500.
            
        Returns:
            DataFrame with columns: date, permno, ticker, ret (daily return).
        """
        if not self.db:
            self.connect()

        # Build SQL query for daily stock returns
        query = """
            SELECT 
                a.date,
                a.permno,
                b.ticker,
                a.ret
            FROM 
                crsp.dsf as a
            LEFT JOIN 
                crsp.dsenames as b
            ON 
                a.permno = b.permno
                AND b.namedt <= a.date
                AND a.date <= b.nameendt
            WHERE 
                a.date BETWEEN '{start_date}' AND '{end_date}'
                AND a.ret IS NOT NULL
        """

        # Add PERMNO filter if provided
        if permnos:
            permno_list = ','.join(map(str, permnos))
            query += f" AND a.permno IN ({permno_list})"
        else:
            # Filter for S&P 500 constituents
            query += """
                AND a.permno IN (
                    SELECT DISTINCT permno 
                    FROM crsp.dsp500list
                    WHERE ending >= '{start_date}'
                )
            """

        query += " ORDER BY a.date, a.permno"

        df = self.db.raw_sql(query.format(
            start_date=start_date,
            end_date=end_date
        ))

        return df

    def get_returns_pivot(
        self,
        start_date: str,
        end_date: str,
        permnos: Optional[list] = None
    ) -> pd.DataFrame:
        """Get daily returns in pivot format (dates x tickers).
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            permnos: List of PERMNOs to query. If None, queries S&P 500.
            
        Returns:
            DataFrame with dates as index and tickers as columns.
        """
        df = self.get_daily_returns(start_date, end_date, permnos)
        
        # Pivot to wide format
        pivot_df = df.pivot(index='date', columns='ticker', values='ret')
        
        return pivot_df

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

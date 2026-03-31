import logging
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_data(ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetches historical OHLCV data for a given ticker from Yahoo Finance.
    
    Args:
        ticker (str): The ticker symbol (e.g., 'SPY', 'KC=F').
        start_date (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to 5 years ago.
        end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to today.
        
    Returns:
        pd.DataFrame: Cleaned OHLCV DataFrame.
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - relativedelta(years=5)).strftime('%Y-%m-%d')

    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
    
    # Primary fetch attempt
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Fallback logic for KC=F
    if ticker == 'KC=F' and len(df) < 100:
        logger.warning(f"KC=F returned fewer than 100 rows ({len(df)}). Falling back to KT=F (Coffee mini).")
        ticker = 'KT=F'
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
    if df.empty:
        logger.warning(f"No data returned for {ticker}.")
        return df

    # MultiIndex handling for yfinance 0.2.x+ (it sometimes returns multi-index columns if downloading single ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel('Ticker') if 'Ticker' in df.columns.names else df.columns.get_level_values(0)

    # Forward-fill NaNs and then drop any remaining
    df = df.ffill().dropna()

    # Save to CSV
    # Extract just date parts for filename safety
    safe_start = start_date.replace('-', '')
    safe_end = end_date.replace('-', '')
    
    # Ensure raw data directory exists
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    # Use ticker in filename, replacing invalid char for files if necessary
    safe_ticker = ticker.replace('=', '_')
    csv_path = os.path.join(raw_dir, f"{safe_ticker}_{safe_start}_{safe_end}.csv")
    
    df.to_csv(csv_path)
    logger.info(f"Saved raw data to {csv_path}")
    
    return df

def fetch_etc_price(ticker="COFF.MI"):
    """Fetch current COFF ETC price in EUR from Borsa Italiana."""
    try:
        data = yf.Ticker(ticker).history(period="5d")
        if len(data) > 0:
            return float(data['Close'].iloc[-1])
    except Exception:
        pass
    return None

import hydra
from omegaconf import DictConfig
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import pathlib
import logging
import warnings

# Suppress specific pandas warnnings that cluftter logs
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
log = logging.getLogger(__name__)

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance."""
    log.info(f"Fetching data for {ticker} from {start} to {end}")
    
    # auto adjust=True fixes OHLC for splits/dividends automatically
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, multi_level_index=False)
    
    if data.empty:
        raise ValueError(f"No data fetched for ticker {ticker}. Check internet connection or dates.")
    
    # Esnure index is datetime
    data.index = pd.to_datetime(data.index)
    
    return data

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical indicators using Pandas TA."""
    df = df.copy()
    
    # RSI (Relative Strength Index) - Momentum Indicator
    rsi = df.ta.rsi(close='Close', length=14)
    df = pd.concat([df, rsi], axis=1)
    # 'length' is the standard 14-day period for RSI
    
    # MACD (Moving Average Convergence Divergence) - Trend-Following Momentum Indicator
    # Returns 3 columns: MACD, Histogream and Signal
    macd = df.ta.macd(close='close', fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    
    # Bellinger Bands - Volatility Indicator
    bbdands = df.ta.bbands(close='Close', length=20, std=2)
    df = pd.concat([df, bbdands], axis=1)
    
    # ATR (Average True Range) - Volatility Indicator
    df['ATR'] = df.ta.atr(close='Close', length=14)
    
    # Log Returns (The Target)
    # Log returns are preferred in RL over % returns because they are additive/
    df['Log_Returrn'] = ta.log_return(df['Close'])
    
    # Clean up
    # Indicatire inroduce NaNs (e.g. you can't have RSI for the first 14 days)
    # We drop these rows to avoid breaking the RL model training
    original_length = len(df)
    df.dropna(inplace=True)
    log.info(f"Dropped {original_length - len(df)} rows due to NaN values (warm-up period.")
    
    return df

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main ETL Pipeline Controlled by Hydra Configs.
    """
    # Resolve Paths
    # Hydra changes the working directory, so we use 'get_original_cwd' to find the absolute root
    root_path = pathlib.Path(hydra.utils.get_original_cwd())
    output_dir = root_path / cfg.env.data_dir
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract
    try:
        raw_data = fetch_data(cfg.env.ticker, cfg.env.start_date, cfg.env.end_date)
    except Exception as e:
        log.error(f"Error fetching data: {e}")
        return

    # Transform
    processed_df = process_data(raw_data)
    
    # Load
    # Parquet is preferred for its efficiency and compatibility with data processing frameworks
    filename = f"{cfg.env.ticker}_processed.parqiuet"
    save_path = output_dir / filename
    
    processed_df.to_parquet(save_path)
    
    log.info(f"Processed data saved to {save_path}")
    log.info(f"FInal Shape: {processed_df.shape}")
    log.info(f"Columns: {processed_df.columns.tolist()}")
    
if __name__ == "__main__":
    main()
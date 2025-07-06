import yfinance as yf
import pandas as pd

def fetch_yahoo_data(ticker, interval):

    interval_map = {
        '15min': '15m',
        '1h': '60m',
        '1d': '1d'
    }
    
    data = yf.download(
        tickers=ticker,
        period="max", 
        interval=interval_map[interval],
        progress=False
    )
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    return data
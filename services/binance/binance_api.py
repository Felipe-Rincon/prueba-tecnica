import pandas as pd
import ccxt
from datetime import datetime, timedelta

def fetch_binance_data(symbol='BTC/USDT', timeframe='1h', limit=10000):
    exchange = ccxt.binance()
    all_ohlcv = []
    
    since_date = datetime.now() - timedelta(days=limit/24)
    since = exchange.parse8601(since_date.strftime('%Y-%m-%d %H:%M:%S'))
    
    try:
        while len(all_ohlcv) < limit:
            ohlcv = exchange.fetch_ohlcv(
                symbol, 
                timeframe, 
                since=since,
                limit=1000 
            )
            if not ohlcv:
                break
            all_ohlcv = ohlcv + all_ohlcv
            since = ohlcv[0][0] - 3600000

        df = pd.DataFrame(all_ohlcv[-limit:], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error downloading data {e}")
        return None
    
    
def get_crypto_data(symbol='BTC/USDT', timeframe='1d', limit=2000):
    exchange = ccxt.binance()

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df
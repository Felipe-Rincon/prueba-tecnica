from alpha_vantage.timeseries import  TimeSeries

def fetch_alpha_vantage_data(ticker, interval):
    
    ALPHA_VANTAGE_API_KEY = '1KUM54ZIWNJLIBXR' # 51QD868E5F90V2NF
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    
    interval_map = {
        '15min': '15min',
        '1h': '60min',
        '1d': 'daily'
    }
    
    data, _ = ts.get_intraday(ticker, interval=interval_map[interval], outputsize='full') \
        if interval != '1d' else ts.get_daily(ticker, outputsize='full')
    
    data = data.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    
    return data
    
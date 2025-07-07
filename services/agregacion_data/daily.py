import requests
import json
from datetime import datetime, timedelta
from collections import defaultdict

BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/24hr"
OUTPUT_FILE = "data_binance_daily.json"

def obtain_data_binance():
    try:
        response = requests.get(BINANCE_API_URL)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error obtaining data: {e}")
        return None

def add_data(data):
    aggregations = {
        'fecha': datetime.now().strftime('%Y-%m-%d'),
        'symbols': {},
        'resumen': {
            'total_volume': 0,
            'symbol_count': 0,
            'avg_price_all': 0,
            'top_5_volume': []
        }
    }
    
    if not data:
        return aggregations
    
    price_sum_all = 0
    symbol_count = 0
    
    for item in data:
        symbol = item['symbol']
        volume = float(item['volume'])
        price = float(item['lastPrice'])
        
        # aggregations por símbolo
        aggregations['symbols'][symbol] = {
            'volume': volume,
            'price': price,
            'price_change': float(item['priceChangePercent']),
            'high': float(item['highPrice']),
            'low': float(item['lowPrice'])
        }
        
        # Para aggregations globales
        aggregations['resumen']['total_volume'] += volume
        price_sum_all += price
        symbol_count += 1
    
    # Calcular promedios globales
    if symbol_count > 0:
        aggregations['resumen']['avg_price_all'] = price_sum_all / symbol_count
        aggregations['resumen']['symbol_count'] = symbol_count
    
    # Top 5 por volumen
    top_symbols = sorted(aggregations['symbols'].items(), key=lambda x: x[1]['volume'], reverse=True)[:5]
    aggregations['resumen']['top_5_volume'] = [{
        'symbol': symbol,
        'volume': data['volume']
    } for symbol, data in top_symbols]
    
    return aggregations

def guardar_data(data):
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    print(f"Running daily aggregation - {datetime.now()}")
    data = obtain_data_binance()
    if data:
        data_agregados = add_data(data)
        guardar_data(data_agregados)
        print("Daily aggregation completed and saved")

if __name__ == "__main__":
    main()
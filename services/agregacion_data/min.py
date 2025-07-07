import requests
import json
from datetime import datetime
import time
from collections import defaultdict

BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/24hr"
OUTPUT_FILE = "datos_binance_min.json"
AGGREGATION_INTERVAL = 60 

def obtain_data_binance():
    try:
        response = requests.get(BINANCE_API_URL)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error obtaining data: {e}")
        return None

def add_datos(datos,    new_data):
    if not datos:
        datos = {
            'fecha': datetime.now().strftime('%Y-%m-%d'),
            'symbols': defaultdict(lambda: {
                'volume_sum': 0,
                'price_avg': 0,
                'count': 0,
                'high_max': -float('inf'),
                'low_min': float('inf')
            })
        }
    
    for item in new_data:
        symbol = item['symbol']
        volume = float(item['volume'])
        price = float(item['lastPrice'])
        high = float(item['highPrice'])
        low = float(item['lowPrice'])
        
        datos['symbols'][symbol]['volume_sum'] += volume
        datos['symbols'][symbol]['price_avg'] = (datos['symbols'][symbol]['price_avg'] * datos['symbols'][symbol]['count'] + price) / (datos['symbols'][symbol]['count'] + 1)
        datos['symbols'][symbol]['count'] += 1
        datos['symbols'][symbol]['high_max'] = max(datos['symbols'][symbol]['high_max'], high)
        datos['symbols'][symbol]['low_min'] = min(datos['symbols'][symbol]['low_min'], low)
    
    return datos

def save_data(datos):
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(datos, f, indent=2)

def main():
    aggregated_data = None
    
    while True:
        actual = datetime.now()
        print(f"Obteniendo datos a las {actual}")
        
        new_data = obtain_data_binance()
        if  new_data:
            aggregated_data = add_datos(aggregated_data,    new_data)
            save_data(aggregated_data)
            print("Datos actualizados y guardados")
        
        time_waiting = AGGREGATION_INTERVAL - (datetime.now().timestamp() % AGGREGATION_INTERVAL)
        time.sleep(time_waiting)

if __name__ == "__main__":
    main()
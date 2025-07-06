import numpy as np 

def calculate_rsi(prices, window=14):
    deltas = np.diff(prices)
    seed = deltas[:window + 1]
    
    average_gain = np.mean(seed[seed >= 0])
    average_loss = -np.mean(seed[seed < 0])
    
    rsi_values = np.zeros_like(prices)
    rsi_values[:window] = 100. - (100. / (1. + average_gain / average_loss if average_loss != 0 else 1.))
    
    for i in range(window, len(prices)):
        delta = deltas[i - 1]
        
        if delta > 0:
            gain = delta
            loss = 0.
        else:
            gain = 0.
            loss = -delta
        
        average_gain = (average_gain * (window - 1) + gain) / window
        average_loss = (average_loss * (window - 1) + loss) / window
        
        rs = average_gain / average_loss if average_loss != 0 else 1.
        rsi_values[i] = 100. - (100. / (1. + rs))
    
    return rsi_values

def calculate_ema(prices, window):
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    alpha = 2. / (window + 1.)
    
    for i in range(1, len(prices)):
        ema[i] = prices[i] * alpha + ema[i - 1] * (1 - alpha)
    return ema

def calculate_macd(prices, fast=10, slow=50, signal=9):
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_ichimoku(high, low, close, tenkan_window=9, kijun_window=26, senkou_b_window=52, offset=26):

    tenkan_high = np.array([max(high[i-tenkan_window:i]) for i in range(tenkan_window, len(high))])
    tenkan_low = np.array([min(low[i-tenkan_window:i]) for i in range(tenkan_window, len(low))])
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    kijun_high = np.array([max(high[i-kijun_window:i]) for i in range(kijun_window, len(high))])
    kijun_low = np.array([min(low[i-kijun_window:i]) for i in range(kijun_window, len(low))])
    kijun_sen = (kijun_high + kijun_low) / 2
    
    min_len = min(len(tenkan_sen), len(kijun_sen))
    tenkan_for_senkou = tenkan_sen[-min_len:]
    kijun_for_senkou = kijun_sen[-min_len:]
    senkou_a = (tenkan_for_senkou + kijun_for_senkou) / 2
    
    senkou_b_high = np.array([max(high[i-senkou_b_window:i]) for i in range(senkou_b_window, len(high))])
    senkou_b_low = np.array([min(low[i-senkou_b_window:i]) for i in range(senkou_b_window, len(low))])
    senkou_b = (senkou_b_high + senkou_b_low) / 2
    
    senkou_b = senkou_b[-len(senkou_a):]
    
    chikou_span = close[offset:offset+len(close)]
    print(tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou_span)
    return tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou_span

def calculate_atr(high, low, close, window=14):
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.zeros(len(tr))
    atr[window - 1] = np.mean(tr[:window])
    
    for i in range(window, len(tr)):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
    
    return atr
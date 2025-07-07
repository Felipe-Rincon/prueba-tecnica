import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import TimeSeriesSplit
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from sklearn.metrics import mean_absolute_error

from services.binance.binance_api import get_crypto_data

def create_features(df):

    df['log_ret'] = np.log(df['close']/df['close'].shift(1))
    df['vol_pct'] = df['volume'].pct_change()

    df['rsi_14'] = RSIIndicator(df['close'], window=14).rsi()
    df['atr_14'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['daily_range'] = (df['high'] - df['low'])/df['open']

    for lag in [1, 2, 3, 5, 7, 14, 21]:
        df[f'ret_lag_{lag}'] = df['log_ret'].shift(lag)
    
    for window in [7, 14, 21]:
        df[f'sma_{window}'] = df['close'].rolling(window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
    
    df['close_to_sma7'] = df['close']/df['sma_7']
    df['close_to_ema14'] = df['close']/df['ema_14']
    
    return df.dropna()

def prepare_data(df, target_col='log_ret', test_size=0.2):
    features = df.drop(columns=[target_col, 'open', 'high', 'low', 'close', 'volume'])
    target = df[target_col]
    
    split_idx = int(len(df) * (1-test_size))
    X_train = features.iloc[:split_idx]
    y_train = target.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_test = target.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test

def train_lgbm(X_train, y_train):
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        objective='huber',
        random_state=42,
        n_jobs=-1
    )
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=True) 
            ]
        )
    
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    return preds, mae

def predict_btc(model, feature_names):

    new_data = get_crypto_data(limit=100)
    features = create_features(new_data)
    
    last_obs = features[feature_names].iloc[-1:].copy()
    
    pred_log_ret = model.predict(last_obs)[0]
    pred_close = new_data['close'].iloc[-1] * np.exp(pred_log_ret)
    
    print(f"\nPredicción para próximo cierre:")
    print(f"Precio actual: ${new_data['close'].iloc[-1]:.2f}")
    print(f"Precio predicho: ${pred_close:.2f}")
    print(f"Retorno esperado: {pred_log_ret:.2%}")
    
    return new_data, pred_close, pred_log_ret
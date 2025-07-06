import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report


def calcular_features(df):

    df['return_1h'] = df['close'].pct_change()
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    df['rsi'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
    
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    
    df['volatility_5h'] = df['return_1h'].rolling(5).std()
    df['volumen_relative'] = df['volume'] / df['volume'].rolling(24).mean()
    
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    return df.dropna()

class backtester_and_predict_next:
    def __init__(self, data, train_size=480, test_size=12):
        self.data = data
        self.train_size = train_size
        self.test_size = test_size
        self.results = []
        self.features = ['rsi', 'ema_50', 'macd', 'volatility_5h', 'volumen_relative']
        self.model = None
    
    def run(self):
        
        for i in range(0, len(self.data) - self.train_size - self.test_size, self.test_size):
            train = self.data.iloc[i:i + self.train_size]
            test = self.data.iloc[i + self.train_size:i + self.train_size + self.test_size]
            
            X_train, y_train = train[self.features], train['target']
            self.model = LGBMClassifier(class_weight='balanced', n_estimators=100)
            self.model.fit(X_train, y_train)
            
            X_test, y_test = test[self.features], test['target']
            preds = self.model.predict(X_test)
            
            accuracy = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds, output_dict=True)
            
            test = test.copy()
            test.loc[:, 'pred'] = preds
            test.loc[:, 'retorno_strategy'] = test['return_1h'].shift(-1) * test['pred']
            retorno = (1 + test['retorno_strategy'].dropna()).cumprod().iloc[-1] - 1
            
            self.results.append({
                'start': test.index[0],
                'end': test.index[-1],
                'accuracy': accuracy,
                'f1': report['weighted avg']['f1-score'],
                'return': retorno
            })
        
        return pd.DataFrame(self.results)

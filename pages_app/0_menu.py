import streamlit as st

st.title("Menu")



def show_full_technical_doc():
    st.title("Documentación Técnica: Modelo Sube o Baja en BTC/USDT")
    
    st.markdown("""
    ## 1. Introducción

    Se describe un sistema de trading algorítmico diseñado para predecir movimientos de precios en el par BTC/USDT en Binance, utilizando un enfoque de machine learning con características técnicas y backtesting recursivo.
    """)

    st.markdown("""
    ## 2. Arquitectura del Sistema

    El sistema consta de tres componentes principales:

    1. **Obtención de datos**: `fetch_binance_data` - Recupera datos históricos de OHLCV de Binance
    2. **Procesamiento y feature engineering**: `calcular_features` - Calcula indicadores técnicos
    3. **Modelado y backtesting**: `backtester_and_predict_next` - Implementa el modelo y estrategia
    """)

    st.markdown("""
    ## 3. Obtención de Datos (`fetch_binance_data`)

    ### 3.1 Método
    - Utiliza la API de CCXT para conectarse a Binance
    - Recupera datos OHLCV (Open-High-Low-Close-Volume) en intervalos de 1 hora
    - Implementa paginación para obtener grandes volúmenes de datos históricos

    ### 3.2 Parámetros
    - **symbol**: Par de trading (BTC/USDT por defecto)
    - **timeframe**: Intervalo temporal ('1h' para datos horarios)
    - **limit**: Número máximo de velas a recuperar (10,000 por defecto ≈ 416 días)

    ### 3.3 Consideraciones Técnicas
    - Los timestamps se convierten a datetime y se usan como índice
    - El límite de 1,000 velas por request es una limitación de la API Binance
    - El manejo de fechas utiliza UTC para evitar problemas de zonas horarias
    """)

    st.markdown("""
    ## 4. Feature Engineering (`calcular_features`)

    ### 4.1 Indicadores Calculados

    #### 4.1.1 Retorno horario (`return_1h`)
    ```python
    df['return_1h'] = df['close'].pct_change()
    ```
    - Mide el cambio porcentual en el precio de cierre cada hora
    - Fundamental para calcular volatilidad y definir el target

    #### 4.1.2 Índice de Fuerza Relativa (RSI 14)
    ```python
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    df['rsi'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
    ```
    - Oscilador de momentum que mide la velocidad y cambio de movimientos de precios
    - Rango: 0-100 (sobrecompra >70, sobreventa <30)
    - Ventana de 14 periodos (14 horas) es estándar en análisis técnico

    #### 4.1.3 Media Móvil Exponencial (EMA 50)
    ```python
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    ```
    - Media móvil que da más peso a los precios recientes
    - Span de 50 periodos (≈ 2 días) como referencia de tendencia media

    #### 4.1.4 MACD
    ```python
    df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    ```
    - Diferencia entre EMA(12) y EMA(26)
    - Indicador de tendencia que muestra relación entre dos medias móviles

    #### 4.1.5 Volatilidad (5 horas)
    ```python
    df['volatility_5h'] = df['return_1h'].rolling(5).std()
    ```
    - Desviación estándar de los retornos en ventana de 5 horas
    - Mide la dispersión de los retornos (riesgo)

    #### 4.1.6 Volumen Relativo
    ```python
    df['volumen_relative'] = df['volume'] / df['volume'].rolling(24).mean()
    ```
    - Ratio entre volumen actual y media móvil de 24 horas (1 día)
    - Indica actividad anormal en el mercado

    ### 4.2 Variable Target
    ```python
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    ```
    - Variable binaria: 1 si el precio sube en la siguiente hora, 0 si baja
    - Se usa shift(-1) para predecir el movimiento futuro
    """)

    st.markdown("""
    ## 5. Modelado y Backtesting (`backtester_and_predict_next`)

    ### 5.1 Configuración del Modelo

    #### 5.1.1 LightGBM Classifier
    ```python
    self.model = LGBMClassifier(class_weight='balanced', n_estimators=100)
    ```
    - **class_weight='balanced'**: Autoajusta pesos para manejar desbalance de clases
    - **n_estimators=100**: Número de árboles 
    - **Características usadas**: ['rsi', 'ema_50', 'macd', 'volatility_5h', 'volumen_relative']

    #### 5.1.2 Métricas de Evaluación
    - **Accuracy**: Porcentaje de predicciones correctas
    - **F1-Score**: Media armónica de precisión y recall 
    - **Retorno acumulado**: Ganancia/pérdida de la estrategia

    ### 5.2 Estrategia de Backtesting Recursivo

    #### 5.2.1 Walk-Forward Validation
    ```python
    for i in range(0, len(self.data) - self.train_size - self.test_size, self.test_size):
        train = self.data.iloc[i:i + self.train_size]
        test = self.data.iloc[i + self.train_size:i + self.train_size + self.test_size]
    ```
    - **train_size=480**: ≈20 días de datos (480 horas)
    - **test_size=12**: 12 horas de prueba (medio día)
    - Paso de 12 horas (test_size) entre iteraciones

    #### 5.2.2 Implementación de la Estrategia
    ```python
    test.loc[:, 'pred'] = preds
    test.loc[:, 'retorno_strategy'] = test['return_1h'].shift(-1) * test['pred']
    retorno = (1 + test['retorno_strategy'].dropna()).cumprod().iloc[-1] - 1
    ```
    - Se opera solo cuando el modelo predice subida (pred=1)
    - El retorno se calcula como producto acumulativo de los retornos horarios

    ### 5.3 Interpretación de Resultados

    #### 5.3.1 Paradoja Accuracy/Retorno
    - **Accuracy ~50%**: Similar a lanzar una moneda, pero:
        - El modelo puede estar capturando momentos clave de alta probabilidad
        - Bitcoin tiene sesgo alcista histórico (≈60% días positivos)
        - Los aciertos en subidas generan más retorno que las pérdidas en bajadas

    #### 5.3.2 Análisis de la Curva de Retorno
    - **Crecimiento exponencial**: Efecto del interés compuesto
    - **Drawdowns controlados**: El modelo evita grandes pérdidas
    - **Ratio Sharpe implícito**: Retornos consistentes con volatilidad controlada
    """)

    st.markdown("""
    ## 6. Limitaciones y Consideraciones

    1. **Look-ahead bias**: Todos los datos son históricos, sin considerar slippage o liquidez
    2. **Overfitting**: Aunque walk-forward validation mitiga esto, se debe validar con out-of-sample data
    3. **Costos de transacción**: No se incluyen fees de trading (0.1% en Binance)
    4. **Impacto de noticias**: El modelo no considera eventos fundamentales
    """)

    st.markdown("""
    ## 7. Recomendaciones para Mejora

    1. **Optimización de hiperparámetros**: Grid search para LGBM
    2. **Nuevas features**: Incluir order book data o sentimiento
    3. **Stop-loss**: Implementar gestión de riesgo activa
    4. **Ensemble models**: Combinar múltiples modelos para reducir varianza
    5. **Análisis de regímenes de mercado**: Modelos separados para mercados alcistas/bajistas
    """)

    st.markdown("""
    ## 8. Conclusión

    El sistema muestra que incluso con métricas de accuracy modestas, una estrategia que capitaliza correctamente los momentos de alta probabilidad puede generar retornos positivos en activos con sesgo alcista como Bitcoin. La clave está en la combinación de:
    - Feature engineering relevante
    - Gestión de riesgo implícita (solo operar en señales fuertes)
    - Efecto compuesto de retornos positivos consistentes

    Este enfoque es especialmente adecuado para mercados con alta volatilidad y tendencia como criptomonedas, donde la dirección correcta de las predicciones puede compensar una precisión moderada.
    """)

    st.markdown("""
    ## 9. Respuestas Técnicas al Escenario 1

    ### 9.1 Definición de la Variable Objetivo
    ```python
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)  # 1=sube, 0=baja
    ```
    **Decisión Técnica**:  
    Se optó por una definición binaria pura porque:
    - Captura la dirección del mercado sin umbrales arbitrarios
    - Maximiza las muestras disponibles para entrenamiento
    - Es consistente con estrategias de momentum intradía

    **Implementación Clave**:
    - Uso de `shift(-1)` para garantizar no look-ahead bias
    - Tipo `int` para compatibilidad con LightGBM

    ### 9.2 Selección de Ventana Temporal
    ```python
    train_size=480,  # 20 días (480 horas)
    test_size=12     # 12 horas (medio día)
    ```
    **Análisis Técnico**:
    - **480 horas**:
      - Mínimo necesario para calcular EMA50 (50 períodos)
      - Captura ciclos semanales observados en BTC
    - **12 horas**:
      - Equivale a 2 sesiones de trading de 6 horas
      - Permite 40 iteraciones completas de backtesting

    ### 9.3 Ingeniería de Features
    **Indicadores Clave**:
    1. **RSI 14**:
        ```python
        df['rsi'] = 100 - (100/(1 + (avg_gain/avg_loss)))
        ```
        - Ventana estándar de 14 períodos
        - Detecta condiciones extremas (backtest muestra mejor performance en rangos 30-70)

    2. **MACD (12,26)**:
        ```python
        df['macd'] = EMA(12) - EMA(26)
        ```
        - Configuración clásica sin optimización
        - Cruces señalan cambios de momentum

    3. **Volumen Relativo**:
        ```python
        df['volumen_relative'] = volumen_actual / media_movil_24h
        ```
        - Identifica anomalías de volumen vs patrón histórico

    ### 9.4 Manejo de Desequilibrio
    ```python
    LGBMClassifier(class_weight='balanced')
    ```
    **Estrategia**:
    - Compensa automáticamente el sesgo alcista natural 
    - No requiere modificación de datos (preserva estructura temporal)

    ### 9.5 Selección de Modelo
    ```python
    LGBMClassifier(
        n_estimators=100,
        class_weight='balanced',
        # Parámetros por defecto:
        # learning_rate=0.1, max_depth=-1
    )
    ```
    **Ventajas Técnicas**:
    - Eficiente con datos temporales
    - Maneja automáticamente:
      - Features no escalados
      - Valores faltantes
      - Interacciones no lineales

    ### 9.6 Métricas de Evaluación
    ```python
    ['accuracy', 'f1', 'return']  # Usadas en backtesting
    ```
    **Selección Justificada**:
    | Métrica | Fórmula | Uso |
    |---------|---------|-----|
    | F1-Score | 2*(precision*recall)/(precision+recall) | Principal (clases desbalanceadas) |
    | Retorno | ∏(1+retorno) - 1 | Validación económica |
    | Accuracy | (TP+TN)/total | Referencia secundaria |
    """)



def show_btc_regression_doc():
    st.title("Documentación Técnica: Modelo Regresión BTC/USDT")
    
    st.markdown("""
    ## 1. Introducción
    Modelo de regresión LightGBM para predecir retornos logarítmicos diarios de BTC/USDT usando:
    - Datos OHLCV de Binance (1d)
    - 18 features técnicas (RSI, ATR, SMAs, retornos pasados)
    - Validación cruzada temporal (TimeSeriesSplit)
    - Función de pérdida Huber (robusta a outliers)
    """)

    st.markdown("""
    ## 2. Obtención de Datos (`get_crypto_data`)
    ```python
    def get_crypto_data(symbol='BTC/USDT', timeframe='1d', limit=2000):
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    ```
    - **limit=2000**: ≈5.5 años de datos diarios
    - **timeframe='1d'**: Velas diarias para reducir ruido
    - **Estructura**: OHLCV estándar con timestamp como índice
    """)

    st.markdown("""
    ## 3. Feature Engineering (`create_features`)
    ### 3.1 Transformaciones Básicas
    ```python
    df['log_ret'] = np.log(df['close']/df['close'].shift(1))  # Retorno logarítmico
    df['vol_pct'] = df['volume'].pct_change()  # Cambio % volumen
    ```

    ### 3.2 Indicadores Técnicos
    ```python
    # RSI 14 días
    df['rsi_14'] = RSIIndicator(df['close'], window=14).rsi()
    
    # Average True Range (ATR 14)
    df['atr_14'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    
    # Daily Range Normalizado
    df['daily_range'] = (df['high'] - df['low'])/df['open']
    ```

    ### 3.3 Retornos Pasados (Lags)
    ```python
    for lag in [1, 2, 3, 5, 7, 14, 21]:
        df[f'ret_lag_{lag}'] = df['log_ret'].shift(lag)
    ```
    - Captura autocorrelación en retornos
    - Lags seleccionados: corto (1-3d), medio (5-7d) y largo plazo (14-21d)

    ### 3.4 Medias Móviles
    ```python
    for window in [7, 14, 21]:
        df[f'sma_{window}'] = df['close'].rolling(window).mean()  # Simple
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean()  # Exponencial
    ```
    - Ventanas típicas para cripto: 1 semana, 2 semanas, 3 semanas

    ### 3.5 Ratios de Precio
    ```python
    df['close_to_sma7'] = df['close']/df['sma_7']  # Distancia a SMA7
    df['close_to_ema14'] = df['close']/df['ema_14']  # Distancia a EMA14
    ```
    - Indica sobrecompra/sobreventa relativa
    """)

    st.markdown("""
    ## 4. Preparación de Datos (`prepare_data`)
    ```python
    def prepare_data(df, target_col='log_ret', test_size=0.2):
        features = df.drop(columns=[target_col, 'open', 'high', 'low', 'close', 'volume'])
        target = df[target_col]
        
        # Split temporal (80% train - 20% test)
        split_idx = int(len(df) * (1-test_size))
        X_train = features.iloc[:split_idx]
        y_train = target.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_test = target.iloc[split_idx:]
    ```
    - **Split temporal**: No shuffle para preservar estructura de tiempo
    - **Target**: Retorno logarítmico del próximo día (log(P_t+1/P_t))
    - **Exclusión**: OHLCV raw no se usan como features directas
    """)

    st.markdown("""
    ## 5. Modelado (`train_lgbm`)
    ### 5.1 Configuración LightGBM
    ```python
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        objective='huber',  # Robust to outliers
        random_state=42,
        n_jobs=-1
    )
    ```
    - **Huber Loss**: Minimiza MAE pero diferenciable (ideal para retornos con colas pesadas)
    - **num_leaves=31**: Balance entre complejidad y overfitting
    - **learning_rate=0.05**: Tasa de aprendizaje conservadora

    ### 5.2 Validación Cruzada Temporal
    ```python
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            callbacks=[early_stopping(stopping_rounds=50)]
        )
    ```
    - **TimeSeriesSplit**: Valida en bloques temporales posteriores
    - **Early Stopping**: Detiene entrenamiento si MAE no mejora en 50 iteraciones
    """)

    st.markdown("""
    ## 6. Evaluación (`evaluate_model`)
    ### 6.1 Métricas Clave
    ```python
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    ```
    - **MAE**: Error absoluto medio en escala logarítmica
    - **Error porcentual equivalente**: `np.exp(mae)-1` (transformación a %)

    ### 6.2 Interpretación de Resultados
    - MAE de 0.01 ≈ 1% error en retorno esperado
    - Gráfico de predicciones vs reales muestra capacidad predictiva direccional
    """)

    st.markdown("""
    ## 7. Pipeline de Predicción (`predict_btc`)
    ```python
    def predict_btc(model, feature_names):
        new_data = get_crypto_data(limit=100)  # Últimos 100 días
        features = create_features(new_data)
        last_obs = features[feature_names].iloc[-1:].copy()
        
        pred_log_ret = model.predict(last_obs)[0]
        pred_close = new_data['close'].iloc[-1] * np.exp(pred_log_ret)
    ```
    - **Flujo**: Obtener datos → Generar features → Predecir retorno → Calcular precio
    - **Output**: Precio esperado para el próximo cierre diario
    """)

    st.markdown("""
    ## 8. Limitaciones
    1. **Asunción de Mercado**: Eficiente sin eventos disruptivos
    2. **Latencia**: No considera tiempo real de ejecución
    3. **Liquidez**: Asume ejecución a precio de cierre
    4. **Overfitting**: Aunque TS-CV ayuda, requiere monitoreo constante
    """)

    st.markdown("""
    ## 9. Mejoras Potenciales
    1. **Features Adicionales**:
        - Datos on-chain (Glassnode)
        - Sentimiento de redes sociales
    2. **Ensamblado de Modelos**:
        - Combinar con modelos ARIMA para componentes lineales
    3. **Optimización Bayesiana**:
        - Fine-tuning de hiperparámetros con Optuna
    4. **Risk Management**:
        - Bandas de confianza para predicciones
    """)

    st.markdown("""
    ## 10. Conclusión
    Este modelo proporciona:
    - Estimaciones cuantitativas de retornos esperados
    - Base estadística para decisiones de trading
    - Infraestructura extensible para nuevos features
    Limitado por naturaleza estocástica de mercados cripto, pero valioso como:
    - Indicador direccional
    - Herramienta de gestión de riesgo
    - Base para sistemas más complejos
    """)
    st.markdown("""
    ## 11. Respuestas Técnicas al Escenario 2

    ### 11.1 Horizonte y Frecuencia (Cierre Diario)
    ```python
    timeframe='1d' 
    ```
    **Razones en el código:**
    - Reducción de ruido: Velas diarias promedian volatilidad intradía
    - Compatibilidad: Indicadores técnicos (RSI/ATR) configurados para escala diaria
    - Limitación práctica: `limit=2000` sería insuficiente para datos intradía

    ### 11.2 Preprocesamiento de Datos
    ```python
    df['log_ret'] = np.log(df['close']/df['close'].shift(1)) 
    ```
    **Tratamiento aplicado:**
    - Retornos logarítmicos para serie estacionaria
    - Control de outliers con función Huber:
    ```python
    objective='huber'  
    ```

    ### 11.3 Selección de Features Clave
    1. **RSI 14:**
       ```python
       RSIIndicator(df['close'], window=14).rsi()  
       ```
       - Identifica condiciones extremas de mercado

    2. **ATR 14:**
       ```python
       AverageTrueRange(high, low, close, 14)  
       ```
       - Mide volatilidad real incluyendo gaps

    3. **Lags Temporales:**
       ```python
       df[f'ret_lag_{lag}'] = df['log_ret'].shift(lag)  
       ```
       - Captura autocorrelación en retornos

    4. **Ratio Precio/Media:**
       ```python
       df['close_to_sma7'] = df['close']/df['sma_7']  
       ```
       - Normaliza precio respecto a tendencia

    ### 11.4 Modelado y Función de Pérdida
    ```python
    LGBMRegressor(
        objective='huber', 
        max_depth=5,
        num_leaves=31,
        n_estimators=500
    )
    ```
    **Elecciones técnicas:**
    - LightGBM por eficiencia con datos tabulares
    - Huber Loss como compromiso entre MSE y MAE
    - Profundidad limitada (max_depth=5) para evitar overfitting

    ### 11.5 Validación Predictiva
    ```python
    TimeSeriesSplit(n_splits=5)  
    early_stopping(stopping_rounds=50)  
    ```
    **Estrategia implementada:**
    - Walk-forward validation con 5 bloques temporales
    - Métrica principal: MAE (coherente con Huber loss)
    - Early stopping después de 50 rondas sin mejora

    ### 11.6 Tabla de Métricas Clave
    | Componente | Configuración | Justificación |
    |------------|---------------|---------------|
    | Timeframe | 1d | Balance señal/ruido |
    | Target | log_ret | Estacionariedad |
    | Modelo | LightGBM | Eficiencia con datos financieros |
    | Validación | TimeSeriesSplit | Respeta estructura temporal |
    """)


def show_full_technical_doc():
    st.title("Documentación Técnica: Análisis de Indicadores Técnicos y Estrategia de Trading")
    
    st.markdown("""
    ## 1. Introducción
    Este documento describe la implementación técnica de un panel de análisis de indicadores técnicos para trading, desarrollado con Python, Streamlit y Plotly. El sistema visualiza múltiples indicadores técnicos en un gráfico integrado que permite identificar oportunidades de trading basadas en convergencia de señales.
    """)

    st.markdown("""
    ## 2. Indicadores Implementados
    ### 2.1 Medias Móviles Exponenciales (EMA 10 y EMA 50)
    **Propósito**: Identificar la dirección de la tendencia y posibles puntos de reversión

    **Interpretación**:
    - EMA 10 (corto plazo): Reacciona rápidamente a cambios de precio
    - EMA 50 (medio plazo): Proporciona tendencia subyacente
    - Cruce alcista (EMA 10 > EMA 50): Señal de compra potencial
    - Cruce bajista (EMA 10 < EMA 50): Señal de venta potencial
    """)

    st.markdown("""
    ### 2.2 Índice de Fuerza Relativa (RSI)
    **Propósito**: Medir condiciones de sobrecompra/sobreventa

    **Interpretación**:
    - RSI > 70: Sobrecompra (posible corrección)
    - RSI < 30: Sobreventa (posible rebote)
    - Divergencias entre RSI y precio pueden indicar debilidad de tendencia
    """)

    st.markdown("""
    ### 2.3 MACD (Moving Average Convergence Divergence)
    **Propósito**: Identificar cambios en el momentum

    **Interpretación**:
    - Cruce alcista MACD > Señal: Momentum positivo
    - Cruce bajista MACD < Señal: Momentum negativo
    - Histograma positivo/negativo refuerza señal
    """)

    st.markdown("""
    ### 2.4 ATR (Average True Range)
    **Propósito**: Medir volatilidad del mercado
                
    **Interpretación**:
    - Valores altos indican alta volatilidad
    - Útil para establecer stops dinámicos (ej: 2xATR)
    - Puede indicar inicio/fin de tendencias fuertes
    """)

    st.markdown("""
    ## 3. Estrategia de Trading Combinada
    ### 3.1 Concepto Base: "Cruce de EMAs con Confirmación"
    La estrategia se basa en el cruce de medias móviles (10 y 50 períodos) con confirmación de otros indicadores para reducir señales falsas.

    ### 3.2 Reglas de Entrada
    **Compra (Tendencia Alcista)**:
    1. EMA 10 cruza por encima de EMA 50 (confirmación primaria)
    2. MACD histograma positivo y línea MACD > señal (confirmación momentum)
    3. RSI entre 30-70 (evitar sobrecompra inicial)
    4. ATR en aumento (confirmación volatilidad/participación)

    **Venta (Tendencia Bajista)**:
    1. EMA 10 cruza por debajo de EMA 50
    2. MACD histograma negativo y línea MACD < señal
    3. RSI entre 30-70 (evitar sobreventa inicial)
    4. ATR en aumento (confirmación fuerza bajista)

    ### 3.3 Gestión de Riesgo
    **Stop Loss**:
    - Dinámico: Precio de entrada ± (2 x ATR actual)
    - Fijo: Porcentaje del capital (ej: 1-2%)

    **Take Profit**:
    - Nivel 1: 1.5 x ATR (retirar 50% posición)
    - Nivel 2: 3 x ATR (retirar 25% posición)
    - Nivel 3: Seguir tendencia con trailing stop (EMA 10 como guía)

    ### 3.4 Filtro Temporal (Tendencias Bajistas)
    En mercados bajistas (EMA 50 con pendiente negativa):
    - Requerir RSI < 40 para compras (mayor seguridad)
    - Aumentar stop loss a 2.5-3 x ATR (mayor volatilidad)
    - Reducir tamaño de posición (50% normal)
    """)

    st.markdown("""
    ## 4. Consideraciones y Mejoras
    ### 4.1 Limitaciones
    - Retraso inherente de indicadores basados en medias móviles
    - Señales falsas en mercados laterales (usar ATR como filtro)
    - Optimización de parámetros necesaria para diferentes activos

    ### 4.2 Mejoras Potenciales
    - Incorporar patrón de velas en puntos de decisión
    - Añadir volumen balanceado para confirmar participación
    - Implementar backtesting integrado para validar estrategia
    - Incluir alertas automáticas para cruces clave
    """)

    st.markdown("""
    ## 5. Conclusión
    Este sistema proporciona un enfoque metodológico para identificar cambios de tendencia utilizando múltiples confirmaciones técnicas. La combinación de EMAs para dirección, MACD para momentum, RSI para condiciones de mercado y ATR para volatilidad crea un marco robusto para la toma de decisiones. La implementación visual interactiva facilita la identificación rápida de oportunidades mientras mantiene el contexto técnico completo.

    La estrategia propuesta funciona particularmente bien en mercados con tendencia definida.
    """)

if st.button("Doc Técnica - Parte 1"):
    if 'show_doc' not in st.session_state or not st.session_state.show_doc:
        st.session_state.show_doc = True
    else:
        st.session_state.show_doc = not st.session_state.show_doc


if st.session_state.get('show_doc', False):
    show_full_technical_doc()


if st.button("Doc Técnica - Parte 2 -1"):
    if 'show_doc' not in st.session_state or not st.session_state.show_doc:
        st.session_state.show_doc = True
    else:
        st.session_state.show_doc = not st.session_state.show_doc

if st.session_state.get('show_doc', False):
    show_full_technical_doc()

if st.button("Doc Técnica - Parte 2 - 2"):
    if 'show_doc' not in st.session_state or not st.session_state.show_doc:
        st.session_state.show_doc = True
    else:
        st.session_state.show_doc = not st.session_state.show_doc


if st.session_state.get('show_doc', False):
    show_btc_regression_doc()

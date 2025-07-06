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

if st.button("Doc Técnica - Parte 2 -1"):
    if 'show_doc' not in st.session_state or not st.session_state.show_doc:
        st.session_state.show_doc = True
    else:
        st.session_state.show_doc = not st.session_state.show_doc

if st.session_state.get('show_doc', False):
    show_full_technical_doc()
import streamlit as st

st.title("Part 3")

import streamlit as st

markdown_string = """
# Estrategia de Trading con IA/ML para Futuros en Binance

## 1. Definición del Edge – Ventaja Competitiva

**Objetivo del sistema:**
Detectar señales de entrada/salida en futuros con apalancamiento moderado (3x a 5x), alta probabilidad de éxito y riesgo controlado.

**Edge técnico específico:**
- Filtrado de entradas con alta probabilidad de éxito (patrones técnicos + divergencias + ML).
- Stop-loss en estructura de precio (mínimo/máximo anterior).
- Take-profit fijo de 2:1 respecto al riesgo.
- Riesgo limitado a 5% del capital.

**Justificación:**  
El edge está en la calidad de las entradas y la estricta gestión del riesgo, evitando operar en condiciones adversas.

---

## 2. Arquitectura de Solución (Pipeline IA/ML)

| Módulo | Descripción técnica |
|--------|---------------------|
| **Ingesta** | Datos de Binance: precio OHLCV, order book, funding rate. |
| **Procesamiento** | Resampleo, limpieza, sincronización temporal. |
| **Feature Engineering** | RSI, MACD, EMAs, volumen relativo, patrones de velas, divergencias, distancia a zonas clave. |
| **Feature Store** | Almacenamiento histórico alineado temporalmente. |
| **Modelado ML** | Clasificación binaria (ej. XGBoost): predice si TP 2xR se alcanzará antes que el SL. |
| **Scoring** | Inferencia en vivo: genera probabilidad y dirección (long/short). |
| **Decisión** | Se filtran señales con probabilidad > 0.7, SL estructural, exposición < 5% del capital. |
| **Ejecución** | Cálculo de posición, apalancamiento, SL/TP, ejecución vía API. |
| **Gestión Activa** | Monitoreo del trade en tiempo real, trailing stop si aplica. |
| **Logging** | Registro detallado de cada trade para análisis y feedback. |

---

## 3. Gestión de Riesgo y Slippage

| Criterio | Implementación |
|---------|----------------|
| **Stop-loss** | Basado en último mínimo/máximo relevante. |
| **Take-profit** | Siempre 2x el riesgo definido por el SL. |
| **Riesgo máximo** | Nunca más del 5% del capital por trade. |
| **Apalancamiento** | Moderado: entre 3x y 5x. Controlado según exposición. |
| **Slippage** | Medido en backtest, umbral máximo del 0.3%. |

---

## 4. Backtest Realista

| Factor | Simulación |
|--------|------------|
| **Costos** | Comisión + slippage por par. |
| **Latencia** | Delay artificial de 1-2 segundos. |
| **Volumen** | Se verifica liquidez disponible. |
| **Overfitting** | Validación *walk-forward*. |

---

## 5. Monitorización y Feedback

| Métrica | Uso |
|--------|-----|
| **Sharpe rolling (30d)** | Medida de consistencia del sistema. |
| **Max Drawdown** | Detona revisión del modelo si se supera cierto nivel. |
| **Hit Ratio** | Precisión de señales con P > 0.7. |
| **Expectancy** | Retorno promedio esperado por trade. |
| **Turnover** | Cantidad de operaciones por período. |
| **Slippage efectivo** | Evaluación de ejecución real vs esperada. |

**Feedback Loop:**
- Si el rendimiento baja (Sharpe o hit ratio), se detiene y reentrena el modelo.
- Se revisan importancias de variables.
- Se aplican alertas automáticas por cambios en métricas clave.

---

## Diagrama Lógico del Sistema

"""
st.markdown(markdown_string, unsafe_allow_html=True)

st.graphviz_chart('''
digraph trading_pipeline {
    Ingesta -> Procesamiento
    Procesamiento -> FeatureEngineering
    FeatureEngineering -> Prediccion
    Prediccion -> Decision

    Decision -> Ejecutar [label="Probabilidad > 0.7"]
    Decision -> No_Operar [label="Probabilidad ≤ 0.7"]

    Ejecutar -> Monitoreo
    Monitoreo -> Logging
}
''')
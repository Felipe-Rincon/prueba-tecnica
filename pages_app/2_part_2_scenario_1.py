import plotly.graph_objects as go
import streamlit as st

from services.binance.binance_api import fetch_binance_data
from modules.model_up_low.up_low import calcular_features, backtester_and_predict_next

st.title("Part 2 - Scenario 1")

st.header("Enter Stock Ticker")
ticker = st.text_input(
    "Enter the ticker symbol of Binance (BTC/USDT):",
    value="BTC/USDT",
    max_chars=20
).upper()


if st.button("Predict Next Hour"):

    datos = fetch_binance_data(symbol=ticker, timeframe='1h', limit=4300)
    with st.spinner("Predict data..."):
        if datos is not None:
            datos = calcular_features(datos)

            backtester = backtester_and_predict_next(datos, train_size=480, test_size=12)
            resultados = backtester.run()

            if not resultados.empty:
                st.subheader("📊 Results of the Backtesting")
                st.dataframe(resultados)

                st.subheader("📌 Average Metrics")
                st.metric("Accuracy", f"{resultados['accuracy'].mean():.2%}")
                st.metric("F1-Score", f"{resultados['f1'].mean():.2%}")
                st.metric("Cumulative Return", f"{resultados['return'].mean():.2%}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=resultados.index,
                    y=(resultados['return'] + 1).cumprod(),
                    line=dict(color='#4CAF50', width=2),
                    name='Return'
                ))
                fig.add_hline(y=1, line_dash="dot", line_color="red")
                fig.update_layout(
                    title='Return Cumulative (1h) - 480 Train/12 Test',
                    yaxis_title='Return (x capital)',
                    xaxis_title='Date',
                    showlegend=False,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)

                if backtester.model and hasattr(backtester, 'features'):
                    latest_data = datos.iloc[-1][backtester.features].values.reshape(1, -1)
                    prediction = "📈 UPLOAD" if backtester.model.predict(latest_data)[0] == 1 else "📉 LOWER"
                    st.success(f"🔮 Next hour forecast: {prediction}")
            else:
                st.warning("⚠️ Results could not be calculated (check amount of data)")
        else:
            st.error("❌ Data download error")
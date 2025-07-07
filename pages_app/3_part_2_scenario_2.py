import plotly.graph_objects as go
import numpy as np
import streamlit as st

from services.binance.binance_api import get_crypto_data
from modules.model_1d.forecast_1d import create_features, prepare_data, train_lgbm, evaluate_model, predict_btc


st.title("Part 2 - Scenario 2")

st.header("Enter Stock Ticker")
ticker = st.text_input(
    "Enter the ticker symbol of Binance (BTC/USDT):",
    value="BTC/USDT",
    max_chars=20
).upper()

if st.button("Predict Next Day"):

    btc_data = get_crypto_data(symbol=ticker, timeframe='1d', limit=2000)

    with st.spinner("Predict data..."):
        if btc_data is not None:

            features = create_features(btc_data)

            X_train, X_test, y_train, y_test = prepare_data(features)

            model = train_lgbm(X_train, y_train)

            preds, mae = evaluate_model(model, X_test, y_test)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test.index, y=np.exp(y_test)-1, 
                                 mode='lines', name='Real'))
            fig.add_trace(go.Scatter(x=y_test.index, y=np.exp(preds)-1, 
                                 mode='lines', name='Predicted', opacity=0.7))
            fig.update_layout(title="Real vs Predicted Returns",
                          xaxis_title="Date",
                          yaxis_title="Price")

            st.plotly_chart(fig)

            st.write(f"\nMAE on test: {mae:.4f}")
            st.write(f"Equivalent percentage error: {np.exp(mae)-1:.2%}")

            new_data, pred_close, pred_log_ret = predict_btc(model, X_train.columns)

            st.subheader(f"\nPrediction for next close:")
            st.write(f"Current price: ${new_data['close'].iloc[-1]:.2f}")
            st.write(f"Predicted price: ${pred_close:.2f}")
            st.write(f"Expected return: {pred_log_ret:.2%}")


            


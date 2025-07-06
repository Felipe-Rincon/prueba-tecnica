import streamlit as st

from services.alpha_vantage.alpha_vantage_api import fetch_alpha_vantage_data
from services.yahoo_finance.yahoo_api import fetch_yahoo_data
from modules.plot.plot_generator import plot_with_indicators

st.title("Part 1")

st.header("Select Data Source")
data_source = st.radio(
    "Choose your data source:",
    ("Yahoo Finance", "Alpha Vantage"),
    horizontal=True
)

st.header("Enter Stock Ticker")
ticker = st.text_input(
    "Enter the stock ticker symbol (e.g., AAPL for Apple):",
    value="AAPL",
    max_chars=10
).upper()


if st.button("Fetch Data"):
        if not ticker:
            st.warning("Please enter a ticker symbol")

        st.subheader(f"Fetching {ticker} data from {data_source}")
        
        tab15, tab4h, tab_daily = st.tabs(["15 Minute", "1 Hour", "Daily"])
        
        with st.spinner("Fetching data..."):
            if data_source == "Yahoo Finance":
                with tab15:
                    df_15min = fetch_yahoo_data(ticker, '15min')
                    st.write("15 Minute Data")
                    st.dataframe(df_15min)
                    plot_with_indicators(df_15min, f"{ticker} 15min")
                
                with tab4h:
                    df_4h = fetch_yahoo_data(ticker, '1h')
                    st.write("1 Hour Data")
                    st.dataframe(df_4h)
                    plot_with_indicators(df_4h, f"{ticker} 1h")
                
                with tab_daily:
                    df_daily = fetch_yahoo_data(ticker, '1d')
                    st.write("Daily Data")
                    st.dataframe(df_daily)
                    plot_with_indicators(df_daily, f"{ticker} Daily")
            
            else:
                with tab15:
                    df_15min = fetch_alpha_vantage_data(ticker, '15min')
                    st.write("15 Minute Data")
                    st.dataframe(df_15min)
                    plot_with_indicators(df_15min, f"{ticker} 15min")

                
                with tab4h:
                    df_4h = fetch_alpha_vantage_data(ticker, '1h')
                    st.write("1 Hour Data")
                    st.dataframe(df_4h)
                    plot_with_indicators(df_4h, f"{ticker} 1h")
                    
                
                with tab_daily:
                    df_daily = fetch_alpha_vantage_data(ticker, '1d')
                    st.write("Daily Data")
                    st.dataframe(df_daily)
                    plot_with_indicators(df_daily, f"{ticker} Daily")

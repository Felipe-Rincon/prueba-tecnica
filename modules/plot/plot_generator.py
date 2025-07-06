import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.indicators.indicators import calculate_rsi, calculate_ema, calculate_macd, calculate_ichimoku, calculate_atr

def plot_with_indicators(df, title):
    try:
        close_prices = df['Close'].values
        high_prices = df['High'].values
        low_prices = df['Low'].values

        ema_20 = calculate_ema(close_prices, 10)
        ema_50 = calculate_ema(close_prices, 50)
        rsi = calculate_rsi(close_prices)
        macd_line, signal_line, histogram = calculate_macd(close_prices)
        atr = calculate_atr(high_prices, low_prices, close_prices)

        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.15, 0.15],
            subplot_titles=(f'{title} - Análisis Técnico', 'ATR', 'RSI', 'MACD', 'Volumen')
        )

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Precio',
            increasing_line_color='#2ECC71',
            decreasing_line_color='#E74C3C'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=ema_20,
            line=dict(color='#F39C12', width=1.5),
            name='EMA 10'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=ema_50,
            line=dict(color='#8E44AD', width=1.5),
            name='EMA 50'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=atr,
            line=dict(color='#3498DB', width=1.2),
            name='ATR',
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.2)'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=rsi,
            line=dict(color='#9B59B6', width=1.5),
            name='RSI',
            fill='tozeroy',
            fillcolor='rgba(155, 89, 182, 0.1)'
        ), row=3, col=1)
        
        fig.add_hline(y=30, row=3, col=1, line_dash="dot", line_color="#E74C3C", opacity=0.7)
        fig.add_hline(y=70, row=3, col=1, line_dash="dot", line_color="#2ECC71", opacity=0.7)
        fig.add_hrect(y0=30, y1=70, row=3, col=1, 
                     fillcolor="rgba(149, 165, 166, 0.1)", line_width=0)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=macd_line,
            line=dict(color='#2980B9', width=1.5),
            name='MACD'
        ), row=4, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=signal_line,
            line=dict(color='#D35400', width=1.5),
            name='Señal'
        ), row=4, col=1)
        
        fig.add_trace(go.Bar(
            x=df.index,
            y=histogram,
            name='Histograma',
            marker_color=np.where(histogram > 0, '#27AE60', '#C0392B'),
            opacity=0.7
        ), row=4, col=1)
        
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volumen',
            marker_color=np.where(df['Close'] >= df['Open'], '#27AE60', '#C0392B'),
            opacity=0.7
        ), row=5, col=1)
        
        fig.update_layout(
            height=1000,
            margin=dict(t=40, b=40, l=40, r=40),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            dragmode='pan'
        )
        
        fig.update_yaxes(autorange=True, fixedrange=False, row=1, col=1)
        fig.update_yaxes(autorange=True, fixedrange=False, row=2, col=1)
        fig.update_yaxes(autorange=True, fixedrange=False, row=3, col=1)
        fig.update_yaxes(autorange=True, fixedrange=False, row=4, col=1)
        fig.update_yaxes(autorange=True, fixedrange=False, row=5, col=1)
        
        st.plotly_chart(fig, use_container_width=True, config={
            'scrollZoom': True,
            'responsive': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
        })
        
    except Exception as e:
        st.error(f"Error al generar gráfico: {str(e)}")
        st.line_chart(df['Close'])
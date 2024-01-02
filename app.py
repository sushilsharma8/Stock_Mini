import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Dummy hardcoded username and password
USERNAME = "qwerty"
PASSWORD = "7410"

st.set_page_config(
    page_title="Stock Market Predictor App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

def login():
    st.title("Stock Market Predictor App - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.success("Logged in successfully!")
            main()
        else:
            st.error("Invalid username or password")

def main():
    st.title("Stock Market Predictor App")

    stock = st.text_input('Enter Stock Symbol', 'GOOG')
    start = '2012-01-01'
    end = '2022-12-31'

    data = yf.download(stock, start, end)

    if data.empty:
        st.warning("No data found for the given stock symbol.")
    else:
        st.subheader('Stock Data')
        st.write(data)

        # Candlestick chart using Plotly
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'],
                                             name='Candlesticks')])
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)

        # Additional charts and indicators can be added here
        # Example: Bollinger Bands
        st.subheader('Bollinger Bands')
        bollinger_data = pd.DataFrame(index=data.index)
        bollinger_data['Middle Band'] = data['Close'].rolling(window=20).mean()
        bollinger_data['Upper Band'] = bollinger_data['Middle Band'] + 2 * data['Close'].rolling(window=20).std()
        bollinger_data['Lower Band'] = bollinger_data['Middle Band'] - 2 * data['Close'].rolling(window=20).std()

        fig_bollinger = go.Figure()
        fig_bollinger.add_trace(go.Scatter(x=bollinger_data.index, y=bollinger_data['Middle Band'], mode='lines',
                                           name='Middle Band', line=dict(color='blue')))
        fig_bollinger.add_trace(go.Scatter(x=bollinger_data.index, y=bollinger_data['Upper Band'], mode='lines',
                                           name='Upper Band', line=dict(color='red')))
        fig_bollinger.add_trace(go.Scatter(x=bollinger_data.index, y=bollinger_data['Lower Band'], mode='lines',
                                           name='Lower Band', line=dict(color='green')))
        fig_bollinger.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_bollinger)

if __name__ == "__main__":
    login()

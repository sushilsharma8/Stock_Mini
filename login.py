import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from keras.models import load_model
import matplotlib.pyplot as plt
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
model = load_model('C:\\Users\\sushi\\Downloads\\Stock_Mini\\Stock Predictions Model.keras')


def main():

    st.title("Stock Market Predictor App")

    # Sidebar for login
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.success("Logged in successfully!")
            show_predictor()
        else:
            st.error("Invalid username or password")


def show_predictor():
    st.header("Stock Market Predictor")

    stock = st.text_input('Enter Stock Symbol', 'GOOG')
    start = '2012-01-01'
    end = '2022-12-31'

    data = yf.download(stock, start, end)

    if data.empty:
        st.warning("No data found for the given stock symbol.")
    else:
        st.subheader('Stock Data')
        st.write(data)

        # Calculate moving averages for buy/sell signals
        data['Short_MA'] = data['Close'].rolling(window=20).mean()
        data['Long_MA'] = data['Close'].rolling(window=50).mean()

        # Create candlestick chart with buy/sell signals
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Stock Prices', 'Volume'))

        # Candlestick chart
        fig.add_trace(go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name='Candlesticks'), row=1, col=1)

        # Moving averages
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Short_MA'], mode='lines', name='Short MA', line=dict(color='red')))
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Long_MA'], mode='lines', name='Long MA', line=dict(color='blue')))

        # Volume chart
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'), row=2, col=1)


        # Update x-axis type to category for better date display
        fig.update_xaxes(type='category')

        # Remove time from x-axis of the volume chart
        fig.update_xaxes(
            tickformat='%Y-%m-%d',  # Set the desired date format
            row=2, col=1
        )

        # Add buy/sell signals to the chart
        for i in range(1, len(data)):
            if data['Short_MA'][i] > data['Long_MA'][i] and data['Short_MA'][i - 1] <= data['Long_MA'][i - 1]:
                fig.add_trace(go.Scatter(x=[data.index[i]], y=[data['Close'][i]], mode='markers',
                                         marker=dict(color='green', size=8),
                                         name='Buy Signal'), row=1, col=1)
            elif data['Short_MA'][i] < data['Long_MA'][i] and data['Short_MA'][i - 1] >= data['Long_MA'][i - 1]:
                fig.add_trace(go.Scatter(x=[data.index[i]], y=[data['Close'][i]], mode='markers',
                                         marker=dict(color='red', size=8),
                                         name='Sell Signal'), row=1, col=1)

        # Update layout
        fig.update_layout(height=700, width=1030, title_text='Stock Price and Volume Chart')
        st.plotly_chart(fig)

        data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
        data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        pas_100_days = data_train.tail(100)
        data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
        data_test_scale = scaler.fit_transform(data_test)

        # Layout for the three graphs side by side
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader('Price vs MA50')
            ma_50_days = data.Close.rolling(50).mean()
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=data.index, y=ma_50_days, mode='lines', name='MA50', line=dict(color='red')))
            fig1.add_trace(go.Scatter(x=data.index, y=data.Close, mode='lines', name='Close', line=dict(color='green')))
            fig1.update_layout(height=400, width=400)
            st.plotly_chart(fig1)

        with col2:
            st.subheader('Price vs MA50 vs MA100')
            ma_100_days = data.Close.rolling(100).mean()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=data.index, y=ma_50_days, mode='lines', name='MA50', line=dict(color='red')))
            fig2.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100', line=dict(color='blue')))
            fig2.add_trace(go.Scatter(x=data.index, y=data.Close, mode='lines', name='Close', line=dict(color='green')))
            fig2.update_layout(height=400, width=400)
            st.plotly_chart(fig2)

        with col3:
            st.subheader('Price vs MA100 vs MA200')
            ma_200_days = data.Close.rolling(200).mean()
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100', line=dict(color='red')))
            fig3.add_trace(go.Scatter(x=data.index, y=ma_200_days, mode='lines', name='MA200', line=dict(color='blue')))
            fig3.add_trace(go.Scatter(x=data.index, y=data.Close, mode='lines', name='Close', line=dict(color='green')))
            fig3.update_layout(height=400, width=400)
            st.plotly_chart(fig3)

        x = []
        y = []

        for i in range(100, data_test_scale.shape[0]):
            x.append(data_test_scale[i - 100:i])
            y.append(data_test_scale[i, 0])

        x, y = np.array(x), np.array(y)

        predict = model.predict(x)

        scale = 1 / scaler.scale_

        predict = predict * scale
        y = y * scale

        st.subheader('Original Price vs Predicted Price')
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=data_test.index, y=predict.flatten(), mode='lines', name='Original Price',
                                  line=dict(color='red')))
        fig4.add_trace(
            go.Scatter(x=data_test.index, y=y, mode='lines', name='Predicted Price', line=dict(color='green')))
        fig4.update_layout(height=600, width=1000)
        st.plotly_chart(fig4)

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
        fig_bollinger.update_layout(xaxis_rangeslider_visible=False, height=700,
                                    width=1000)  # Adjust height and width here
        st.plotly_chart(fig_bollinger)


if __name__ == "__main__":
    main()

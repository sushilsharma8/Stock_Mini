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
    # Set the background color
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(to right, #3494e6, #ec6ead) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
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
        fig.update_layout(height=600, width=800, title_text='Stock Price and Volume Chart')
        st.plotly_chart(fig)


        data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
        data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        pas_100_days = data_train.tail(100)
        data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
        data_test_scale = scaler.fit_transform(data_test)

        st.subheader('Price vs MA50')
        ma_50_days = data.Close.rolling(50).mean()
        fig1 = plt.figure(figsize=(8, 6))
        plt.plot(ma_50_days, 'r')
        plt.plot(data.Close, 'g')
        plt.show()
        st.pyplot(fig1)

        st.subheader('Price vs MA50 vs MA100')
        ma_100_days = data.Close.rolling(100).mean()
        fig2 = plt.figure(figsize=(8, 6))
        plt.plot(ma_50_days, 'r')
        plt.plot(ma_100_days, 'b')
        plt.plot(data.Close, 'g')
        plt.show()
        st.pyplot(fig2)

        st.subheader('Price vs MA100 vs MA200')
        ma_200_days = data.Close.rolling(200).mean()
        fig3 = plt.figure(figsize=(8, 6))
        plt.plot(ma_100_days, 'r')
        plt.plot(ma_200_days, 'b')
        plt.plot(data.Close, 'g')
        plt.show()
        st.pyplot(fig3)

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
        fig4 = plt.figure(figsize=(8, 6))
        plt.plot(predict, 'r', label='Original Price')
        plt.plot(y, 'g', label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.show()
        st.pyplot(fig4)
        # Rest of your code for the stock predictor

if __name__ == "__main__":
    main()

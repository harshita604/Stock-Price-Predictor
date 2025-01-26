import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import math
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import plotly.graph_objs as go

# Function to download the data for a given ticker
def download_data(ticker):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_year = datetime.now().year - 4
    start_date = f'{start_year}-01-01'
    
    # Download the data
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

# Function to visualize the last 6 months of data
def plot_last_6_months(df, ticker):
    # Filter the last 6 months of data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=180)
    df_last_6_months = df[df.index >= start_date]
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df_last_6_months['Close'])
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title(f'{ticker} - Closing Price History (Last 6 Months)')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)

# Function to predict next 7 days using LSTM model
def predict_next_7_days(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    data = df[['Close', 'SMA_50', 'SMA_200']] 
    data.dropna()
    dataset= data.values
    train_data_len= math.ceil(len(dataset) * 0.8)
    #Data Scaling
    scaler= MinMaxScaler(feature_range=(0,1))
    scaled_data= scaler.fit_transform(dataset)  # computes the minimum and maximum data for scaling
    train_data= scaled_data[0:train_data_len, :]
    x_train=[]
    y_train=[]
    
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    
    x_train= np.array(x_train)
    y_train= np.array(y_train)
    x_train= np.reshape( x_train, (x_train.shape[0], x_train.shape[1], 1))
    model= Sequential()
    model.add(LSTM(200, return_sequences=True,  input_shape= (x_train.shape[1],1)))
    model.add(LSTM(200, return_sequences= False))
    #model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=3)
    model.fit(x_train, y_train, batch_size=128, epochs=50)
    last_60_days = scaled_data[-60:,0]
    forecast_input = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
    forecast = []
    # Predict for the next 7 days
    for day in range(7):
        predicted_price = model.predict(forecast_input)
        forecast.append(predicted_price[0, 0])
        # Update the input with the new prediction
        new_input = np.append(forecast_input[0, 1:, 0], predicted_price[0, 0])
        forecast_input = np.reshape(new_input, (1, 60, 1))
    forecast_full = np.zeros((len(forecast), 3))
    forecast_full[:, 0] = forecast
    predicted_prices = scaler.inverse_transform(forecast_full)[:, 0]
    # Print predictions
    print("Predicted stock prices for the next 7 days:")
    for i, price in enumerate(predicted_prices, 1):
        print(f"Day {i}: {price:.2f}")
    return predicted_prices



# Function to visualize the predicted 7-day prices
def plot_predicted_prices(predicted_prices):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 8), predicted_prices, marker='o', linestyle='-', color='b', label='Predicted Prices')
    plt.title('Stock Price Prediction for the Next 7 Days')
    plt.xlabel('Day')
    plt.ylabel('Price (USD)')
    plt.xticks(range(1, 8))  # Set x-ticks for days 1 to 7
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)


# Streamlit App
def main():
    st.title('Stock Price Prediction App')
    st.markdown("""
    ## What This App Does:
    This application predicts stock prices for the next 7 days using historical data from Yahoo Finance.
    It uses **LSTM (Long Short-Term Memory)** models to make predictions based on the closing stock prices.
    
    The app also incorporates **Moving Averages** (50-day and 200-day) to provide insights into trends over 
    different periods. These moving averages help the model better capture long-term and short-term trends.
    
    ### Disclaimer:
    Please note that this is just a prediction model. The stock market is highly volatile and unpredictable.
    These predictions should not be considered financial advice.
    """)
    
    # Ticker input 
    ticker = st.text_input('Enter stock ticker (e.g., AAPL, TSLA, MSFT):')
    
    if ticker:
        # Fetch data
        df = download_data(ticker)
        
        # Show current details
        st.subheader(f'Current details for {ticker}:')
        st.write(df.tail())
        
        # Plot the last 6 months of data
        st.subheader('Closing Price History (Last 6 Months)')
        plot_last_6_months(df, ticker)
        
        # Predict and visualize the next 7 days
        st.subheader(f'Predicted Prices for the Next 7 Days:')
        predicted_prices = predict_next_7_days(df)
        predicted_df = pd.DataFrame({
            "Date": pd.date_range(start=datetime.today(), periods=7),
            "Predicted Price": predicted_prices })
        st.write(predicted_df)
        plot_predicted_prices(predicted_prices)

if __name__ == '__main__':
    main()

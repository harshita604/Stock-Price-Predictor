# Stock Price Prediction App

This web application predicts stock prices for the next 7 days using historical data from Yahoo Finance. The model uses Long Short-Term Memory (LSTM), a type of recurrent neural network (RNN), to make predictions based on the closing stock prices. It also incorporates moving averages (50-day and 200-day) to provide insights into long-term and short-term trends.


### Features:
 1) Predicts stock prices for the next 7 days based on historical data.
 2) Visualizes the stock's closing prices and moving averages (SMA 50, SMA 200) for the last 6 months.
 3) LSTM model with adjustable hyperparameters (epochs and batch size).
 4) Displays performance metrics (R2 Score and RMSE) of the model.

To run the application:
1) Clone the respository
 ```
    git clone https://github.com/harshita604/stock-price-prediction-app.git
  ````

2) Navigate to the project folder
   
4) Install the required libraries using the requirements.txt file:
```
   pip install -r requirements.txt
```
4) Run the project:
 ```
   streamlit run app.py
  ```

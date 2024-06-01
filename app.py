import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# Initialize session state
if 'predict_button_pressed' not in st.session_state:
    st.session_state.predict_button_pressed = False

# List of stock tickers for each stock exchange
stock_tickers_by_exchange = {
    'New York Stock Exchange': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
    'Nasdaq': ['TSLA', 'NFLX', 'INTC', 'AMD', 'NVDA'],
    'Shanghai Stock Exchange': ['600000.SS', '600004.SS', '600009.SS', '600016.SS', '600028.SS'],
    'Euronext': ['ASML.AS', 'ADYEN.AS', 'MT.AS', 'HEIA.AS', 'DSM.AS'],
    'HKEX': ['0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK'],
    'TYO': ['7203.T', '6758.T', '9984.T', '9432.T', '6861.T'],
    'London Stock Exchange': ['HSBA.L', 'BP.L', 'RDSA.L', 'GLEN.L', 'VOD.L'],
    'Toronto Stock Exchange': ['RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'ENB.TO'],
    'Bombay Stock Exchange': ['RELIANCE.BO', 'TCS.BO', 'INFY.BO', 'HDFCBANK.BO', 'ICICIBANK.BO'],
}

st.title('Stock Trend Prediction')

# Select Stock Exchange at the top
selected_exchange = st.selectbox('Choose a Stock Exchange', list(stock_tickers_by_exchange.keys()))

# Provide top tickers for the selected stock exchange
top_tickers = stock_tickers_by_exchange.get(selected_exchange, [])[:10]

# Display top tickers in a dropdown menu with an "Other" option
user_input = st.selectbox('Enter the Stock Ticker', top_tickers + ['Other'])

# If the user selects "Other," provide an input field for custom ticker
if user_input == 'Other':
    user_input = st.text_input('Enter Custom Ticker')

# Date range selection
start_date = st.date_input("From:", value=pd.to_datetime('2020-01-01'))
end_date = st.date_input("To:", value=pd.to_datetime('2024-06-01'))

# Check if Enter key is pressed
if st.button('Predict') or st.session_state.predict_button_pressed:
    st.session_state.predict_button_pressed = True

    if user_input:
        try:
            # Fetch stock data
            ticker = yf.Ticker(user_input)
            df = ticker.history(start=start_date, end=end_date)

            # Check if the dataframe is not empty
            if not df.empty:
                # Describing Data
                st.subheader('Data Description')
                st.write(df.drop(['Dividends', 'Stock Splits'], axis=1).describe())

                # Visualizations
                st.subheader('Closing Price Vs Time Chart')
                fig = plt.figure(figsize=(12, 6))
                plt.plot(df.Close)
                st.pyplot(fig)

                st.subheader('Closing Price Vs Time Chart with 100MA')
                ma100 = df.Close.rolling(100).mean()
                fig = plt.figure(figsize=(12, 6))
                plt.plot(df.Close, label='Close Price')
                plt.plot(ma100, label='100MA')
                plt.legend()
                st.pyplot(fig)

                st.subheader('Closing Price Vs Time Chart with 100MA & 200MA')
                ma200 = df.Close.rolling(200).mean()
                fig = plt.figure(figsize=(12, 6))
                plt.plot(df.Close, label='Close Price')
                plt.plot(ma100, label='100MA')
                plt.plot(ma200, label='200MA')
                plt.legend()
                st.pyplot(fig)

                # Splitting Data into Training and Testing
                data_training = df['Close'][0:int(len(df) * 0.70)]  # 70% of data for training
                data_testing = df['Close'][int(len(df) * 0.70):]  # 30% of data for testing

                # Scaling data
                scaler = MinMaxScaler(feature_range=(0, 1))
                data_training_array = scaler.fit_transform(np.array(data_training).reshape(-1, 1))

                # Prepare data for LSTM model
                x_train = []
                y_train = []

                for i in range(100, len(data_training_array)):
                    x_train.append(data_training_array[i-100:i])
                    y_train.append(data_training_array[i, 0])

                x_train, y_train = np.array(x_train), np.array(y_train)

                # Model building
                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(Dropout(0.2))
                model.add(LSTM(units=50, return_sequences=False))
                model.add(Dropout(0.2))
                model.add(Dense(units=1))

                # Compile model
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x_train, y_train, epochs=25, batch_size=32)

                # Prepare test data
                past_100_days = data_training.tail(100)
                final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
                input_data = scaler.transform(np.array(final_df).reshape(-1, 1))

                x_test = []
                y_test = []

                for i in range(100, input_data.shape[0]):
                    x_test.append(input_data[i-100:i, 0])
                    y_test.append(input_data[i, 0])

                x_test, y_test = np.array(x_test), np.array(y_test)
                x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))  # Reshape for LSTM model

                y_predicted = model.predict(x_test)

                scale_factor = 1 / scaler.scale_[0]
                y_predicted = y_predicted * scale_factor
                y_test = y_test * scale_factor

                # Display the actual closing price on the previous day
                previous_day = end_date - timedelta(days=1)
                st.subheader(f'Actual Closing Price on {previous_day}')
                original_currency_closing_price = df['Close'].iloc[-1]
                st.write(f"The actual closing price on {previous_day} is: {original_currency_closing_price:.2f} {ticker.info['currency']}")

                # Display the last 30 days of data for the selected stock ticker
                st.subheader(f'Last 30 Days of Data for {user_input}')
                last_30_days_data = df.tail(30)[['Open', 'Close', 'High', 'Low', 'Volume']]
                last_30_days_data.index = last_30_days_data.index.date  # Extract only the date from the index
                st.write(last_30_days_data)

                # Predict for the next day
                x_next_day = np.array([input_data[-100:]])
                x_next_day = x_next_day.reshape((x_next_day.shape[0], x_next_day.shape[1], 1))  # Reshape for LSTM model
                y_next_day = model.predict(x_next_day)
                y_next_day = y_next_day * scale_factor

                # Convert predicted closing price to original currency format
                predicted_next_day_closing_price = y_next_day[0, 0]

                # Display the predicted closing price for the current day
                st.subheader(f'Predicted Closing Price for {end_date}')
                st.write(f"The predicted closing price for {end_date} is: {predicted_next_day_closing_price:.2f} {ticker.info['currency']}")

                # Final Graph
                st.subheader('Prediction Vs Original')
                fig2 = plt.figure(figsize=(12, 6))
                plt.plot(y_test, label='Original Price')
                plt.plot(y_predicted, label='Predicted Price')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.legend()
                st.pyplot(fig2)

            else:
                st.warning(f"No data available for the entered stock ticker ({user_input}). Please choose a valid stock ticker.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

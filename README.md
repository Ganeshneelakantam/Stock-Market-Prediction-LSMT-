# Stock Market Prediction(LSMT) App

This Streamlit app predicts the closing price of stocks based on historical data using LSTM (Long Short-Term Memory) neural networks. Users can select a stock exchange and a stock ticker to view the prediction.

## This is how app looks like
![image](https://github.com/Ganeshneelakantam/Stock-Market-Prediction-LSMT-/assets/133114368/f69bae66-cfd0-43e4-b2c7-5e3fa98ebc7f)


## Features

- Predicts the closing price of stocks for the next day based on historical data.
- Provides visualizations of historical closing prices and moving averages.
- Displays descriptive statistics and recent data for the selected stock.
- Supports selection of stock exchanges and custom stock tickers.
  
## Supported Stock Market Exchanges

- New York Stock Exchange
- Nasdaq
- Shanghai Stock Exchange
- Euronext
- Hong Kong Stock Exchange
- Tokyo Stock Exchange
- London Stock Exchange
- Toronto Stock Exchange
- Bombay Stock Exchange
  
## Usage

1. Clone the repository:
   ```
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

4. Select a stock exchange, choose a stock ticker, and specify the date range.
5. Click on the "Predict" button to view the prediction and analysis.

## Dependencies

- numpy
- pandas
- matplotlib
- yfinance
- keras
- streamlit
- scikit-learn

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

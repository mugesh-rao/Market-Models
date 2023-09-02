# algorithmic trading system with real-time buy/sell decisions involves several components, and it's a complex task

import yfinance as yf
import pandas as pd
import numpy as np

# Step 1: Data Collection
stock_symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2021-01-01"
df = yf.download(stock_symbol, start=start_date, end=end_date)

# Step 2: Data Preprocessing
df = df[['Adj Close']]
df.fillna(method='ffill', inplace=True)

# Step 3: Feature Engineering (None in this example)

# Step 4: Strategy Implementation (Moving Average Crossover)
# Define short-term and long-term moving average windows
short_window = 10
long_window = 50

# Calculate short-term and long-term moving averages
df['Short_MA'] = df['Adj Close'].rolling(window=short_window).mean()
df['Long_MA'] = df['Adj Close'].rolling(window=long_window).mean()

# Initialize positions (1 for long, -1 for short, 0 for neutral)
df['Position'] = 0

# Generate buy/sell signals based on moving average crossovers
df['Position'][short_window:] = np.where(df['Short_MA'][short_window:] > df['Long_MA'][short_window:], 1, -1)

# Calculate daily returns based on positions
df['Returns'] = df['Adj Close'].pct_change() * df['Position'].shift(1)

# Step 5: Evaluation (Calculate cumulative returns)
cumulative_returns = (1 + df['Returns']).cumprod()

# Step 6: Visualization (Cumulative Returns)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns, label="Strategy Returns")
plt.legend()
plt.title("Cumulative Returns")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.show()

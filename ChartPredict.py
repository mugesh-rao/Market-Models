import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Data Collection
stock_symbol = "TATAMOTORS.NS"  # NSE symbol for Tata Motors
start_date = "2020-01-01"
end_date = "2021-01-01"
df = yf.download(stock_symbol, start=start_date, end=end_date)

# Step 2: Data Preprocessing
df = df[['Adj Close']]
df.fillna(method='ffill', inplace=True)

# Step 3: Feature Engineering (None in this example)

# Step 4: Splitting the Data
split_percentage = 0.8
split_index = int(split_percentage * len(df))
train_data = df[:split_index]
test_data = df[split_index:]

# Step 5: Building the Model (Linear Regression)
model = LinearRegression()
X_train = np.arange(len(train_data)).reshape(-1, 1)
y_train = train_data.values
model.fit(X_train, y_train)
X_test = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)
y_test = test_data.values
y_pred = model.predict(X_test)

# Step 6: Evaluation and Visualization
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data.values, label="Training Data")
plt.plot(test_data.index, test_data.values, label="True Test Data")
plt.plot(test_data.index, y_pred, label="Predictions")
plt.legend()
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import yfinance as yf
# Define the stock symbol and the time period
symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-01-01'
# Fetch the stock data
data = yf.download(symbol, start=start_date, end=end_date)
# Save the data to a CSV file
data.to_csv('AAPL_stock_data.csv')
# Step 1: Data Collection
# For demonstration, we will use a CSV file containing historicalstock price data.
# Assuming the CSV file has columns 'Date', 'Open', 'High', 'Low','Close', 'Volume'
data = pd.read_csv('AAPL_stock_data.csv')
# Step 2: Data Preprocessing
# Convert 'Date' to datetime and set it as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
# Select features and target
features = data[['Open', 'High', 'Low', 'Volume']]
target = data['Close']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target,test_size=0.2, random_state=42)
# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Step 3: Model Training
# Create and train the SVM model
svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svm_model.fit(X_train_scaled, y_train)
# Step 4: Prediction
# Predict stock prices on the test set
y_pred = svm_model.predict(X_test_scaled)
# Step 5: Evaluation
# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# Plot the actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Actual Prices')
plt.plot(y_pred, label='Predicted Prices')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Actual vs Predicted Stock Prices')
plt.legend()
plt.show()
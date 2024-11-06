import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Define a date parser function
date_parser = lambda x: pd.to_datetime(x, format='%Y-%m-%d')  # Adjust the format as needed

# Load the data (replace with your actual path)
try:
    data = pd.read_csv('assignment_dataset.csv', parse_dates=['Date'], index_col='Date', date_parser=date_parser)
    #print("Data loaded successfully.")
except FileNotFoundError:
    #print("The file 'assignment_dataset.csv' was not found. Please check the path and try again.")
    raise

# Checking the data
#print(data.head())

# Ensure there are enough data points
if len(data) < 15:
    raise ValueError("Not enough data points. Ensure the dataset has at least 15 data points.")

# Split the data into training (10 days) and testing (5 days)
train_data = data['DNI'][:10]
test_data = data['DNI'][10:15]

# Check for NaNs in train_data
if train_data.isna().any():
    #print("Warning: Training data contains NaN values. Please check the dataset.")
    train_data = train_data.fillna(method='ffill')  # Forward fill NaNs

# Define the range for p, d, q values

# Define the range for p, d, q values
p_values = range(0, 5)  # Reduced to 0-2 due to small dataset
d_values = range(0, 2)
q_values = range(0, 5)

best_mse = float('inf')
best_order = None
best_forecast = None

# Grid search for the best p, d, q
for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                # Fit the ARIMA model
                model = ARIMA(train_data, order=(p, d, q))
                model_fit = model.fit()

                # Forecast the next 5 days
                forecast = model_fit.forecast(steps=len(test_data))
                forecast = pd.Series(forecast, index=test_data.index)

                # Check for NaNs in forecast
                if forecast.isna().any():
                    #print(f"Forecast produced NaNs for ARIMA(p={p}, d={d}, q={q}). Skipping this model.")
                    continue

                # Calculate the Mean Squared Error
                mse = mean_squared_error(test_data, forecast)
                rmse = np.sqrt(mse)

                # Update the best model if this one is better
                if mse < best_mse:
                    best_mse = mse
                    best_rmse = rmse
                    best_order = (p, d, q)
                    best_forecast = forecast

            except Exception as e:
                #print(f"Error with ARIMA(p={p}, d={d}, q={q}): {e}")
                continue

# Check if we found a valid model
if best_order is not None:
    # Plotting the results with the best model
    plt.figure(figsize=(10, 5))
    plt.plot(train_data, label='Training Data')
    plt.plot(test_data, label='Actual Data')
    plt.plot(best_forecast, label=f'Forecast (p,d,q={best_order})')
    plt.title(f'DNI Forecast using ARIMA (Order={best_order})')
    plt.xlabel('Date')
    plt.ylabel('DNI')
    plt.legend()
    plt.show()

    # Output the best MSE, RMSE, and the corresponding order
    #print(f'Best ARIMA Order: {best_order}')
    #print(f'Best Mean Squared Error (MSE): {best_mse}')
    #print(f'Best Root Mean Squared Error (RMSE): {best_rmse}')
else:
    print("No valid ARIMA model found. Consider expanding the training data or adjusting the parameter ranges.")
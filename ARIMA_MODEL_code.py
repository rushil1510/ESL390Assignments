import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Update the date_parser format to match 'day/month/year' (two-digit year)
date_format = '%d/%m/%y'  # Adjust the format as needed

# Load the data (replace with your actual path)
data = pd.read_csv('assignment_dataset.csv', parse_dates=['Day'], index_col='Day', date_format=date_format)

# Checking the data
print(data.head())

# Split the data into training (10 days) and testing (5 days)
train_data = data['DNI'][:10]
test_data = data['DNI'][10:15]

# Define the range for p, d, q values
p_values = range(0, 5)  
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
                continue

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
print(f'Best ARIMA Order: {best_order}')
print(f'Best Mean Squared Error (MSE): {best_mse}')
print(f'Best Root Mean Squared Error (RMSE): {best_rmse}')
'''
abcd
'''
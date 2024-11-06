# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv('/content/BTC-USD(1).csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Plot the Closing Price to inspect for trends
plt.figure(figsize=(10, 5))
plt.plot(data['Close'], label='Closing Price')
plt.title('Time Series of Bitcoin Closing Price')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Check stationarity with ADF test
result = adfuller(data['Close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# If p-value > 0.05, apply differencing
data['Close_diff'] = data['Close'].diff().dropna()
result_diff = adfuller(data['Close_diff'].dropna())
print('Differenced ADF Statistic:', result_diff[0])
print('Differenced p-value:', result_diff[1])

# Plot ACF and PACF for differenced data
plot_acf(data['Close_diff'].dropna())
plt.title('ACF of Differenced Closing Price')
plt.show()

plot_pacf(data['Close_diff'].dropna())
plt.title('PACF of Differenced Closing Price')
plt.show()

# Plot Differenced Representation
plt.figure(figsize=(10, 5))
plt.plot(data['Close_diff'], label='Differenced Closing Price', color='red')
plt.title('Differenced Representation of Closing Price')
plt.xlabel('Date')
plt.ylabel('Differenced Closing Price')
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.legend()
plt.show()

# Use auto_arima to find the optimal (p, d, q) parameters
stepwise_model = auto_arima(data['Close'], start_p=1, start_q=1,
                            max_p=3, max_q=3, seasonal=False, trace=True)
p, d, q = stepwise_model.order
print(stepwise_model.summary())

# Fit the ARIMA model using the optimal parameters
model = sm.tsa.ARIMA(data['Close'], order=(p, d, q))
fitted_model = model.fit()
print(fitted_model.summary())

# Forecast the next 30 days
forecast = fitted_model.forecast(steps=30)
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Actual Closing Price')
plt.plot(forecast_index, forecast, label='Forecast', color='orange')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('ARIMA Forecast of Bitcoin Closing Price')
plt.legend()
plt.show()

# Evaluate the model with MAE and RMSE
predictions = fitted_model.predict(start=0, end=len(data['Close']) - 1)
mae = mean_absolute_error(data['Close'], predictions)
rmse = np.sqrt(mean_squared_error(data['Close'], predictions))
print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)

```

### OUTPUT:

![download](https://github.com/user-attachments/assets/705470a0-0a76-4303-a6e8-bd99f186c96d)

![image](https://github.com/user-attachments/assets/d34b3c61-2fda-41d5-ad5d-cfac4a5c39d9)


### RESULT:
Thus the program run successfully based on the ARIMA model using python.

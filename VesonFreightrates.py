import pandas as pd

# Load the main Excel file and filter data from 2013 onwards
file_path_main = r"C:\Users\sadee\Downloads\PMX FFA  2005-2023.xlsx"
sheet_name_main = "Sheet1"  # Replace with your sheet name if necessary

data_main = pd.read_excel(file_path_main, sheet_name=sheet_name_main)
data_main["Date"] = pd.to_datetime(
    data_main["Date"]
)

# Convert 'Date' column to datetime
data_main = data_main[data_main["Date"] >= "2013-01-01"]
data_main.dropna(inplace=True)  # Drop rows with any NaN values

file_path_other = r"C:\Users\sadee\Downloads\Baltic dry data (1).xlsx"
sheet_name_other = "SIN Timeseries - D"
data_other = pd.read_excel(
    file_path_other, sheet_name=sheet_name_other, skiprows=5, usecols="A:B"
)
data_other.dropna(inplace=True)  # Drop rows with any NaN values

data_other["Date"] = pd.to_datetime(
    data_other["Date"]
)

# Convert 'Date' column to datetime
data_other = data_other[data_other["Date"] >= "2013-01-01"]

# Merge DataFrames on a common column (e.g., date)
merged_data = pd.merge(data_main, data_other, on="Date", how="left")
merged_data.dropna(inplace=True)

# Display the updated dataframe
print(merged_data.head())
print(merged_data.columns)
# -------Initial Graphs
import matplotlib.pyplot as plt

# First Plot
plt.plot(merged_data["Date"], merged_data["Index"])
plt.title("Baltic Dry Index over Time")
plt.xlabel("Date")
plt.ylabel("Index")
plt.gcf().autofmt_xdate()  # Improve date formatting on the x-axis
plt.savefig("Baltic_Dry_Index.png")

# Second plot
plt.figure()
plt.plot(merged_data["Date"], merged_data["4TC_P+1MON"])
plt.plot(merged_data["Date"], merged_data["4TC_P+1CAL"])
plt.plot(merged_data["Date"], merged_data["4TC_P+1Q"])
plt.title("Shorter term FFAs over time")
plt.xlabel("Date")
plt.ylabel("FFA price")
plt.gcf().autofmt_xdate()
plt.savefig("ExplanatoryVariables.png")
plt.show()

# -------We can see the lack of mean reversion, testing covariance stationarity test
from statsmodels.tsa.stattools import adfuller

for column in merged_data.columns:
    if column != "Date":
        series = merged_data[column]
        result = adfuller(series)
        print(f"Column: {column}")
        print("ADF Statistic:", result[0])
        print("p-value:", result[1])
        print("Critical Values:")
        for key, value in result[4].items():
            print(f"\t{key}: {value}")
        print("\n")
# all series are non stationary, so we must take the differences.

Data_New = pd.DataFrame()
Data_New["Date"] = merged_data["Date"]
# Loop through each column in data_main except for the 'Date' column
for column in merged_data.columns:
    if column != "Date":
        # Compute first difference (return series) and store in Data_New
        Data_New[column + "_Return"] = merged_data[column].diff()
# Display the updated dataframe
print(Data_New.head())

# Retry ADF tests:
Data_New.dropna(inplace=True)
for column in Data_New.columns:
    if column != "Date":
        series = Data_New[column]
        result = adfuller(series)
        print(f"Column: {column}")
        print("ADF Statistic:", result[0])
        print("p-value:", result[1])
        print("Critical Values:")
        for key, value in result[4].items():
            print(f"\t{key}: {value}")
        print("\n")


import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# Create the bdi_return DataFrame
bdi_return = Data_New[["Date", "Index_Return"]].dropna()

# Convert the 'Date' column to datetime format and set it as index
bdi_return["Date"] = pd.to_datetime(bdi_return["Date"], errors="coerce")
bdi_return.set_index("Date", inplace=True)

print(bdi_return)
# Split the data into in-sample and out-of-sample
train_size = int(len(bdi_return) * 0.9)  # Use 90% of the data for training
train, test = bdi_return[:train_size], bdi_return[train_size:]

# Fit the ARIMA model on the in-sample data
model = sm.tsa.ARIMA(train, order=(1, 0, 1))  # Example order (1, 0, 1)
model_fit = model.fit()

# Print the model summary
print(model_fit.summary())

# Forecast the out-of-sample period
forecast_steps = len(test)
forecast = model_fit.forecast(steps=forecast_steps)

# Plot the forecast versus actual out-of-sample data
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label="In-sample Data")
plt.plot(test.index, test, label="Actual Out-of-sample Data")
plt.plot(test.index, forecast, label="Forecast", color="red")
plt.xlabel("Date")
plt.gcf().autofmt_xdate()
plt.ylabel("Baltic Dry Index Return")
plt.legend()
plt.title("Baltic Dry Index Return Forecast vs Actual")
plt.savefig("ARIMA Forecast.png")
plt.show()


# Evaluate the forecast performance using Mean Squared Error
mse11 = mean_squared_error(test, forecast)
print(f"Mean Squared Error: {mse11}")

# Diagnostic tests on in-sample model
# Plot residuals
residuals = model_fit.resid
plt.figure(figsize=(10, 5))
plt.plot(residuals)
plt.title("Residuals")
plt.show()

# Plot ACF and PACF of residuals
fig, ax = plt.subplots(2, figsize=(12, 8))
sm.graphics.tsa.plot_acf(residuals, lags=40, ax=ax[0])
sm.graphics.tsa.plot_pacf(residuals, lags=40, ax=ax[1])
plt.show()

# Q-Q plot of residuals
sm.qqplot(residuals, line="s")
plt.title("Q-Q Plot")
plt.show()

# Histogram of residuals
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=30)
plt.title("Histogram of Residuals")
plt.show()

# Conduct Ljung-Box test
lb_test = sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True)
print(lb_test)

#----New VECM model
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
import numpy as np
from sklearn.preprocessing import StandardScaler

vecm_data = Data_New[["Date", "Index_Return", "4TC_P+1MON_Return"]].dropna()

#Converting date to datetime
vecm_data['Date'] = pd.to_datetime(vecm_data['Date'], errors='coerce')

#setting date as index
vecm_data.set_index('Date', inplace=True)

# Split data into training (90%) and test (10%)
train_size = int(len(vecm_data) * 0.9)
train_data = vecm_data.iloc[:train_size]
test_data = vecm_data.iloc[train_size:]

# Test for cointegration on training data
result = coint_johansen(train_data, det_order=0, k_ar_diff=1)
print(result.lr2)

# Fit VECM model on training data with lag order 1
model = VECM(train_data, k_ar_diff=1, coint_rank=1, deterministic='co', exog=None)  # exog=None means no exogenous variables
vecm_fit = model.fit()
print("\nVECM Model Summary:")
print(vecm_fit.summary())

# Static forecast: one-step-ahead forecasting
actuals = []
predictions = []

for i in range(len(test_data)):
    # Fit the model on the combined training and current actuals data
    combined_data = vecm_data.iloc[:train_size + i]
    model = VECM(combined_data, k_ar_diff=1, coint_rank=1, deterministic='co', exog=None)
    vecm_fit = model.fit()

    # Predict the next step (one step ahead)
    forecast = vecm_fit.predict(steps=1)
    predictions.append(forecast[0][0])
    actuals.append(test_data['Index_Return'].iloc[i])

# Create forecast DataFrame with appropriate dates
forecast_index = test_data.index[:len(predictions)]
forecast_df = pd.DataFrame({'Actual': actuals, 'Forecast': predictions}, index=forecast_index)

# Plot forecast vs actual for the test data
plt.figure(figsize=(12, 6))
plt.plot(forecast_df.index, forecast_df['Actual'], label='Actual Baltic Dry Index Return')
plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast Baltic Dry Index Return', color='red')
plt.xlabel('Date')
plt.ylabel('Baltic Dry Index Return')
plt.legend()
plt.title('Baltic Dry Index Return Static Forecast vs Actual (Test Data)')
plt.savefig("VECM Forecast.png")
plt.show()

# Diagnostic tests on the residuals
residuals = np.array(actuals) - np.array(predictions)
residuals = residuals.flatten()  # Ensure residuals are one-dimensional

plt.figure(figsize=(12, 6))
plt.plot(forecast_df.index, residuals)
plt.title('Residuals')
plt.show()

sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot')
plt.show()

plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=30)
plt.title('Histogram of Residuals')
plt.show()


sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot')
plt.show()

plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=30)
plt.title('Histogram of Residuals')
plt.show()

print(Data_New.tail(10))

msevecm = mean_squared_error(test_data['Index_Return'], forecast_df['Forecast'])
print(f'MSE of the VECM is: {msevecm}')
print(f'MSE of the ARIMA is: {mse11}')# Write your code here :-)

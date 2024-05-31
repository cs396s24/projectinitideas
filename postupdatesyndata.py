import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# Define the causal structure
def generate_synthetic_data(n_periods):
    np.random.seed(42)
    
    # Initialize time series data
    real_gdp = np.zeros(n_periods)
    yield_curve = np.zeros(n_periods)
    federal_funds_rate = np.zeros(n_periods)
    unemployment_rate = np.zeros(n_periods)
    
    # Define parameters for the structural equations
    beta_yc_to_gdp = 0.3
    beta_ffr_to_gdp = -0.2
    beta_ur_to_gdp = -0.1
    beta_ffr_to_yc = 0.4
    beta_ur_to_yc = -0.3
    
    # Generate synthetic data
    for t in range(1, n_periods):
        yield_curve[t] = (beta_ffr_to_yc * federal_funds_rate[t-1] +
                          beta_ur_to_yc * unemployment_rate[t-1] +
                          np.random.normal())
        
        real_gdp[t] = (beta_yc_to_gdp * yield_curve[t-1] +
                       beta_ffr_to_gdp * federal_funds_rate[t-1] +
                       beta_ur_to_gdp * unemployment_rate[t-1] +
                       np.random.normal())
        
        federal_funds_rate[t] = np.random.normal()
        unemployment_rate[t] = np.random.normal()
    
    dates = pd.date_range(start='1992-01-01', periods=n_periods, freq='M')
    synthetic_df = pd.DataFrame({
        'real_gdp': real_gdp,
        'yield_curve': yield_curve,
        'federal_funds_rate': federal_funds_rate,
        'unemployment_rate': unemployment_rate
    }, index=dates)
    
    return synthetic_df

# Generate synthetic data with a known causal structure
n_periods = 336  # From 1992-01-01 to 2020-01-31 (28 years)
synthetic_df = generate_synthetic_data(n_periods)

# Run ADF test for stationarity
adf_results = pd.DataFrame(columns=['p-value'])
for column in synthetic_df.columns:
    adf_result = adfuller(synthetic_df[column])
    adf_results.loc[column] = [adf_result[1]]

print("ADF Test Results:")
print(adf_results)

# Fit the VAR model
model = VAR(synthetic_df)
results = model.fit(maxlags=15, ic='aic')
print(results.summary())

# Forecasting
forecast_steps = 12  # Number of months to forecast
forecast = results.forecast(synthetic_df.values[-results.k_ar:], steps=forecast_steps)
forecast_df = pd.DataFrame(forecast, index=pd.date_range(start=synthetic_df.index[-1], periods=forecast_steps + 1, freq='M')[1:], columns=synthetic_df.columns)

# Combine original data and forecasted data
combined_df = pd.concat([synthetic_df, forecast_df])

# Plot the original and forecasted data
plt.figure(figsize=(12, 8))
for column in synthetic_df.columns:
    plt.plot(combined_df.index, combined_df[column], label=f'Forecast {column}')
    plt.axvline(x=synthetic_df.index[-1], color='r', linestyle='--')
plt.legend()
plt.show()

# Granger causality tests for each variable against real_gdp on synthetic data
max_lag = 12  # Define the maximum number of lags to test
granger_results_synthetic = {}
for col in synthetic_df.columns:
    if col != 'real_gdp':
        result = grangercausalitytests(synthetic_df[['real_gdp', col]], max_lag, verbose=False)
        p_values = [round(result[i+1][0]['ssr_ftest'][1], 4) for i in range(max_lag)]
        granger_results_synthetic[col] = p_values
        print(f'Granger causality test results for {col} causing real_gdp (synthetic): {p_values}')

        # Plotting Granger causality results
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_lag + 1), p_values, marker='o')
        plt.axhline(y=0.05, color='r', linestyle='--')
        plt.title(f'Granger Causality Test p-values\n{col} causing real_gdp (synthetic)')
        plt.xlabel('Lag')
        plt.ylabel('p-value')
        plt.xticks(range(1, max_lag + 1))
        plt.grid(True)
        plt.show()

# Function to perform Granger causality tests
def perform_granger_causality_tests(data, target='real_gdp', maxlag=12):
    causality_results = {}
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    for col in data.columns:
        if col != target:
            test_result = grangercausalitytests(data[[target, col]], maxlag=maxlag, verbose=False)
            causality_results[col] = [round(test_result[i+1][0]['ssr_ftest'][1], 4) for i in range(maxlag)]
    return causality_results

# Granger causality tests for the synthetic data
synthetic_causality_results = perform_granger_causality_tests(synthetic_df)

# Display the results
print("Granger Causality Test Results (Synthetic Data):")
for key, value in synthetic_causality_results.items():
    print(f"{key} causing real_gdp: {value}")

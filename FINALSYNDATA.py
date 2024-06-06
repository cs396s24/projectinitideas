# Jupyter Notebook: Time Series Causal Analysis with Synthetic Data and Real Data Preprocessing

# 1. Imports and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from sklearn.utils import resample
from fredapi import Fred
from copulas.multivariate import GaussianMultivariate
from ctgan import CTGAN

# Set random seed for reproducibility
np.random.seed(42)

# FRED API key
fred = Fred(api_key='2762cecd3d02ff4b96df9a6049d8daf4')

# Define the series IDs
series_ids = {
    'yield_curve': 'T10Y2Y',
    'real_gdp': 'external source',
    'real_pce': 'DPCERAM1M225NBEA',
    'unemployment_rate': 'UNRATE',
    'federal_funds_rate': 'FEDFUNDS',
    'sticky_cpi': 'CORESTICKM157SFRBATL',
    'us_policy_uncertainty': 'external source',
    'geopolitical_risk_index': 'external source',
    'cboe_volatility': 'VIXCLS',
    'business_confidence': 'BSCICP03USM665S',
    'consumer_confidence': 'CSCICP03USM665S',
}
series_freq = {
    'yield_curve': 'daily',
    'real_gdp': 'monthly',
    'real_pce': 'monthly',
    'unemployment_rate': 'monthly',
    'federal_funds_rate': 'monthly',
    'sticky_cpi': 'monthly',
    'us_policy_uncertainty': 'monthly',
    'geopolitical_risk_index': 'monthly',
    'cboe_volatility': 'daily',
    'business_confidence': 'monthly',
    'consumer_confidence': 'monthly',
}
# Define the observation start and end dates
start_date = '1990-01-01'
end_date = '2024-01-01'
# Create an empty DataFrame to store the data
df = pd.DataFrame()

# Load and preprocess the data
for series_name, series_id in series_ids.items():
    if series_name == 'real_gdp':
        file_path = "US-Monthly-GDP-History-Data.xlsx"
        data = pd.read_excel(file_path, sheet_name="Data")
        data.reset_index(inplace=True)
        data.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
        data["Date"] = pd.to_datetime(data["Date"], format='mixed')
        data.set_index("Date", inplace=True)
        data.index = data.index + pd.offsets.MonthEnd(0)
        data = data["Monthly Real GDP Index"]
    elif series_name == 'us_policy_uncertainty':
        file_path = "US_Policy_Uncertainty_Data.xlsx"
        data = pd.read_excel(file_path, sheet_name="Main Index")
        data = data.iloc[:-1]
        data["Date"] = pd.to_datetime(data[['Year', 'Month']].assign(day=1))
        data.set_index("Date", inplace=True)
        data.index = data.index + pd.offsets.MonthEnd(0)
        data = data["News_Based_Policy_Uncert_Index"]
    elif series_name == 'geopolitical_risk_index':
        file_path = "data_gpr_export.xls"
        data = pd.read_excel(file_path, sheet_name="Sheet1")
        data["Date"] = pd.to_datetime(data['month'], format="%m/%d/%Y")
        data.set_index("Date", inplace=True)
        data.index = data.index + pd.offsets.MonthEnd(0)
        data = data["GPR"]
    else:
        data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)

    if series_name == 'yield_curve':
        data = data.interpolate(method='linear')
        data = data.apply(lambda x: 1 if x <= 0 else 0)
        data = data.resample('M').max()
        conditions = [(data.shift(i).fillna(0) == 0) for i in range(1, 24 + 1)]
        combined_condition = pd.concat(conditions, axis=1).all(axis=1)
        positive_values_condition = (data == 1)
        final_condition = positive_values_condition & combined_condition
        indices = final_condition[final_condition].index
        df["YC_INV"] = data
        df["YC_INV"] = 0
        df.loc[df.index.isin(indices), "YC_INV"] = 1
        continue
    elif series_name == 'sticky_cpi':
        data.index = pd.to_datetime(data.index)
        data.index = data.index.to_period('M').to_timestamp('M') + pd.offsets.MonthEnd(0)
        df["CPI"] = 1 + data/100
    elif series_name == 'real_pce':
        data.index = pd.to_datetime(data.index)
        data.index = data.index.to_period('M').to_timestamp('M') + pd.offsets.MonthEnd(0)
        df["RPCE"] = 1 + data/100
    else:
        data.index = data.index + pd.offsets.MonthEnd(0)
        if series_freq[series_name] == "daily":
            data = data.interpolate(method='linear')
            data = data.resample('M').mean()
        data = 1 + data.pct_change()
        df[series_name] = data.ffill()

df = df[df.index >= '1992-02-01']

for column in df.columns:
    if column != "YC_INV":
        df[column] = np.log(df[column])

# Run ADF test for stationarity
adf_results = pd.DataFrame(columns=['p-value'])
for column in df.columns:
    adf_result = adfuller(df[column])
    adf_results.loc[column] = [adf_result[1]]

print(df.head())
print(adf_results)

# Fit the VAR model
model = VAR(df)
results = model.fit(maxlags=15, ic='aic')
print(results.summary())

# Forecasting
forecast_steps = 12  # Number of months to forecast
forecast = results.forecast(df.values[-results.k_ar:], steps=forecast_steps)
forecast_df = pd.DataFrame(forecast, index=pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='ME')[1:], columns=df.columns)

# Adding white noise to the forecasted values
noise = np.random.normal(0, forecast_df.std(), forecast_df.shape)
forecast_df_noisy = forecast_df + noise

# Combine original data and forecasted data
combined_df = pd.concat([df, forecast_df_noisy])

# Plot the original and forecasted data
plt.figure(figsize=(12, 8))
for column in df.columns:
    plt.plot(combined_df.index, combined_df[column], label=f'Forecast {column}')
    plt.axvline(x=df.index[-1], color='r', linestyle='--')
plt.legend()
plt.show()

# Granger causality tests for each variable against real_gdp on synthetic data
max_lag = 12  # Define the maximum number of lags to test
granger_results_synthetic = {}
for col in combined_df.columns:
    if col != 'real_gdp':
        result = grangercausalitytests(combined_df[['real_gdp', col]], max_lag, verbose=False)
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

# 2. Generating Synthetic Data
def generate_synthetic_data(n_points=200, lags=3, true_coeffs=None):  # Increased the number of points
    if true_coeffs is None:
        true_coeffs = {
            'business_confidence': 0.3,
            'consumer_confidence': 0.2,
            'cboe_volatility': -0.1,
        }
    df_synthetic = pd.DataFrame(np.random.normal(0, 1, (n_points, len(true_coeffs) + 1)),
                                columns=['real_gdp'] + list(true_coeffs.keys()))

    for i in range(lags, n_points):
        for var, coeff in true_coeffs.items():
            df_synthetic.at[i, 'real_gdp'] += coeff * df_synthetic.at[i-lags, var]
        df_synthetic.at[i, 'real_gdp'] += np.random.normal(0, 0.1)
    
    return df_synthetic

# Generate synthetic dataset
synthetic_data = generate_synthetic_data()

# 3. Performing Granger Causality Tests
def perform_granger_causality_tests(data, target='real_gdp', maxlag=12):
    causality_results = {}
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    for col in data.columns:
        if col != target:
            test_result = grangercausalitytests(data[[target, col]], maxlag=maxlag, verbose=False)
            causality_results[col] = [round(test_result[i+1][0]['ssr_ftest'][1], 4) for i in range(maxlag)]
    return causality_results

# Bootstrap Granger causality tests
def bootstrap_granger_causality(data, target='real_gdp', n_bootstrap=100, maxlag=12):
    bootstrap_results = {col: [] for col in data.columns if col != target}
    for _ in range(n_bootstrap):
        boot_data = resample(data)
        gc_results = perform_granger_causality_tests(boot_data)
        for col, p_values in gc_results.items():
            bootstrap_results[col].append(p_values)
    
    confidence_intervals = {}
    for col, values in bootstrap_results.items():
        values = np.array(values)
        ci_lower = np.percentile(values, 2.5, axis=0)
        ci_upper = np.percentile(values, 97.5, axis=0)
        confidence_intervals[col] = (ci_lower, ci_upper)
    
    return confidence_intervals

# Generate bootstrap confidence intervals
bootstrap_cis = bootstrap_granger_causality(synthetic_data)

# 4. Plotting Results
def plot_time_series(data):
    plt.figure(figsize=(14, 8))
    for col in data.columns:
        plt.plot(data.index, data[col], label=col)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Synthetic Economic Time Series Data')
    plt.legend()
    plt.show()

def plot_granger_results(granger_results):
    plt.figure(figsize=(14, 8))
    for var, p_values in granger_results.items():
        plt.plot(range(1, len(p_values) + 1), p_values, label=var)
    plt.xlabel('Number of Lags')
    plt.ylabel('p-value')
    plt.title('Granger Causality Test Results')
    plt.axhline(y=0.05, color='r', linestyle='--')
    plt.legend()
    plt.show()

# Plot the synthetic data
plot_time_series(synthetic_data)

# Perform Granger causality tests
granger_results = perform_granger_causality_tests(synthetic_data)

# Plot Granger causality test results
plot_granger_results(granger_results)

print(granger_results)

# 5. Interpreting Causal Relationships
print("Bootstrap Confidence Intervals (Synthetic Data):")
for key, (ci_lower, ci_upper) in bootstrap_cis.items():
    print(f"{key} causing real_gdp: CI Lower={ci_lower}, CI Upper={ci_upper}")

# Additional synthetic data generation with Gaussian Copula and CTGAN

# Fit the Gaussian Copula model
gc = GaussianMultivariate()
gc.fit(df)

# Generate synthetic data
synthetic_data_gc = gc.sample(num_rows=df.shape[0])

# Convert synthetic data to DataFrame
synthetic_df_gc = pd.DataFrame(synthetic_data_gc, columns=df.columns, index=df.index)

# Add Gaussian noise to the synthetic data
noise = np.random.normal(loc=0, scale=0.1, size=synthetic_df_gc.shape)
synthetic_df_gc_noisy = synthetic_df_gc + noise

# Fit the CTGAN model
ctgan = CTGAN(epochs=500)
ctgan.fit(df)

# Generate synthetic data
synthetic_data_gan = ctgan.sample(df.shape[0])

# Convert synthetic data to DataFrame
synthetic_df_gan = pd.DataFrame(synthetic_data_gan, columns=df.columns, index=df.index)

# Clean the synthetic data
synthetic_df_gc.replace([np.inf, -np.inf], np.nan, inplace=True)
synthetic_df_gc.dropna(inplace=True)

synthetic_df_gc_noisy.replace([np.inf, -np.inf], np.nan, inplace=True)
synthetic_df_gc_noisy.dropna(inplace=True)

synthetic_df_gan.replace([np.inf, -np.inf], np.nan, inplace=True)
synthetic_df_gan.dropna(inplace=True)

# Visualize the synthetic data vs true data
plt.figure(figsize=(12, 8))
for col in synthetic_df_gc.columns:
    plt.plot(df.index, df[col], label=f'{col} (True)')
    plt.plot(synthetic_df_gc.index, synthetic_df_gc[col], label=f'{col} (Gaussian Copula Synthetic)')
plt.legend()
plt.title('True and Gaussian Copula Synthetic Data')
plt.show()

# Visualize the noisy synthetic data vs true data
plt.figure(figsize=(12, 8))
for col in synthetic_df_gc_noisy.columns:
    plt.plot(df.index, df[col], label=f'{col} (True)')
    plt.plot(synthetic_df_gc_noisy.index, synthetic_df_gc_noisy[col], label=f'{col} (Noisy Gaussian Copula Synthetic)')
plt.legend()
plt.title('True and Noisy Gaussian Copula Synthetic Data')
plt.show()

# Visualize the GAN synthetic data vs true data
plt.figure(figsize=(12, 8))
for col in synthetic_df_gan.columns:
    plt.plot(df.index, df[col], label=f'{col} (True)')
    plt.plot(synthetic_df_gan.index, synthetic_df_gan[col], label=f'{col} (GAN Synthetic)')
plt.legend()
plt.title('True and GAN Synthetic Data')
plt.show()

# Perform Granger causality tests on the synthetic data with checks
def perform_granger_causality_tests_with_checks(data, target='real_gdp', maxlag=12):
    causality_results = {}
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    min_obs_required = maxlag + 1  # Minimum number of observations required
    
    if data.shape[0] < min_obs_required:
        raise ValueError("Insufficient observations for Granger causality test with max_lag = {}. Need at least {} observations.".format(maxlag, min_obs_required))
    
    for col in data.columns:
        if col != target:
            test_result = grangercausalitytests(data[[target, col]], maxlag=maxlag, verbose=False)
            causality_results[col] = [round(test_result[i+1][0]['ssr_ftest'][1], 4) for i in range(maxlag)]
    
    return causality_results

# Perform Granger causality tests on the synthetic data
max_lag = 6  # Adjusted maximum number of lags
gc_causality_results = perform_granger_causality_tests_with_checks(synthetic_df_gc, maxlag=max_lag)
gc_noisy_causality_results = perform_granger_causality_tests_with_checks(synthetic_df_gc_noisy, maxlag=max_lag)
gan_causality_results = perform_granger_causality_tests_with_checks(synthetic_df_gan, maxlag=max_lag)

# Display the results
print("Granger Causality Test Results (Gaussian Copula):")
for key, value in gc_causality_results.items():
    print(f"{key} causing real_gdp: {value}")

print("\nGranger Causality Test Results (Noisy Gaussian Copula):")
for key, value in gc_noisy_causality_results.items():
    print(f"{key} causing real_gdp: {value}")

print("\nGranger Causality Test Results (GAN):")
for key, value in gan_causality_results.items():
    print(f"{key} causing real_gdp: {value}")

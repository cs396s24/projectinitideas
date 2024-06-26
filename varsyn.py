import pandas as pd
import numpy as np
from fredapi import Fred
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import matplotlib.pyplot as plt
from copulas.multivariate import GaussianMultivariate
from ctgan import CTGAN

# FRED API key
fred = Fred(api_key='2762cecd3d02ff4b96df9a6049d8daf4')

start_date = '1992-01-01'
end_date = '2020-01-31'

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

def perform_granger_causality_tests(data, target='real_gdp', maxlag=12):
    causality_results = {}
    data = data.replace([np.inf, -np.inf], np.nan).dropna()  # Add this line
    for col in data.columns:
        if col != target:
            test_result = grangercausalitytests(data[[target, col]], maxlag=maxlag, verbose=False)
            causality_results[col] = [round(test_result[i+1][0]['ssr_ftest'][1], 4) for i in range(maxlag)]
    return causality_results


# Granger causality tests for Gaussian Copula synthetic data
gc_causality_results = perform_granger_causality_tests(synthetic_df_gc)
# Granger causality tests for Noisy Gaussian Copula synthetic data
gc_noisy_causality_results = perform_granger_causality_tests(synthetic_df_gc_noisy)
# Granger causality tests for GAN synthetic data
gan_causality_results = perform_granger_causality_tests(synthetic_df_gan)

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

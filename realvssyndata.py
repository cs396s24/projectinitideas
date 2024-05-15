import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt

# Assuming df is your preprocessed dataframe with the true data
# Fitting the VAR model
model = VAR(df)
results = model.fit(maxlags=15, ic='aic')

# Generate synthetic data
forecast_steps = 100  # Number of steps to forecast
forecast = results.forecast(y=df.values[-results.k_ar:], steps=forecast_steps)

# Add Gaussian noise to the forecasted data
noise = np.random.normal(loc=0, scale=np.std(forecast, axis=0), size=forecast.shape)
synthetic_data = forecast + noise

# Convert synthetic data to a DataFrame
forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='M')[1:]
synthetic_df = pd.DataFrame(synthetic_data, index=forecast_dates, columns=df.columns)

# Combine true and synthetic data for Granger causality testing
combined_df = pd.concat([df, synthetic_df])

# Granger causality tests on synthetic data
maxlag = 12
causality_results = {}

for col in combined_df.columns:
    if col != 'real_gdp':
        test_result = grangercausalitytests(combined_df[['real_gdp', col]], maxlag=maxlag, verbose=False)
        causality_results[col] = [round(test_result[i+1][0]['ssr_ftest'][1], 4) for i in range(maxlag)]

# Display Granger causality results
for key, value in causality_results.items():
    print(f"Granger causality test results for {key} causing real_gdp: {value}")

# Visualize the synthetic data vs true data
plt.figure(figsize=(12, 8))
for col in synthetic_df.columns:
    plt.plot(combined_df.index, combined_df[col], label=f'{col} (True and Synthetic)')
plt.legend()
plt.title('True and Synthetic Data')
plt.show()

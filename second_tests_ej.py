import pandas as pd
from fredapi import Fred
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt

# Initialize Fred API
fred = Fred(api_key='2762cecd3d02ff4b96df9a6049d8daf4')  # Replace with your actual FRED API key

# Series IDs for the 10-Year and 2-Year Treasury rates (daily data)
daily_yield_curve_10yr_id = 'DGS10'  # Daily 10-Year Treasury Constant Maturity Rate
daily_yield_curve_2yr_id = 'DGS2'    # Daily 2-Year Treasury Constant Maturity Rate

# Fetch the daily data
start_date = '1990-01-01'
end_date = '2023-01-01'
daily_yield_curve_10yr = fred.get_series(daily_yield_curve_10yr_id, observation_start=start_date, observation_end=end_date)
daily_yield_curve_2yr = fred.get_series(daily_yield_curve_2yr_id, observation_start=start_date, observation_end=end_date)

# Calculate the daily yield curve as the difference
yield_curve_daily = daily_yield_curve_10yr - daily_yield_curve_2yr

# Resample to monthly data by taking the mean of daily readings within each month
yield_curve_monthly = yield_curve_daily.resample('M').mean()

# Checkpoint: print to see a snippet of the resampled monthly yield curve data
print(yield_curve_monthly.head())


##########3 Combining GDP Data and recession period definitions
# Series ID for Quarterly GDP Growth Rate, converted to monthly by forward filling the data
gdp_growth_id = 'A191RL1Q225SBEA'
gdp_growth_quarterly = fred.get_series(gdp_growth_id, observation_start=start_date, observation_end=end_date)
gdp_growth_monthly = gdp_growth_quarterly.resample('M').ffill()  # Forward fill to convert quarterly to monthly data

# Prepare DataFrame
data = pd.DataFrame({
    'Yield_Curve': yield_curve_monthly,
    'GDP_Growth': gdp_growth_monthly
}).dropna()  # Dropping NA to ensure all data points are aligned

# Visualizing the data
fig, ax1 = plt.subplots(figsize=(14, 7))
ax1.set_xlabel('Date')
ax1.set_ylabel('Yield Curve', color='tab:red')
ax1.plot(data.index, data['Yield_Curve'], color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('GDP Growth', color='tab:blue')
ax2.plot(data.index, data['GDP_Growth'], color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title('Monthly Yield Curve vs. GDP Growth')
plt.show()

###############3 econometric modeling for causality testing

# Define yield curve inversions (yield curve less than 0 implies inversion)
data['Yield_Curve_Inverted'] = (data['Yield_Curve'] < 0).astype(int)

# Granger Causality Test to check if inversions predict negative GDP growth
# Negative GDP growth is a proxy for recession
data['Negative_GDP_Growth'] = (data['GDP_Growth'] < 0).astype(int)

# We test the causality with a lag of up to 12 months to cover the 6-12 month prediction window
print("Granger Causality Test Results (Yield Curve Inversions predicting Negative GDP Growth):")
gc_results = grangercausalitytests(data[['Negative_GDP_Growth', 'Yield_Curve_Inverted']], maxlag=12, verbose=True)


###########
print("now daily lags...")

import pandas as pd
from fredapi import Fred
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt

# Initialize Fred API
fred = Fred(api_key='2762cecd3d02ff4b96df9a6049d8daf4')  # Replace with your actual FRED API key

# Fetch daily data for 10-Year and 2-Year Treasury Constant Maturity Rates
daily_yield_curve_10yr = fred.get_series('DGS10', observation_start='1990-01-01', observation_end='2023-01-01')
daily_yield_curve_2yr = fred.get_series('DGS2', observation_start='1990-01-01', observation_end='2023-01-01')

# Calculate the daily yield curve
yield_curve_daily = daily_yield_curve_10yr - daily_yield_curve_2yr

# Using S&P 500 as a proxy for daily economic activity (change in index can reflect economic outlooks)
sp500 = fred.get_series('SP500', observation_start='1990-01-01', observation_end='2023-01-01')
sp500_daily_returns = sp500.pct_change()  # Daily returns as a proxy for economic activity

# Combine into a single DataFrame
data = pd.DataFrame({
    'Yield_Curve': yield_curve_daily,
    'SP500_Returns': sp500_daily_returns
}).dropna()  # Dropping NA for consistent data

# Visualizing the data
fig, ax1 = plt.subplots(figsize=(15, 8))

ax1.set_xlabel('Date')
ax1.set_ylabel('Yield Curve', color='tab:red')
ax1.plot(data.index, data['Yield_Curve'], color='tab:red', label='Daily Yield Curve')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('S&P 500 Daily Returns', color='tab:blue')
ax2.plot(data.index, data['SP500_Returns'], color='tab:blue', label='S&P 500 Daily Returns', alpha=0.5)
ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title('Daily Yield Curve and S&P 500 Returns')
plt.legend(loc='upper left')
plt.show()

# Defining a binary variable for negative S&P 500 returns (proxy for recessions)
data['Negative_Returns'] = (data['SP500_Returns'] < 0).astype(int)

# Granger Causality Test
print("Granger Causality Test Results (Yield Curve predicting Negative S&P 500 Returns):")
gc_results = grangercausalitytests(data[['Negative_Returns', 'Yield_Curve']], maxlag=30, verbose=True)  # Using 30 days lag

import pandas as pd
from fredapi import Fred
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt

fred = Fred(api_key='2762cecd3d02ff4b96df9a6049d8daf4')  # Replace 'your_api_key_here' with your actual FRED API key

# Define the series IDs from FRED
yield_curve_id = 'T10Y2Y'  # 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
gdp_growth_id = 'A191RL1Q225SBEA'  # Real GDP, 1 Decimal

# Fetch the data
start_date = '1990-01-01'
end_date = '2022-12-31'
yield_curve = fred.get_series(yield_curve_id, observation_start=start_date, observation_end=end_date)
gdp_growth = fred.get_series(gdp_growth_id, observation_start=start_date, observation_end=end_date)

# Combine into a single DataFrame
data = pd.DataFrame({
    'Yield_Curve': yield_curve,
    'GDP_Growth': gdp_growth
}).dropna()  # Dropping NA to align the data

fig, ax1 = plt.subplots(figsize=(14, 7))

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Yield Curve', color=color)
ax1.plot(data.index, data['Yield_Curve'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('GDP Growth', color=color)  
ax2.plot(data.index, data['GDP_Growth'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.title('Yield Curve and GDP Growth Over Time')
plt.show()

# Maxlag is the number of lags to test, we're using 4 quarters (assuming quarterly data)
# If data is annual, you might want to adjust or interpolate to a finer scale.
gc_test_result = grangercausalitytests(data[['GDP_Growth', 'Yield_Curve']], maxlag=4, verbose=True)

print("monthly tests now")

# Define the series IDs from FRED
yield_curve_id = 'T10Y2YM'  # Monthly data for 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
gdp_growth_id = 'GDP'  # This is quarterly, need to find a monthly equivalent, like Industrial Production: IPMANSICS

# Fetch the data
start_date = '1990-01-01'
end_date = '2023-01-01'  # Adjusted to current date for more data
yield_curve = fred.get_series(yield_curve_id, observation_start=start_date, observation_end=end_date)
industrial_prod = fred.get_series('IPMANSICS', observation_start=start_date, observation_end=end_date)  # Monthly industrial production

# Combine into a single DataFrame
data = pd.DataFrame({
    'Yield_Curve': yield_curve,
    'Industrial_Prod': industrial_prod
}).dropna()  # Dropping NA to align the data

fig, ax1 = plt.subplots(figsize=(14, 7))

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Yield Curve', color=color)
ax1.plot(data.index, data['Yield_Curve'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Industrial Production', color=color)
ax2.plot(data.index, data['Industrial_Prod'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Yield Curve and Industrial Production Over Time')
plt.show()

# Testing Granger Causality at different lags
maxlags = 12  # Test up to 12 months
results = grangercausalitytests(data[['Industrial_Prod', 'Yield_Curve']], maxlag=maxlags, verbose=True)

print("daily now")

# Define the series IDs from FRED
daily_yield_curve_id = 'DGS10'  # Daily data for the 10-Year Treasury Constant Maturity Rate
daily_short_term_rate_id = 'DGS2'  # Daily data for the 2-Year Treasury Constant Maturity Rate
sp500_id = 'SP500'  # S&P 500 index as a proxy for economic activity

# Fetch the data
start_date = '1990-01-01'
end_date = '2023-01-01'
daily_yield_curve = fred.get_series(daily_yield_curve_id, observation_start=start_date, observation_end=end_date)
daily_short_term_rate = fred.get_series(daily_short_term_rate_id, observation_start=start_date, observation_end=end_date)
sp500 = fred.get_series(sp500_id, observation_start=start_date, observation_end=end_date)

# Compute the yield curve as the difference between the long-term and short-term interest rates
yield_curve_daily = daily_yield_curve - daily_short_term_rate

# Combine into a single DataFrame
data_daily = pd.DataFrame({
    'Yield_Curve': yield_curve_daily,
    'SP500': sp500
}).dropna()  # Dropping NA to ensure alignment

fig, ax1 = plt.subplots(figsize=(15, 8))

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Yield Curve', color=color)
ax1.plot(data_daily.index, data_daily['Yield_Curve'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('S&P 500', color=color)
ax2.plot(data_daily.index, data_daily['SP500'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Daily Yield Curve and S&P 500 Index Over Time')
plt.show()

# Using 30 days as a maximum number of lags
maxlags = 30
results_daily = grangercausalitytests(data_daily[['SP500', 'Yield_Curve']], maxlag=maxlags, verbose=True)
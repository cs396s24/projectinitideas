import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import qqplot

# Set random seed for reproducibility
np.random.seed(42)

# Constants for data simulation
N = 1000
MEAN_FFR, STD_FFR = 2.5, 0.5  # Federal funds rate mean and std dev
MEAN_UR, STD_UR = 5.0, 1.0    # Unemployment rate mean and std dev
PROB_YCI = 0.2                # Probability of yield curve inversion

# Function to simulate economic data
def simulate_data(N):
    """Generate synthetic economic data with given parameters."""
    federal_funds_rate = np.random.normal(MEAN_FFR, STD_FFR, N)
    unemployment_rate = np.random.normal(MEAN_UR, STD_UR, N)
    yield_curve_inversion = np.random.binomial(1, PROB_YCI, N)

    # Mediators influenced by treatment and confounders
    VIX = 20 + 10 * yield_curve_inversion + 0.2 * federal_funds_rate + np.random.normal(0, 5, N)
    BCI = 95 - 5 * yield_curve_inversion + 0.1 * unemployment_rate + np.random.normal(0, 2, N)
    CCI = 95 - 3 * yield_curve_inversion + 0.1 * unemployment_rate + np.random.normal(0, 2, N)

    # Outcome variable with interactions
    GDP_growth = 3 - 1 * yield_curve_inversion + 0.5 * VIX - 0.05 * BCI + 0.05 * CCI + \
                 0.1 * yield_curve_inversion * VIX + np.random.normal(0, 0.5, N)

    return pd.DataFrame({
        'Federal Funds Rate': federal_funds_rate,
        'Unemployment Rate': unemployment_rate,
        'Yield Curve Inversion': yield_curve_inversion,
        'VIX': VIX,
        'BCI': BCI,
        'CCI': CCI,
        'GDP Growth': GDP_growth
    })

data = simulate_data(N)

# Calculate the correlation matrix
predictors = data[['Yield Curve Inversion', 'VIX', 'BCI', 'CCI']]
corr_matrix = predictors.corr()
print("Correlation Matrix:")
print(corr_matrix)

# Visualize the correlation matrix
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Predictors')
plt.show()

# Calculate Variance Inflation Factor (VIF) to identify multicollinearity
def calculate_vif(X):
    """Calculate VIF for each predictor in the DataFrame."""
    vif_data = pd.DataFrame()
    vif_data['variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Display VIF values
vif_results = calculate_vif(predictors)
print("VIF Results:")
print(vif_results)

# Addressing multicollinearity by combining highly correlated variables
data['Confidence Index'] = (data['BCI'] + data['CCI']) / 2

# Using robust regression to reduce the influence of outliers
X = data[['Yield Curve Inversion', 'VIX', 'Confidence Index']]
X = sm.add_constant(X)  # Adding a constant term for the intercept
y = data['GDP Growth']
robust_model = sm.RLM(y, X).fit()
print(robust_model.summary())

# Plotting the residuals of the robust model
plt.scatter(robust_model.fittedvalues, robust_model.resid)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot for Robust Model')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.show()

# QQ plot for residual normality check
qqplot(robust_model.resid, line='s')
plt.title('QQ Plot of Residuals')
plt.show()

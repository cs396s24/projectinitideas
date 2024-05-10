import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.decomposition import PCA

# Set seed for reproducibility
np.random.seed(42)

# Constants for the data
N = 1000
MEAN_FFR, STD_FFR = 2.5, 0.5  # Federal funds rate mean and std dev
MEAN_UR, STD_UR = 5.0, 1.0    # Unemployment rate mean and std dev
PROB_YCI = 0.2                # Probability of yield curve inversion

# Function to simulate data
def simulate_data(N):
    federal_funds_rate = np.random.normal(MEAN_FFR, STD_FFR, N)
    unemployment_rate = np.random.normal(MEAN_UR, STD_UR, N)
    yield_curve_inversion = np.random.binomial(1, PROB_YCI, N)
    VIX = 20 + 10 * yield_curve_inversion + 0.2 * federal_funds_rate + np.random.normal(0, 5, N)
    BCI = 95 - 5 * yield_curve_inversion + 0.1 * unemployment_rate + np.random.normal(0, 2, N)
    CCI = 95 - 3 * yield_curve_inversion + 0.1 * unemployment_rate + np.random.normal(0, 2, N)
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

# PCA for dimensionality reduction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data[['VIX', 'BCI', 'CCI']])
data_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Adding PCA components to data
data['PC1'], data['PC2'] = data_pca['PC1'], data_pca['PC2']

# Quantile regression
X = data[['Yield Curve Inversion', 'PC1', 'PC2']]
X = sm.add_constant(X)
y = data['GDP Growth']
quantiles = [0.25, 0.5, 0.75]

# Fit and summary of Quantile Regression at different quantiles
for qt in quantiles:
    mod = QuantReg(y, X).fit(q=qt)
    print(f'Results for {qt} quantile')
    print(mod.summary())

# Visualization of PCA results
sns.scatterplot(x='PC1', y='PC2', hue='Yield Curve Inversion', data=data)
plt.title('PCA Component Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

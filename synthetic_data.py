import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(42)  # for reproducibility

# Number of data points
N = 1000

# Simulating confounders
federal_funds_rate = np.random.normal(2.5, 0.5, N)  # mean = 2.5%, std = 0.5%
unemployment_rate = np.random.normal(5.0, 1.0, N)  # mean = 5%, std = 1%

# Treatment variable
yield_curve_inversion = np.random.binomial(1, 0.2, N)  # 20% chance of inversion

# Mediators influenced by both the treatment and confounders
VIX = 20 + 10 * yield_curve_inversion + 0.2 * federal_funds_rate + np.random.normal(0, 5, N)
BCI = 95 - 5 * yield_curve_inversion + 0.1 * unemployment_rate + np.random.normal(0, 2, N)
CCI = 95 - 3 * yield_curve_inversion + 0.1 * unemployment_rate + np.random.normal(0, 2, N)

# Outcome variable with interactions
GDP_growth = 3 - 1 * yield_curve_inversion + 0.5 * VIX - 0.05 * BCI + 0.05 * CCI + \
             0.1 * yield_curve_inversion * VIX + np.random.normal(0, 0.5, N)

# Create DataFrame
data = pd.DataFrame({
    'Federal Funds Rate': federal_funds_rate,
    'Unemployment Rate': unemployment_rate,
    'Yield Curve Inversion': yield_curve_inversion,
    'VIX': VIX,
    'BCI': BCI,
    'CCI': CCI,
    'GDP Growth': GDP_growth
})

print(data.head())


# Regression Analysis

X = data[['Yield Curve Inversion', 'VIX', 'BCI', 'CCI']]
X = sm.add_constant(X)  # adding a constant
y = data['GDP Growth']

model = sm.OLS(y, X).fit()
print(model.summary())


# Visualization

plt.figure(figsize=(10, 6))
for yc_inv in data['Yield Curve Inversion'].unique():
    subset = data[data['Yield Curve Inversion'] == yc_inv]
    plt.scatter(subset['VIX'], subset['GDP Growth'], alpha=0.5, label=f'Yield Curve Inversion = {yc_inv}')
plt.title('GDP Growth vs. VIX for different states of Yield Curve Inversion')
plt.xlabel('VIX')
plt.ylabel('GDP Growth')
plt.legend()
plt.show()


# More graphs for each predictor and GDP Growth

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Plotting each variable against GDP Growth
variables = ['Yield Curve Inversion', 'VIX', 'BCI', 'CCI']
for var, ax in zip(variables, axes.flatten()):
    ax.scatter(data[var], data['GDP Growth'], alpha=0.5)
    ax.set_title(f'GDP Growth vs. {var}')
    ax.set_xlabel(var)
    ax.set_ylabel('GDP Growth')

plt.tight_layout()
plt.show()

# Correlation Matrix

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame
predictors = data[['Yield Curve Inversion', 'VIX', 'BCI', 'CCI']]

# Calculate the correlation matrix
corr_matrix = predictors.corr()

print("Correlation Matrix:")
print(corr_matrix)
# Set up the matplotlib figure
plt.figure(figsize=(8, 6))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
            cbar_kws={'shrink': .8}, square=True, linewidths=.5)

plt.title('Correlation Matrix of Predictors')
plt.show()


# Variance Inflation Factor (VIF) for each variable

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Function to calculate VIF
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["variable"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

# Calculate VIF
predictors_vif = predictors.assign(const=1)  # adding a constant for VIF calculation
vif_results = calculate_vif(predictors_vif.drop('const', axis=1))
print(vif_results)



#findings: BCI and CCI: massive colinearity
#apply regularization techqniues:

data['Confidence Index'] = (data['BCI'] + data['CCI']) / 2


from sklearn.linear_model import Ridge

# Assuming predictors and 'GDP Growth' as the response variable
X = data[['Yield Curve Inversion', 'VIX', 'Confidence Index']]
y = data['GDP Growth']

# Ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

# Print coefficients
print("Coefficients:", ridge.coef_)

#advanced feature engineering PCA on correlated predictors
# use principal components as new features in reg model
# transforms the correlated variables into set of linearly uncorrelated vars

from sklearn.decomposition import PCA

# PCA to reduce dimensionality
pca = PCA(n_components=2)  # Adjust components based on explained variance
principal_components = pca.fit_transform(data[['VIX', 'BCI', 'CCI']])

# Create a DataFrame with principal components
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
new_data = pd.concat([data['Yield Curve Inversion'], principal_df], axis=1)

# Regression with principal components
X_new = new_data[['Yield Curve Inversion', 'PC1', 'PC2']]
model = sm.OLS(y, sm.add_constant(X_new)).fit()
print(model.summary())

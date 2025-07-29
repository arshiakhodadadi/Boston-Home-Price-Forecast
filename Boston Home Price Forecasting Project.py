import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.stats import skew
from sklearn.neighbors import LocalOutlierFactor
# Instead of RandomizedSearchCV you can use GridSearchCV not much difference
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

# Load data
df = pd.read_csv("BostonHousing.csv")

# Feature engineering: combine tax and rad
df["tax_per_rad"] = np.log1p(df["tax"] / df["rad"])
df.drop(["tax", "rad"], axis=1, inplace=True)

# Normalize skewed features
features = df.drop("medv", axis=1)
skew_vals = features.apply(skew).sort_values(ascending=False)
skewed_features = skew_vals[abs(skew_vals) > 0.75]

if df['chas'].nunique() <= 2:
    skewed_features = skewed_features.drop('chas', errors='ignore')

for feature in skewed_features.index:
    df[feature] = np.log1p(df[feature])

# Train-test split
X = df.drop('medv', axis=1).values
y = df['medv'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Remove outliers using LOF
lof = LocalOutlierFactor(n_neighbors=10)
X_sel = lof.fit_predict(X_train)
mask = X_sel != -1
X_train_lof, y_train_lof = X_train[mask], y_train[mask]

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500, 600],
    'max_depth': [10, 20, 30, 40, 50, 60, None],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2', None]
}

# Grid search with cross-validation
rf = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=70,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=2,
    random_state=42
)
random_search.fit(X_train_lof, y_train_lof)

# Best model
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"""
Best Parameters: {random_search.best_params_}
Best Cross-Validated R2 Score: {random_search.best_score_:.4f}

Test Set Performance:
MAE  = {mae:.4f}
RMSE = {rmse:.4f}
MAPE = {mape:.4f}
R2   = {r2:.4f}
""")

# Plot prediction vs true values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True Price')
plt.ylabel('Predicted Price')
plt.title('RandomForest Predictions vs True')

# Plot residuals
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, color='purple')
plt.xlabel('Residuals')
plt.title('Residual Distribution')

plt.tight_layout()
plt.show()
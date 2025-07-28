# Boston house price data set
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from numpy import mean, std, percentile
from pandas import DataFrame, read_csv
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

df = pd.read_csv("BostonHousing.csv")
# Check the database information
df.head(3)
df.columns.tolist()
df.isnull().sum()
df.describe()

# Converting a data frame to an array
data = df.values
X = data[:, :-1]
y = data[:, -1]

# Visualizing the relationship between attribute columns and the target column (house price)

for k ,_ in DataFrame(X).items():
    plt.scatter(DataFrame(X)[k],y)
    plt.xlabel(k)
    plt.ylabel('Target')
    plt.title(f'Diagram illustration :{k}')
    plt.show()

# Training-test data partitioning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f'Shape X_train :{X_train.shape} Shape y_train :{y_train.shape}')

# Removing outliers with the IQR method
Q1 = percentile(X_train, 25, axis=0)
Q3 = percentile(X_train, 75, axis=0)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

mask = ((X_train >= lower_bound) & (X_train <= upper_bound)).all(axis=1)  
X_trainq = X_train[mask]  
y_trainq = y_train[mask] 

print(f'Shape X_trainq :{X_trainq.shape} Sahpe y_trainq :{y_trainq.shape}')

# Outlier removal with the Three Sigma method

data_mean ,data_std = mean(X_train),std(X_train)
cut_off = data_std * 3

lower , upper = data_mean - cut_off ,data_mean + cut_off
mask = ((X_train >= lower) & (X_train <= upper)).all(axis=1)

X_train_ms = X_train[mask]
y_train_ms = y_train[mask]

print(f'Shape X_tarin_ms :{X_train_ms.shape} ,Shape y_train_ms{y_train_ms.shape}')
 
# Removing outliers using the LOF method
lof = LocalOutlierFactor(n_neighbors=10)
X_sel = lof.fit_predict(X_train)

mask = X_sel != -1

X_train_lof , y_train_lof = X_train[mask,:],y_train[mask]
print(f'Shape X_train_lof :{X_train.shape} Shape y_train_lof{y_train.shape}')

# Deleted data dictionary :IQR ,3Sigma and LOF
shape_dict = {'q':[X_trainq ,y_trainq],
              'ms':[X_train_ms,y_train_ms],
              'lof':[X_train_lof,y_train_lof]}
# Making models


def make_models(shape_dict, X_test, y_test):
    results = {}
    models = {}
    y_preds = {}

    model_defs = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'RandomForestRegressor': RandomForestRegressor()
    }

    for name, model in model_defs.items():
        models[name] = model
        results[name] = {}
        for split, (X_train, y_train) in shape_dict.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[name][split] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            }

            y_preds[(name, split)] = y_pred
            print(f"{name} | {split} -> MAE: {mae:.3f}, MSE: {mse:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.3f}, R2: {r2:.3f}")

    return results, models, y_preds

results, trained_models, y_preds = make_models(shape_dict, X_test, y_test)

# Visualizing the performance and relationship between features and the target column (house price)
model, split = 'Ridge', list(results['Ridge'].keys())[0]
y_pred = y_preds[(model, split)]
y_true = y_test  

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel('True Price'); plt.ylabel('Predicted Price')
plt.title(f'{model} Predictions vs True')

plt.subplot(1, 2, 2)
residuals = y_true - y_pred
sns.histplot(residuals, kde=True, color='purple')
plt.xlabel('Residual'); plt.title(f'{model} Residual Distribution')

plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('household_water_consumption.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['Day'] = range(1, len(df) + 1)

X = df[['Day']].values
y = df['Total_Liters'].values

split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("=" * 55)
print("  PREDICTIVE MODEL EVALUATION")
print("=" * 55)
print(f"Training samples : {len(X_train)} (Day 1-{split})")
print(f"Testing samples  : {len(X_test)}  (Day {split+1}-{len(df)})")
print()

def report(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{name:28} | MAE {mae:7.2f} L | RMSE {rmse:7.2f} L | MAPE {mape:6.2f}%")
    return mae, rmse, mape

results = {}

df['MA7'] = df['Total_Liters'].rolling(window=7).mean()
ma_test = df['MA7'].iloc[split:].values
valid = ~np.isnan(ma_test)
results['Moving Average (7-day)'] = report(
    "Moving Average (7-day)", y_test[valid], ma_test[valid])

lin = LinearRegression().fit(X_train, y_train)
y_lin_test = lin.predict(X_test)
results['Linear Regression'] = report("Linear Regression", y_test, y_lin_test)

pf = PolynomialFeatures(degree=2)
X_train_p = pf.fit_transform(X_train)
X_test_p  = pf.transform(X_test)
poly = LinearRegression().fit(X_train_p, y_train)
y_poly_test = poly.predict(X_test_p)
results['Polynomial Regression (degree 2)'] = report(
    "Polynomial Regression (degree 2)", y_test, y_poly_test)

print()
best = min(results, key=lambda k: results[k][1])
print(f"Lowest RMSE: {best}")
print()

print("=" * 55)
print("  COPY THESE VALUES INTO TABLE 6.5")
print("=" * 55)
for name, (mae, rmse, mape) in results.items():
    print(f'["{name}", "{mae:.2f}", "{rmse:.2f}", "{mape:.2f}"],')

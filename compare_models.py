import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('household_water_consumption.csv')
df['Date'] = pd.to_datetime(df['Date']); df = df.sort_values('Date').reset_index(drop=True)
df['Day'] = range(1, len(df)+1)
X, y = df[['Day']].values, df['Total_Liters'].values
split = int(len(df)*0.8)
Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

def metrics(yt, yp):
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mape = np.mean(np.abs((yt-yp)/yt))*100
    return mae, rmse, mape

# Moving average
ma = pd.Series(y).rolling(7).mean().values[split:]
valid = ~np.isnan(ma)
print("MA   :", [round(m,2) for m in metrics(yte[valid], ma[valid])])

# Linear
lin = LinearRegression().fit(Xtr, ytr)
print("Lin  :", [round(m,2) for m in metrics(yte, lin.predict(Xte))])

# Polynomial deg 2
pf = PolynomialFeatures(2)
poly = LinearRegression().fit(pf.fit_transform(Xtr), ytr)
print("Poly :", [round(m,2) for m in metrics(yte, poly.predict(pf.transform(Xte)))])
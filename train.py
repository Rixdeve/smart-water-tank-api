import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
df = pd.read_csv('household_water_consumption.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['Day'] = (df['Date'] - df['Date'].min()).dt.days + 1

X = df[['Day']].values
y = df['Total_Liters'].values

# Train polynomial regression degree 2
pf = PolynomialFeatures(degree=2)
X_poly = pf.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# Save locally (same sklearn version now)
joblib.dump(model, 'water_model.pkl')
joblib.dump(pf,    'water_poly_features.pkl')

print('✅ Model retrained and saved locally')
print(f'   sklearn version used: ', end='')
import sklearn; print(sklearn.__version__)
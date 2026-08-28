import joblib

model = joblib.load('water_model.pkl')
pf = joblib.load('water_poly_features.pkl')

print("=" * 50)
print("  POLYNOMIAL REGRESSION COEFFICIENTS")
print("=" * 50)
print(f"Polynomial degree: {pf.degree}")
print(f"Intercept (a):  {model.intercept_}")
print(f"Coefficients:   {model.coef_}")
print()

b = model.coef_[1] if len(model.coef_) > 1 else 0
c = model.coef_[2] if len(model.coef_) > 2 else 0

print("Copy these three values into the ESP32 firmware:")
print(f"const float PRED_INTERCEPT = {model.intercept_:.6f};")
print(f"const float PRED_COEF_B    = {b:.6f};")
print(f"const float PRED_COEF_C    = {c:.6f};")
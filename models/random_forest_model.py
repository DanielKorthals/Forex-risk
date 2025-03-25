import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings

# 🚫 Warnings onderdrukken
warnings.filterwarnings("ignore")

# 📂 Data inladen
data = pd.read_csv("datafiles/forex_rates.csv")
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')
data = data.ffill()

# 🎯 Doel: Voorspellen van EUR/USD koers
base_currency = "USD"
target_currency = "EUR"

# 🎲 Kies de data die we willen voorspellen
ts = data[target_currency]

# 🔧 Lag features genereren (vooruitkijken met 1 tot 30 dagen)
for lag in range(1, 31):
    data[f'lag_{lag}'] = ts.shift(lag)

# 🎲 Drop NaN's die door lags ontstaan
data = data.dropna()

# 📊 Features en Doelvariabele
X = data[[f'lag_{i}' for i in range(1, 31)]]
y = data[target_currency]

# 🧪 Train-test splitsing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 🌲 Random Forest Model trainen
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 🔮 Voorspellingen
y_pred = model.predict(X_test)

# 📊 Evaluatie
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# 📝 MLflow logging
mlflow.set_experiment("Forex Risk - Random Forest")
with mlflow.start_run(run_name="Random_Forest_Model"):
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "random_forest_model")
    print("✅ Model logged to MLflow")

# 📊 Visualisatie van Voorspellingen
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Werkelijke waarde")
plt.plot(y_pred, label="Voorspelde waarde", color="orange")
plt.title(f"Random Forest Voorspelling voor {target_currency}/{base_currency}")
plt.xlabel("Datum")
plt.ylabel(f"Wisselkoers ({target_currency}/{base_currency})")
plt.legend()
plt.savefig("visualizations/random_forest_forecast.png")
plt.show()

print("🔗 Voorspelling en model succesvol afgerond!")

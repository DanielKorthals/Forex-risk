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

# 🔧 Lag features genereren (1 tot 30 dagen vertraging)
for lag in range(1, 31):
    data[f'lag_{lag}'] = ts.shift(lag)

# 📊 Rolling statistieken toevoegen (7 en 14 dagen gemiddelde en standaarddeviatie)
data['rolling_mean_7'] = ts.rolling(window=7).mean().fillna(0)
data['rolling_std_7'] = ts.rolling(window=7).std().fillna(0)
data['rolling_mean_14'] = ts.rolling(window=14).mean().fillna(0)
data['rolling_std_14'] = ts.rolling(window=14).std().fillna(0)

# 💡 Verschillen (returns) toevoegen
data['diff_1'] = ts.diff().fillna(0)
data['diff_7'] = ts.diff(periods=7).fillna(0)

# 🎲 Drop NaN's die door lags en rolling ontstaan
data = data.dropna()

# 📊 Features en Doelvariabele
feature_cols = [f'lag_{i}' for i in range(1, 31)] + ['rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14', 'diff_1', 'diff_7']
X = data[feature_cols]
y = data[target_currency]

# 🧪 Train-test splitsing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 🌲 Random Forest Model met geoptimaliseerde hyperparameters
model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# 🔮 Voorspellingen
y_pred = model.predict(X_test)

# 📊 Evaluatie
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# 📝 MLflow logging
mlflow.set_experiment("Forex Risk - Random Forest Enhanced")
with mlflow.start_run(run_name="Random_Forest_Enhanced_Model"):
    mlflow.log_param("model_type", "Random Forest Enhanced")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 20)
    mlflow.log_param("lags", 30)
    mlflow.log_param("rolling_features", "mean_7, std_7, mean_14, std_14")
    mlflow.log_param("diff_features", "diff_1, diff_7")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "random_forest_enhanced_model")
    print("✅ Enhanced Model logged to MLflow")

# 📊 Visualisatie van Voorspellingen
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Werkelijke waarde")
plt.plot(y_pred, label="Voorspelde waarde", color="orange")
plt.title(f"Random Forest Enhanced Voorspelling voor {target_currency}/{base_currency}")
plt.xlabel("Datum")
plt.ylabel(f"Wisselkoers ({target_currency}/{base_currency})")
plt.legend()

# 💾 Grafiek opslaan
plt.savefig("visualizations/random_forest_enhanced_forecast.png")

plt.show()

print("🔗 Verbeterde voorspelling en model succesvol afgerond!")



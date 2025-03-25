import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
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
ts = data[target_currency]

# 🔧 Lag features genereren (1 tot 50 dagen vertraging)
for lag in range(1, 51):
    data[f'lag_{lag}'] = ts.shift(lag)

# 📊 Rolling statistieken toevoegen (7, 14, 30 en 60 dagen gemiddelde en standaarddeviatie)
data['rolling_mean_7'] = ts.rolling(window=7).mean().fillna(0)
data['rolling_std_7'] = ts.rolling(window=7).std().fillna(0)
data['rolling_mean_14'] = ts.rolling(window=14).mean().fillna(0)
data['rolling_std_14'] = ts.rolling(window=14).std().fillna(0)
data['rolling_mean_30'] = ts.rolling(window=30).mean().fillna(0)
data['rolling_std_30'] = ts.rolling(window=30).std().fillna(0)
data['rolling_mean_60'] = ts.rolling(window=60).mean().fillna(0)
data['rolling_std_60'] = ts.rolling(window=60).std().fillna(0)

# 💡 Verschillen (returns) toevoegen
data['diff_1'] = ts.diff().fillna(0)
data['diff_7'] = ts.diff(periods=7).fillna(0)
data['diff_30'] = ts.diff(periods=30).fillna(0)

# 🎲 Drop NaN's die door lags en rolling ontstaan
data = data.dropna()

# 📊 Features en Doelvariabele
feature_cols = [f'lag_{i}' for i in range(1, 51)] + [
    'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14',
    'rolling_mean_30', 'rolling_std_30', 'rolling_mean_60', 'rolling_std_60',
    'diff_1', 'diff_7', 'diff_30'
]
X = data[feature_cols]
y = data[target_currency]

# 🧪 Train-test splitsing met TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
train_index, test_index = list(tscv.split(X))[-1]
X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# 🌲 Geoptimaliseerd Random Forest Model
model = RandomForestRegressor(
    n_estimators=300,      # Meer bomen voor betere generalisatie
    max_depth=25,          # Grotere diepte om complexere patronen te vangen
    min_samples_split=5,   # Minimaliseer overfitting door minimum splits te verhogen
    min_samples_leaf=2,    # Zorg dat bladeren niet te klein zijn
    random_state=42,
    n_jobs=-1              # Gebruik meerdere cores voor snelheid
)
model.fit(X_train, y_train)

# 🔮 Voorspellingen
y_pred = model.predict(X_test)

# 📊 Evaluatie
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# 📝 MLflow logging
mlflow.set_experiment("Forex Risk - Random Forest Optimized")
with mlflow.start_run(run_name="Random_Forest_Optimized_Model"):
    mlflow.log_param("model_type", "Random Forest Optimized")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 25)
    mlflow.log_param("min_samples_split", 5)
    mlflow.log_param("min_samples_leaf", 2)
    mlflow.log_param("n_lags", 50)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "random_forest_optimized_model")
    print("✅ Optimized Model logged to MLflow")

# 📊 Visualisatie van Voorspellingen
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Werkelijke waarde")
plt.plot(y_pred, label="Voorspelde waarde", color="orange")
plt.title(f"Optimized Random Forest Voorspelling voor {target_currency}/{base_currency}")
plt.xlabel("Datum")
plt.ylabel(f"Wisselkoers ({target_currency}/{base_currency})")
plt.legend()

# 💾 Grafiek opslaan
plt.savefig("visualizations/random_forest_optimized_forecast.png")

plt.show()

print("🔗 Verbeterde voorspelling en model succesvol afgerond!")



import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
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

# Visualiseer de tijdreeks om stationariteit te controleren
plt.figure(figsize=(10, 5))
plt.plot(ts, label="Exchange Rate")
plt.title(f"Time Series Plot of {target_currency}/{base_currency}")
plt.xlabel("Date")
plt.ylabel(f"Exchange Rate ({target_currency}/{base_currency})")
plt.legend()
plt.show()

# 📈 Stationariteit controleren met de ADF-test
print("🔍 Stationariteit controleren met de ADF-test...")
result = adfuller(ts)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

if result[1] < 0.05:
    print("✅ De tijdreeks is stationair.")
else:
    print("🚨 De tijdreeks is NIET stationair. Differencing nodig!")
    # Indien niet stationair: differentiëren
    ts = ts.diff().dropna()
    print("🔄 Tijdreeks is gedifferentieerd om stationariteit te bereiken.")

# 💡 ARIMA Parameters (p, d, q) automatisch vinden
print("🔍 Finding optimal ARIMA parameters...")
arima_model = auto_arima(ts, start_p=1, start_q=1,
                         max_p=5, max_q=5, seasonal=False,
                         trace=True, error_action='ignore', suppress_warnings=True)

# Beste parameters
p, d, q = arima_model.order
print(f"✅ Best ARIMA parameters: p={p}, d={d}, q={q}")

# 💡 ARIMA model fitten met de gevonden parameters
model = ARIMA(ts, order=(p, d, q))
fitted_model = model.fit()

# 🔮 Voorspellingen maken
n_periods = 30  # Aantal dagen vooruit voorspellen
forecast = fitted_model.forecast(steps=n_periods)
forecast_dates = pd.date_range(start=data.index[-1], periods=n_periods+1, freq='D')[1:]

# 🎯 Bereken de MSE op de laatste 30 dagen
y_true = ts[-n_periods:]
y_pred = forecast[:n_periods]
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# 📝 MLflow logging
mlflow.set_experiment("Forex Risk - ARIMA")
with mlflow.start_run(run_name="ARIMA_Model"):
    mlflow.log_param("model_type", "ARIMA")
    mlflow.log_param("p", p)
    mlflow.log_param("d", d)
    mlflow.log_param("q", q)
    mlflow.log_param("n_periods", n_periods)
    mlflow.log_metric("mse", mse)
    mlflow.log_artifact("datafiles/forex_rates.csv")
    mlflow.sklearn.log_model(fitted_model, "arima_model")
    print("✅ Model logged to MLflow")

# 📊 Visualisatie van voorspellingen
plt.figure(figsize=(10, 5))
plt.plot(ts[-100:], label="Historical Data")
plt.plot(forecast_dates, forecast, label="Forecast", color="orange")
plt.title(f"ARIMA Model - Forecast for {target_currency}/{base_currency}")
plt.xlabel("Date")
plt.ylabel(f"Exchange Rate ({target_currency}/{base_currency})")
plt.legend()
plt.savefig("visualizations/ARIMA_forecast.png")
plt.show()

print("🔗 Voorspelling en model succesvol afgerond!")


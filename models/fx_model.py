import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn

# Simuleer Forex-data
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
exchange_rates = np.cumsum(np.random.randn(100)) + 100  # Simuleer wisselkoersen

df = pd.DataFrame({"date": dates, "exchange_rate": exchange_rates})

# Train een simpel regressiemodel
X = np.arange(len(df)).reshape(-1, 1)  # Dagen als feature
y = df["exchange_rate"]  # Doelvariabele

model = LinearRegression()
model.fit(X, y)

# MLflow setup (zorg ervoor dat dit overeenkomt met hoe we MLflow eerder instelden)
mlflow.set_tracking_uri("sqlite:///mlruns.db")  
mlflow.set_experiment("Forex Risk")

# Log model in MLflow
with mlflow.start_run():
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("coef", model.coef_[0])
    mlflow.sklearn.log_model(model, "model")

print("✅ Model getraind en gelogd in MLflow!")

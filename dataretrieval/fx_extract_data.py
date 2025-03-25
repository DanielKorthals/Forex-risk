import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from datetime import datetime

# 📂 Load the data
data = pd.read_csv("datafiles/forex_rates.csv")

# 🧹 Preprocess the data
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')
data = data.ffill()

# 🎯 Use EUR as base currency and USD as target
base_currency = "USD"
target_currency = "EUR"

# 💡 Feature Engineering: Create time-based features
data['day_of_week'] = data.index.dayofweek
data['day_of_month'] = data.index.day
data['month'] = data.index.month
data['day_of_year'] = data.index.dayofyear

# 🎯 Create features (X) and target (y)
X = np.arange(len(data)).reshape(-1, 1)  # Using time steps as features
y = data[target_currency].values  # Targeting USD/EUR conversion rate

# 📊 Train/Test Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 💡 Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 📊 Make predictions
y_pred = model.predict(X_test)

# 📝 Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 📝 Log with MLflow
mlflow.set_experiment("Forex Risk - Linear Regression")
with mlflow.start_run(run_name="Linear_Regression_Model"):
    mlflow.log_param("model_type", "Linear Regression")
    mlflow.log_param("base_currency", base_currency)
    mlflow.log_param("target_currency", target_currency)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "linear_regression_model")
    print("✅ Model logged to MLflow")

# 📊 Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(data.index[train_size:], y_test, label="Actual Values", color="blue")
plt.plot(data.index[train_size:], y_pred, label="Predicted Values", color="orange")
plt.title(f"Linear Regression Model - {target_currency}/{base_currency}")
plt.xlabel("Date")
plt.ylabel(f"Exchange Rate ({target_currency}/{base_currency})")
plt.legend()

# 💾 Save the plot
plt.savefig("visualizations/linear_regression_forecast.png")
plt.show()

print("✅ Model training, evaluation, and visualization completed!")


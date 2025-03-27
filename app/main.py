from fastapi import FastAPI
import pandas as pd
import mlflow
from pydantic import BaseModel

# 🎉 Start de FastAPI-app
app = FastAPI()

# 📝 Model laden vanuit MLflow (Enhanced Random Forest)
model = mlflow.pyfunc.load_model("models:/random_forest_enhanced_model/1")

# 📦 Input data model
class ForexInput(BaseModel):
    lag_1: float
    lag_7: float
    lag_14: float
    lag_30: float
    rolling_mean_7: float
    rolling_std_7: float
    rolling_mean_14: float
    rolling_std_14: float
    rolling_mean_30: float
    rolling_std_30: float
    rolling_mean_60: float
    rolling_std_60: float
    diff_1: float
    diff_7: float
    diff_30: float

# 🎯 Endpoint voor voorspelling met Enhanced Random Forest
@app.post("/predict")
async def predict(data: ForexInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {"model": "Enhanced Random Forest", "prediction": prediction[0]}

# 🌐 Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Forex Risk Prediction API - Enhanced Random Forest"}

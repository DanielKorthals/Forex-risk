from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel

# 🎉 Start de FastAPI-app
app = FastAPI()

# 💾 Model laden vanuit het opgeslagen .pkl-bestand
model = joblib.load("models/enhanced_random_forest.pkl")

# 📦 Input data model (verwacht gestructureerde input)
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

# 🌐 Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Forex Risk Prediction API - Enhanced Random Forest"}

# 🎯 Voorspellingsendpoint
@app.post("/predict")
async def predict(data: ForexInput):
    # 🎯 Zet input om naar een pandas DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # 🔮 Maak een voorspelling
    prediction = model.predict(input_df)
    
    # 💡 Response met voorspelling
    return {
        "model": "Enhanced Random Forest",
        "prediction": prediction[0]
    }

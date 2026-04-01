from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from typing import List
import numpy as np
import uvicorn

app = FastAPI()

# Load the trained model
with open('output/model/trained_model.pkl', 'rb') as f:
    model = joblib.load(f)

with open('output/model/scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)

class Features(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    name: str
    roll_number: str
    prediction: int
    probability: float
    
@app.post("/predict", response_model=PredictionResponse)
def predict(features: Features) -> PredictionResponse:
    """
    Endpoint to get classification prediction based on input features
    """
    # Convert features to numpy array
    feature_array = np.array([features.features])
    feature_array = scaler.transform(feature_array)
    # Make prediction
    prediction = model.predict(feature_array)[0]
    probability = model.predict_proba(feature_array)[0].max()
    
    return PredictionResponse(
        name = "nivitha noble",
        roll_number = "2022bcs0008",
        prediction=int(prediction),
        probability=float(probability)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
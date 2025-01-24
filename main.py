# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the trained model
model = joblib.load("decision_tree_model.joblib")

# Define a request body model
class FeatureData(BaseModel):
    feature1: float
    feature2: float
    feature3: float

# Define an endpoint for making predictions
@app.post("/predict")
async def predict(data: FeatureData):
    try:
        input_data = np.array([[data.feature1, data.feature2, data.feature3]])
        prediction = model.predict(input_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Add this root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Decision Tree Classifier API!"}

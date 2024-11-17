from fastapi import FastAPI
import pandas as pd
from preprocessing import preprocess_data
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    data: dict

@app.post("/predict")
def predict(request: PredictionRequest):
    data = pd.DataFrame([request.data])
    X, _ = preprocess_data(data, target_column="label_column")
    probabilities = model.predict_proba(X)
    return {"predictions": probabilities.tolist()}
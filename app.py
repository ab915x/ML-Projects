from fastapi import Body, Depends, FastAPI
import uvicorn
from pydantic import BaseModel
from typing import Annotated, List
import joblib
import requests
import os

from create_model import train_model
from data_processing import extract_features_for_training, extract_features_for_inference
from data_tests import test_and_report_inference_data
from utils import download_data

class PredictRequest(BaseModel):
    passwords: List[str]

class PredictResponse(BaseModel):
    prediction: List[float]

model_path = "models:/anton-belousov-fyb5457-mlops-project-model@prod"

model = mlflow.pyfunc.load_model(model_path)

app = FastAPI()

model_ = None


def get_model():
    global model_
    if model_ is None:
        model_ = joblib.load(
            model_path
                             )
    return model_


@app.post("/predict")
def predict(request: PredictRequest, model=Depends(get_model)) -> PredictResponse:
    prediction = model.predict(request.passwords)  
    return PredictResponse(prediction=prediction.tolist()) 

@app.post("/trigger_retrain")
def retrain_model(url: str):
    data = pd.read_csv(download_data)
    data_quality_flag = True
    features = extract_features_for_training(data)
    if os.path.exists('reference_data.csv'):
        data_quality_flag = test_and_report_inference_data(features)
        
    if data_quality_flag:
        new_model = train_model(features)
    else:
        pass


def main():
    uvicorn.run(app, host="0.0.0.0")


if __name__ == '__main__':
    main()
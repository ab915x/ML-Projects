from fastapi import Body, Depends, FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import Annotated, List
import os
import mlflow
from create_model import train_model
from data_processing import extract_features_for_training, extract_features_for_inference
from data_tests import test_and_report_inference_data
from utils import download_data
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    passwords: List[str]

class PredictResponse(BaseModel):
    predictions: List[float]  

MODEL_PATH = "models:/anton-belousov-fyb5457-mlops-project-model@prod"
app = FastAPI()
model_ = None


def get_model():
    global model_
    try:
        logger.info("Loading model from MLflow")
        model_ = mlflow.pyfunc.load_model(MODEL_PATH)
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Model loading failed: {str(e)}"
        )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        input = pd.DataFrame({"Password": request.passwords})
        features = extract_features_for_inference(input) 
        predictions = model_.predict(features)
        return PredictResponse(predictions=predictions.tolist())
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/trigger_retrain")
def retrain_model(url: str):
    data_path = download_data(url)
    data = pd.read_csv(data_path)
    data_quality_flag = True
    features = extract_features_for_training(data)
    if os.path.exists('reference_data.csv'):
        data_quality_flag = test_and_report_inference_data(features)
        
    if data_quality_flag:
        train_model(features)
        get_model()
    else:
        pass


def main():
    uvicorn.run(app, host="0.0.0.0")


if __name__ == '__main__':
    main()
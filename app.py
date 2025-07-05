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
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_NAME = "anton-belousov-mlops-project-model"
MODEL_ALIAS = "prod"
MODEL_PATH = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

class PredictRequest(BaseModel):
    passwords: List[str]

class PredictResponse(BaseModel):
    predictions: List[float]

class TriggerRequest(BaseModel):
    data_url: str

app = FastAPI()

model_ = None
is_training = False
training_lock = threading.Lock()
global last_trained 
last_trained = None


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
    global is_training

    if is_training:
        return {
            "status": "rejected",
            "message": "Training already in progress",
            "success": False
        }

    is_training = True
    
    data_path = download_data(url)
    data = pd.read_csv(data_path)
    data_quality_flag = True
    features = extract_features_for_training(data)
    if os.path.exists('reference_data.csv'):
        data_quality_flag = test_and_report_inference_data(features)
        
    if data_quality_flag:
        train_model(features)
        get_model()
        last_trained = pd.Timestamp.now().isoformat()
        is_training = False
    else:
        pass

@app.get("/status")
def get_status():
    return {
        "model_loaded": model_ is not None,
        "is_training": is_training,
        "last_trained": last_trained
    }


def main():
    uvicorn.run(app, host="0.0.0.0")


if __name__ == '__main__':
    main()
from fastapi import Body, Depends, FastAPI
import uvicorn
from pydantic import BaseModel
from typing import Annotated
import joblib
import os

class PredictRequest(BaseModel):
    passwords: list[str]

class PredictResponse(BaseModel):
    prediction: list[float]

model_path = os.getenv("MODEL_PATH", "pipeline.joblib")

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




def main():
    uvicorn.run(app, host="0.0.0.0")


if __name__ == '__main__':
    main()
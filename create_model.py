from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
# Это файл для обучения модели и сохранения ее в корне

def train_model(train_data: pd.DataFrame):
    model = CatBoostRegressor(eval_metric="AUC")
    y = train_data["target"]
    X = train_data.drop(columns=["target"])
    X_train, y_train, X_val, y_val = train_test_split(X, y, test_size=0.15)
    train_pool = Pool(X_train, y_train)
    eval_pool = Pool(X_val, y_val)
    model.fit(train_data=train_pool, eval=eval_pool)
    joblib.dump('model.joblib')
    return True

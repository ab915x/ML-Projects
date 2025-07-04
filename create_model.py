from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
# Это файл для обучения модели и сохранения ее в корне

def train_model(train_data: pd.DataFrame):
    y = train_data["target"]
    X = train_data.drop(columns=["target"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    train_pool = Pool(X_train, y_train)
    eval_pool = Pool(X_val, y_val)
    with mlflow.start_run():
        model = CatBoostRegressor(eval_metric="AUC")
        model.fit(train_data=train_pool, eval=eval_pool)
        mlflow.log_params(model.get_params())
        mlflow.log_dict({"columns": list(train_data.columns)}, "train_data_schema.json")
        stats = data.describe().to_dict()
        mlflow.log_dict(stats, "dataset_stats.json")
        model_name = f"anton-belousov-fyb5457-mlops-project-model"
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
        client.set_registered_model_alias(model_name, "prod", latest_version)
    return True


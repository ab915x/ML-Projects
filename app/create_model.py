from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
# Это файл для обучения модели и сохранения ее в корне

MODEL_NAME = "anton-belousov-mlops-project-model"


def train_model(train_data: pd.DataFrame):
    mlflow.set_experiment("my-mlops-project")
    y = train_data["target"]
    X = train_data.drop(columns=["target"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    train_pool = Pool(X_train, y_train)
    eval_pool = Pool(X_val, y_val)
    with mlflow.start_run():
        model = CatBoostClassifier(eval_metric="AUC", early_stopping_rounds=50)
        model.fit(train_pool, eval_set=eval_pool)

        best_metric = model.best_score_["validation"]["AUC"]
        mlflow.log_metric("Best AUC", best_metric)

        eval_history = model.evals_result_["validation"]["AUC"]
        for i, metric_value in enumerate(eval_history):
            mlflow.log_metric("AUC", metric_value, step=i)

        final_train_metric = model.evals_result_["learn"]["AUC"][-1]
        final_val_metric = eval_history[-1]
        mlflow.log_metrics(
            {"final_train_auc": final_train_metric, "final_val_auc": final_val_metric}
        )

        mlflow.log_params(model.get_params())
        mlflow.log_dict({"columns": list(train_data.columns)}, "train_data_schema.json")
        stats = train_data.describe().to_dict()
        mlflow.log_dict(stats, "dataset_stats.json")

        model_name = MODEL_NAME
        client = mlflow.tracking.MlflowClient()
        try:
            mlflow.catboost.log_model(model, "model", registered_model_name=model_name)
        except Exception:
            mlflow.catboost.log_model(model, "model")
            client.create_registered_model(model_name)
        latest_version = client.search_model_versions(f"name='{model_name}'")[0].version
        client.set_registered_model_alias(model_name, "prod", latest_version)

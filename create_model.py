from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
# Это файл для обучения модели и сохранения ее в корне

def train_model(train_data: pd.DataFrame):
    mlflow.set_experiment("my-mlops-project")
    y = train_data["target"]
    X = train_data.drop(columns=["target"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    train_pool = Pool(X_train, y_train)
    eval_pool = Pool(X_val, y_val)
    with mlflow.start_run():
        model = CatBoostRegressor(
            eval_metric="RMSE",
            early_stopping_rounds=50

            )
        model.fit(train_pool, eval_set=eval_pool)

        best_metric = model.best_score_['validation']['RMSE']
        mlflow.log_metric("best_rmse", best_metric)

        eval_history = model.evals_result_['validation']['RMSE']
        for i, metric_value in enumerate(eval_history):
            mlflow.log_metric("rmse", metric_value, step=i)

        final_train_metric = model.evals_result_['learn']['RMSE'][-1]
        final_val_metric = eval_history[-1]
        mlflow.log_metrics({
            "final_train_rmse": final_train_metric,
            "final_val_rmse": final_val_metric
        })

        mlflow.log_params(model.get_params())
        mlflow.log_dict({"columns": list(train_data.columns)}, "train_data_schema.json")
        stats = train_data.describe().to_dict()
        mlflow.log_dict(stats, "dataset_stats.json")

        model_name = f"anton-belousov-fyb5457-mlops-project-model"
        client = mlflow.tracking.MlflowClient()
        try:
            model_info = mlflow.catboost.log_model(model, "model", registered_model_name=model_name)
        except Exception as e:
            mlflow.catboost.log_model(model, "model")
            client.create_registered_model(model_name)
            model_info = mlflow.catboost.log_model(model, "model", registered_model_name=model_name)
        latest_version = client.search_model_versions(f"name='{model_name}'")[0].version
        client.set_registered_model_alias(model_name, "prod", latest_version)


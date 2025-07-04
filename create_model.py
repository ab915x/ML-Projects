from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import pandas as pd

# Это файл для обучения модели и сохранения ее в корне

def extract_features_for_training(data: list) -> pd.DataFrame:
    train_data = {
    "length": [len(password) for password in data['Password']],
    "num_uppercase": [sum(1 for c in password if c.isupper()) for password in data['Password']],
    "num_lowercase": [sum(1 for c in password if c.islower()) for password in data['Password']],
    "num_digits": [sum(1 for c in password if c.isdigit()) for password in data['Password']],
    "num_special": [sum(1 for c in password if not c.isalnum()) for password in data['Password']],
    "unique_chars": [len(set(password)) for password in data['Password']],
    "entropy": [len(password) * np.log2(len(set(password))) if password else 0 for password in data['Password']],
    "target": data["Times"]
    }
    return pd.DataFrame(train_data)

def extract_features_for_inference(data: list) -> pd.DataFrame:
    inference_data = {
    "length": [len(password) for password in data],
    "num_uppercase": [sum(1 for c in password if c.isupper()) for password in data],
    "num_lowercase": [sum(1 for c in password if c.islower()) for password in data],
    "num_digits": [sum(1 for c in password if c.isdigit()) for password in data],
    "num_special": [sum(1 for c in password if not c.isalnum()) for password in data],
    "unique_chars": [len(set(password)) for password in data],
    "entropy": [len(password) * np.log2(len(set(password))) if password else 0 for password in data]
    }
    return pd.DataFrame(inference_data)


def train_model(train_data: pd.DataFrame):
    model = CatBoostRegressor(eval_metric="AUC")
    y = train_data["target"]
    X = train_data.drop(columns=["target"])
    X_train, y_train, X_val, y_val = train_test_split(X, y, test_size=0.15)
    train_pool = Pool(X_train, y_train)
    eval_pool = Pool(X_val, y_val)
    model.fit(train_data=train_pool, eval=eval_pool)
    return model

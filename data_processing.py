import pandas as pd
import numpy as np


def extract_features_for_training(data: list) -> pd.DataFrame:
    train_data = pd.DataFrame(
        {
    "length": [len(password) for password in data['Password']],
    "num_uppercase": [sum(1 for c in password if c.isupper()) for password in data['Password']],
    "num_lowercase": [sum(1 for c in password if c.islower()) for password in data['Password']],
    "num_digits": [sum(1 for c in password if c.isdigit()) for password in data['Password']],
    "num_special": [sum(1 for c in password if not c.isalnum()) for password in data['Password']],
    "unique_chars": [len(set(password)) for password in data['Password']],
    "entropy": [len(password) * np.log2(len(set(password))) if password else 0 for password in data['Password']],
    "target": data["Times"]
    }
    )
    train_data.to_csv("reference_data.csv", index=False)
    return train_data

def extract_features_for_inference(data: list) -> pd.DataFrame:
    inference_data = pd.DataFrame({
    "length": [len(password) for password in data],
    "num_uppercase": [sum(1 for c in password if c.isupper()) for password in data],
    "num_lowercase": [sum(1 for c in password if c.islower()) for password in data],
    "num_digits": [sum(1 for c in password if c.isdigit()) for password in data],
    "num_special": [sum(1 for c in password if not c.isalnum()) for password in data],
    "unique_chars": [len(set(password)) for password in data],
    "entropy": [len(password) * np.log2(len(set(password))) if password else 0 for password in data]
    }
    )
    return inference_data


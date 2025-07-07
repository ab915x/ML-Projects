import os
from fastapi import HTTPException
import requests


def download_data(url: str):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        file_path = "data/new_data.csv"

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if os.path.exists(file_path):
            os.remove(file_path)

        with open(file_path, "wb") as f:
            f.write(response.content)

        return file_path

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Data download failed: {e}")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"File operation failed: {e}")

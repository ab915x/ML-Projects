from python:3.13.5-bullseye

WORKDIR /app

COPY app.py . 
COPY create_model.py .
COPY data_tests.py .
COPY data_processing.py .
COPY utils.py .
COPY requirements.txt .

RUN python -m venv .venv .

RUN source .venv/bin/activate

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
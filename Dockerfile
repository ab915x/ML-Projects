FROM python:3.13.5-bullseye

WORKDIR /app

COPY app/app.py . 
COPY app/create_model.py .
COPY app/data_tests.py .
COPY app/data_processing.py .
COPY app/utils.py .
COPY requirements.txt .

RUN mkdir -p /app/data /app/reports

RUN python -m venv .venv .

RUN . .venv/bin/activate

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
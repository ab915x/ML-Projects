FROM python:3.13.5-bullseye

WORKDIR /app

RUN mkdir -p data reports && \
    chmod -R 750 data reports

COPY requirements.txt .

RUN python -m venv .venv && \
    . .venv/bin/activate && \
    pip install -r requirements.txt

COPY app/app.py . 
COPY app/create_model.py .
COPY app/data_tests.py .
COPY app/data_processing.py .
COPY app/utils.py .

CMD [".venv/bin/python", "app.py"]
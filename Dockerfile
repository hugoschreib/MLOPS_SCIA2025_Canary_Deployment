FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow

WORKDIR /app

COPY server.py server.py

CMD ["uvicorn",  "--host", "0.0.0.0", "server:app"]

FROM python:3.10-slim

RUN pip install --no-cache-dir mlflow==2.12.1

EXPOSE 5000
ENTRYPOINT ["mlflow"]

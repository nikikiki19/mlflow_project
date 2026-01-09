- `src/train.py` — обучение, MLflow‑логирование, параметры из CLI.
- `src/inference.py` — батч‑инференс из Registry- `models:/Churn_Model/Production`.
- `src/client.py` — REST‑клиент для `mlflow models serve`
- `src/data_validation.py` — проверки данных перед обучением
- `mlruns/`, `mlflow.db` — артефакты и метаданные MLflow.

## требования
- Python 3.10+
- Java 
- Docker 

## датасеты лежат:
- train: `data/raw/cell2celltrain.csv`
- holdout: `data/raw/cell2cellholdout.csv`

## для запуска:
```bash
python -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

## docker для поднятия mlflow
```bash
docker compose build mlflow
docker compose up -d mlflow
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
```
UI: http://localhost:5001

## запуск обучения(пример)
```bash
python -m src.train \
  --data-path data/raw/cell2celltrain.csv \
  --experiment-name cell2cell-churn \
  --run-name gbt_depth5_lr01 \
  --max-depth 5 \
  --max-iter 60 \
  --step-size 0.1 \
  --use-class-weights
```
скрипт:
- детектит категориальные/числовые признаки
- заполняет пропуски
- строит Pipeline
- логирует ROC‑AUC / F1 / Accuracy и модель

сделай 3–5 запусков с разными гиперпараметрами и сохраните скриншот UI.

## model registry
Лучшая модель регистрируется в MLflow Registry и помечается `Production`.
В проекте используется имя модели: `Churn_Model`

регистрация через ui:
1. открыть лучший run.
2. `Register model` → `Churn_Model`.
3. назначить стадию `Production`

## batch‑инференс
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
python -m src.inference \
  --data-path data/raw/cell2cellholdout.csv \
  --output-path data/predictions/holdout
```
результат в папке `data/predictions/holdout/part-*.csv`.

## rest api 
запуск сервера:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 \
  mlflow models serve -m "models:/Churn_Model/Production" \
  -p 5002 --host 127.0.0.1 --env-manager local
```


проверка: 
```bash
.venv311/bin/python -m src.client --data-path data/raw/cell2celltrain.csv --n-rows 1
```

результат:
- `prediction = 1.0` → churn 
- `prediction = 0.0` → no churn 




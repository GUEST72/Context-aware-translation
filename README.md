# Context-aware-translation

## Upload parsing with Redis + Celery

`POST /upload_pdf` now queues PDF parsing in Celery instead of parsing inside the request.

### 1) Install backend dependencies

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2) Start Redis locally (Docker)

```bash
docker run -d --name context-translation-redis -p 6379:6379 redis
```

### 3) Start Celery worker

```bash
cd backend/app
../.venv/bin/celery -A celery_config.celery_app worker --loglevel=info --concurrency=2
```

### 4) Start FastAPI

```bash
cd backend/app
../.venv/bin/python main.py
```

### 5) Queue upload work

```bash
curl -F "file=@backend/app/uploads/computer-networking-a-top-down-approach-8th-edition.pdf" \
  http://127.0.0.1:8000/upload_pdf
```

The response returns `task_id`.

### 6) Check task status

```bash
curl http://127.0.0.1:8000/upload_pdf/status/<task_id>
```

State moves from `PENDING`/`STARTED` to `SUCCESS` or `FAILURE`.

## Load tests

Single-run or concurrent queueing test:

```bash
python3 backend/tests/load/test_upload_file_load.py \
  --url http://127.0.0.1:8000/upload_pdf \
  --requests 20 \
  --concurrency 5 \
  --file backend/app/uploads/computer-networking-a-top-down-approach-8th-edition.pdf \
  --report backend/tests/load/report.json
```

import os

from celery import Celery


BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/1")

celery_app = Celery(
    "context_aware_translation",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    imports=("tasks",),
    task_track_started=True,
    timezone="UTC",
    enable_utc=True,
)

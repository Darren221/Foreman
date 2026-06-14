"""The Celery application: specialists execute as tasks on Redis-brokered workers.

The graph fans subtasks out to this app as a parallel group and gathers the
results. Tasks are registered with `@shared_task` (see `tasks.py`), so they bind to
whichever app is current — this module creates that app and makes it the default.

With `task_always_eager` (tests, local single-process runs) tasks run synchronously
in the calling process and the broker is never touched, so the suite stays offline.
Everything crossing the task boundary is JSON, so eager and real workers behave
identically.
"""

from __future__ import annotations

from celery import Celery

from foreman.config import Settings


def build_celery_app(settings: Settings | None = None) -> Celery:
    settings = settings or Settings()
    app = Celery(
        "foreman",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_always_eager=settings.celery_task_always_eager,
        # Eager failures are captured, not raised, so eager runs match a real
        # worker gathered with propagate=False — the graph degrades a failed
        # subtask into a failure-output rather than crashing.
        task_eager_propagates=False,
    )
    return app


app = build_celery_app()
app.set_default()

# Import the tasks so their @shared_task registrations attach to the app.
from foreman.workers import tasks  # noqa: E402,F401

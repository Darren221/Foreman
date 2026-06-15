"""Skip infrastructure-dependent tests unless explicitly opted into.

Tests marked `requires_docker` / `requires_postgres` / `requires_redis` need real
services, so plain `pytest` (CI, local) skips them and stays offline. They run
under docker-compose, where the matching env var is set.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from foreman.storage import Conn

_INFRA_MARKERS = {
    "requires_docker": "FOREMAN_TEST_DOCKER",
    "requires_postgres": "FOREMAN_TEST_POSTGRES",
    "requires_redis": "FOREMAN_TEST_REDIS",
}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    for item in items:
        for marker, env in _INFRA_MARKERS.items():
            if marker in item.keywords and not os.environ.get(env):
                item.add_marker(pytest.mark.skip(reason=f"infra test; set {env}=1 to run"))


@pytest.fixture(autouse=True)
def _celery_eager() -> Iterator[None]:
    """Run Celery tasks in-process for the offline suite — the graph fans specialists
    out to Celery, so without eager mode it would block on a real broker. A
    `requires_redis` test flips this off to exercise real workers."""
    from foreman.workers.celery_app import app

    previous = app.conf.task_always_eager
    app.conf.task_always_eager = True
    try:
        yield
    finally:
        app.conf.task_always_eager = previous


@pytest.fixture(
    params=[
        "sqlite",
        pytest.param("postgres", marks=pytest.mark.requires_postgres),
    ]
)
def open_conn(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[Callable[[], Conn]]:
    """A backend-parameterized factory for storage tests: call it to open a `Conn`.

    Every test that requests it runs twice — embedded SQLite (always) and Postgres
    (only when FOREMAN_TEST_POSTGRES is set; otherwise that param is skipped by the
    marker logic above). Calling the factory more than once opens fresh connections
    to the *same* backing store, which is how the "survives reopen / across
    processes" tests obtain a second handle. The store classes own table creation,
    so the factory only hands back connections.
    """
    if request.param == "sqlite":
        path = tmp_path / "store.sqlite"
        yield lambda: Conn.sqlite(path)
        return

    # Postgres persists across tests (unlike a fresh tmp_path file), so we drop the
    # stores' tables before and after to give each test a clean slate.
    dsn = os.environ.get(
        "FOREMAN_TEST_POSTGRES_DSN",
        "postgresql://postgres:postgres@localhost:5432/foreman_test",
    )
    opened: list[Conn] = []

    def _drop() -> None:
        admin = Conn.postgres(dsn)
        admin.execute("DROP TABLE IF EXISTS approvals")
        admin.execute("DROP TABLE IF EXISTS spans")
        admin.commit()
        admin.close()

    def _open() -> Conn:
        conn = Conn.postgres(dsn)
        opened.append(conn)
        return conn

    _drop()
    yield _open
    for conn in opened:
        conn.close()
    _drop()

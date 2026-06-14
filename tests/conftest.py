"""Skip infrastructure-dependent tests unless explicitly opted into.

Tests marked `requires_docker` / `requires_postgres` / `requires_redis` need real
services, so plain `pytest` (CI, local) skips them and stays offline. They run
under docker-compose, where the matching env var is set.
"""

from __future__ import annotations

import os

import pytest

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

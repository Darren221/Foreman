"""Durable-state storage.

A backend-agnostic `Conn` seam (SQLite | Postgres) that `ApprovalQueue` and
`TraceStore` sit on, plus (in `factory`) the construction of the checkpointer,
approval queue, and trace store from configuration — so calling code never picks
a backend itself.
"""

from foreman.storage.db import Conn, assert_translatable

__all__ = ["Conn", "assert_translatable"]

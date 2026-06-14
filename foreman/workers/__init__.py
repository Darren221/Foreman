"""Distributed execution: the Celery app and the specialist task workers run on.

LangGraph stays the orchestrator and fans subtasks out to these workers; the graph
is never itself a Celery task, so pause/resume and tracing stay intact.
"""

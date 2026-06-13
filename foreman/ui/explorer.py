"""Streamlit trace explorer: walk a recorded run's span tree.

Thin by design — it lists recorded runs and renders the selected run's tree
(status-coloured, with per-node latency/cost and click-to-expand prompt/response)
plus the cost summary. All shaping lives in `explorer_wiring.py` and `cost.py`.

Run it with:  streamlit run foreman/ui/explorer.py
"""

from __future__ import annotations

import streamlit as st

from foreman.config import Settings
from foreman.observability import RunSummary, TraceStore, summarize
from foreman.ui.explorer_wiring import flatten, format_detail, span_content, status_icon


@st.cache_resource  # type: ignore[untyped-decorator, unused-ignore]  # untyped only when streamlit absent
def _store() -> TraceStore:
    return TraceStore(Settings().trace_path)


def _run_label(run: RunSummary) -> str:
    task = run.attributes.get("foreman.task", run.name)
    return f"{task}  ·  {run.trace_id[:8]}"


def _render_summary(store: TraceStore, trace_id: str) -> None:
    root = store.get_trace(trace_id)
    if root is None:
        st.info("No spans for this run.")
        return

    cost = summarize(root)
    columns = st.columns(4)
    columns[0].metric("Wall clock", f"{cost.wall_clock_ms:.0f} ms")
    columns[1].metric("Tool calls", str(cost.tool_calls))
    columns[2].metric("Tokens", f"{cost.total.input_tokens}+{cost.total.output_tokens}")
    columns[3].metric("Cost", f"${cost.total.cost_usd:.4f}")

    st.subheader("Trace")
    for row in flatten(root):
        indent = " " * row.depth  # em-spaces for tree indentation
        st.write(f"{indent}{status_icon(row.span)} `{row.span.name}` — {format_detail(row.span)}")
        prompt, response = span_content(row.span)
        if prompt is not None or response is not None:
            with st.expander(f"{indent}↳ prompt / response"):
                st.caption("Prompt")
                st.code(prompt or "")
                st.caption("Response")
                st.code(response or "")


def main() -> None:
    st.title("Foreman — trace explorer")
    store = _store()
    runs = store.list_runs()
    if not runs:
        st.info("No recorded runs.")
        return
    run = st.selectbox("Run", runs, format_func=_run_label)
    _render_summary(store, run.trace_id)


main()

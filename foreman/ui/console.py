"""Streamlit submit-and-watch console: type a task, dispatch it, and watch the result.

Thin by design, like the other UIs. It's a *client* of the orchestration API, so a
submission runs on the api+worker services exactly as any caller would drive them; all
the HTTP and parsing lives in `console_wiring.py`. This page is just the form, the result,
and links into the review queue and trace explorer.

Run it with:  streamlit run foreman/ui/console.py
"""

from __future__ import annotations

import os

import streamlit as st

from foreman.ui.console_wiring import (
    ForemanClient,
    HttpForemanClient,
    TaskView,
    body_text,
    is_pending,
)

_API_URL = os.environ.get("FOREMAN_API_URL", "http://localhost:8000")
_REVIEW_URL = os.environ.get("FOREMAN_REVIEW_URL", "http://localhost:8501")
_EXPLORER_URL = os.environ.get("FOREMAN_EXPLORER_URL", "http://localhost:8502")


@st.cache_resource  # type: ignore[untyped-decorator, unused-ignore]  # untyped only when streamlit absent
def _client() -> ForemanClient:
    """One API client, reused across reruns. Points at the api service (`FOREMAN_API_URL`)."""
    return HttpForemanClient(_API_URL)


def _render_result(view: TaskView) -> None:
    if is_pending(view):
        st.warning(f"Run `{view.id}` paused for approval.")
        st.write(body_text(view))
        st.markdown(f"Resolve it in the [review queue]({_REVIEW_URL}), then re-check below.")
        if st.button("Re-check status"):
            st.session_state["view"] = _client().status(view.id)
            st.rerun()
    else:
        st.success(f"Run `{view.id}` completed.")
        st.markdown(body_text(view))
    st.caption(f"Inspect the run's trace in the [explorer]({_EXPLORER_URL}).")


def main() -> None:
    st.title("Foreman — submit a task")
    st.write(
        "Describe a task; the supervisor plans it, the crew runs it, and the result "
        "appears below. Tick *require approval* to route it through the human-in-the-loop."
    )

    with st.form("submit"):
        description = st.text_area("Task", placeholder="Research the history of the bicycle")
        col1, col2 = st.columns(2)
        require_approval = col1.checkbox(
            "Require approval", help="Pause for a human before finishing"
        )
        sensitive = col2.checkbox("Sensitive", help="Flag as sensitive; raises the approval bar")
        submitted = st.form_submit_button("Submit")

    if submitted and description.strip():
        with st.spinner("Planning and running..."):
            st.session_state["view"] = _client().submit(
                description.strip(), require_approval=require_approval, sensitive=sensitive
            )
    elif submitted:
        st.error("Enter a task description first.")

    view = st.session_state.get("view")
    if view is not None:
        st.divider()
        _render_result(view)


main()

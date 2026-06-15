# README screenshots

The main README references two captures from a live run. Drop the PNGs here with these
exact names (the recording session in [`../DEMO.md`](../DEMO.md) is the natural time to grab
them):

- **`review-queue.png`** — the Streamlit review UI (`:8501`) showing a paused run awaiting
  approval: the task context, the proposed action, recalled memories, and the decision
  controls. Trigger one by submitting a task with a sensitive/approval flag.
- **`trace-explorer.png`** — the Streamlit trace explorer (`:8502`) showing a run's
  status-coloured span tree, with a specialist/tool/LLM span expanded to reveal the nested
  cross-worker spans and per-node cost/latency. This is the visual highlight — pick a run
  with the full crew so the tree is rich.

Until they're added, the README's screenshot links show as broken images on GitHub — that's
expected for the in-progress portfolio and resolves the moment you record.

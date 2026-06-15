#!/usr/bin/env bash
# Submit the showcase task to the running stack and surface the outcome.
# Bring the stack up first:  docker compose up --build -d
set -euo pipefail

API="${FOREMAN_API:-http://localhost:8000}"
TASK="${FOREMAN_TASK:-Research the most popular Python web frameworks, analyse which has grown fastest, and write a short recommendation.}"

echo "Submitting showcase task to $API ..."
resp=$(curl -sf -X POST "$API/tasks" \
  -H 'content-type: application/json' \
  -d "$(python3 -c 'import json,os; print(json.dumps({"description": os.environ["TASK"]}))' TASK="$TASK")")

id=$(printf '%s' "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
status=$(printf '%s' "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])')
echo "run id: $id   status: $status"

if [ "$status" = "pending" ]; then
  echo "Approval required — review queue:"
  curl -sf "$API/approvals" | python3 -m json.tool
  echo "Approve it in the review UI:  http://localhost:8501"
else
  echo "Result:"
  curl -sf "$API/tasks/$id" | python3 -m json.tool
fi

echo "Trace explorer:  http://localhost:8502"

#!/usr/bin/env bash
# InferFleet benchmark runner.
#
# Starts a fresh all-in-one server (own temp DB => cold cache, fair A/B),
# warms it with one request, drives load with inferfleet.loadgen, writes
# results/<name>.json, and tears the server down.
#
# Usage: scripts/bench.sh <name> <static|continuous> [loadgen args...]
set -euo pipefail

cd "$(dirname "$0")/.."

NAME="${1:?usage: bench.sh <name> <static|continuous> [loadgen args...]}"
SCHED="${2:?scheduler must be static or continuous}"
shift 2

PORT="${IF_BENCH_PORT:-8123}"
KEY="${INFERFLEET_KEY:-ifk_live_pro00000000000000000000000000001}"
RUN_DIR="$(mktemp -d -t "if-bench-XXXXXX")"
LOG="${RUN_DIR}/server.log"
OUT="results/${NAME}.json"
mkdir -p results

cleanup() {
  kill "${SERVER_PID:-0}" 2>/dev/null || true
  wait "${SERVER_PID:-0}" 2>/dev/null || true
}
trap cleanup EXIT

IF_SCHEDULER="$SCHED" \
IF_INLINE=true \
IF_PORT="$PORT" \
IF_AUTO_SEED=true \
IF_RATE_PRO=1000000 \
DATABASE_URL="sqlite:///${RUN_DIR}/bench.db" \
  .venv/bin/python -m inferfleet >"$LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 150); do
  if curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server died during startup:" >&2
    tail -20 "$LOG" >&2
    exit 1
  fi
  sleep 0.1
done

# One warmup request: seeds demo keys/DB, JIT-ifies the process.
curl -sf -X POST "http://127.0.0.1:${PORT}/v1/generate" \
  -H "Authorization: Bearer ${KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"bench warmup","max_tokens":8}' >/dev/null

echo "== bench ${NAME} (scheduler=${SCHED}) =="
.venv/bin/python -m inferfleet.loadgen \
  --url "http://127.0.0.1:${PORT}" \
  --key "$KEY" \
  --json "$OUT" \
  "$@"

# Keep the server log next to the result for post-mortems.
cp "$LOG" "results/${NAME}.server.log"
echo "wrote ${OUT}"

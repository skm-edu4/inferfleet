#!/usr/bin/env bash
# InferFleet live demo — run `make run` in another terminal first.
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"
KEY="${INFERFLEET_KEY:-ifk_live_pro00000000000000000000000000001}"
AUTH=(-H "Authorization: Bearer ${KEY}")

step() { printf '\n\033[1;34m== %s\033[0m\n' "$1"; }

step "1. health"
curl -s "${BASE}/healthz" | python3 -m json.tool

step "2. who am I (tier, quota, usage)"
curl -s "${BASE}/v1/me" "${AUTH[@]}" | python3 -m json.tool

step "3. generate (sync)"
curl -s "${BASE}/v1/generate" "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"How do I reset my account password?","max_tokens":64,"temperature":0.2}' \
  | python3 -m json.tool

step "4. generate again -> exact cache hit (no GPU time)"
curl -s "${BASE}/v1/generate" "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"How do I reset my account password?","max_tokens":64,"temperature":0.2}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({"cached":d["cached"],"kind":d["cache_kind"],"latency_ms":d["latency_ms"]},indent=2))'

step "5. streaming (server-sent events, token by token)"
{ curl -sN "${BASE}/v1/generate" "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain dynamic batching for GPUs.","stream":true,"max_tokens":64,"temperature":0.9}' \
  | head -c 900; } || true
echo

step "6. history (written asynchronously by the settlement service)"
sleep 1
curl -s "${BASE}/v1/history?limit=3" "${AUTH[@]}" | python3 -m json.tool | head -40

step "7. ops stats (queue depth, circuit breaker, p99)"
curl -s "${BASE}/v1/admin/stats" | python3 -m json.tool | head -30

step "8. prometheus metrics"
curl -s "${BASE}/metrics" | grep -E '^inferfleet_(requests_total|ttfb_seconds_bucket|queue_depth|circuit_state)' | head -12

echo
echo "Dashboard: ${BASE}/dashboard   API docs: ${BASE}/docs"

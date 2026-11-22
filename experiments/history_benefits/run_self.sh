#!/usr/bin/env bash
# Full-pipeline evaluation (规则 §2.3), queued after run_medium.sh finishes. 6 jobs in 3 lanes,
# then the report is rebuilt. Resumes like every other stage.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="${CONFIG:-experimental/history_benefits/medium.yaml}"
OUT="$ROOT/$(grep '^output:' "$ROOT/$CONFIG" | awk '{print $2}')"
LOGS="$OUT/logs"; PY="$ROOT/.venv/bin/python"
cd "$ROOT/experiments/history_benefits"
stage() { local name="$1"; shift; echo "$(date '+%F %T') start $name"
  "$PY" medium.py --config "$CONFIG" "$@" >>"$LOGS/$name.log" 2>&1; echo "$(date '+%F %T') done  $name"; }
lane() { for job in "$@"; do eval "$job"; done; }
# independent of the equal-time reruns (different files, no timing used): 2 lanes next to them
lane "stage self_C_two self --condition self_C_two" "stage self_C_direct self --condition self_C_direct" \
     "stage self_A_direct self --condition self_A_direct" &
lane "stage self_B_two self --condition self_B_two" "stage self_B_direct self --condition self_B_direct" \
     "stage self_T8_direct self --condition self_T8_direct" &
wait
echo "waiting for run_medium.sh to finish before the report..."
until grep -q "ALL DONE" "$LOGS/pipeline.log" 2>/dev/null; do sleep 60; done
stage report_self report
echo "$(date '+%F %T') SELF DONE"

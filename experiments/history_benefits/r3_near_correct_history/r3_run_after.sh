#!/usr/bin/env bash
# Exploratory analyses of round 3 (0-实验规则.md §12.7), queued after r3_run.sh finishes:
# the input ablation of the history on the selected C checkpoint, then the report is rebuilt
# (it then also carries the copy-rate and ablation tables). r3_run.sh itself is not edited
# while it runs (bash reads a running script lazily).
#
#   tmux new-window -t r3 -n after "experiments/history_benefits/r3_near_correct_history/r3_run_after.sh; exec bash"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIG="${CONFIG:-experiments/history_benefits/r3_near_correct_history/config.yaml}"
OUT="$ROOT/$(grep '^output:' "$ROOT/$CONFIG" | awk '{print $2}')"
LOGS="$OUT/logs"; PY="$ROOT/.venv/bin/python"
cd "$ROOT/experiments/history_benefits/r3_near_correct_history"
echo "waiting for r3_run.sh (ALL DONE in $LOGS/pipeline.log)..."
until grep -q "ALL DONE" "$LOGS/pipeline.log" 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') start ablation_selected" | tee -a "$LOGS/pipeline.log"
"$PY" -W ignore ablate.py --config "$CONFIG" --selected --threads 4 >>"$LOGS/ablation_selected.log" 2>&1
echo "$(date '+%F %T') done  ablation_selected" | tee -a "$LOGS/pipeline.log"
echo "$(date '+%F %T') start report_after" | tee -a "$LOGS/pipeline.log"
"$PY" -W ignore r3_run.py --config "$CONFIG" report >>"$LOGS/report_after.log" 2>&1
echo "$(date '+%F %T') done  report_after" | tee -a "$LOGS/pipeline.log"
echo "$(date '+%F %T') AFTER DONE" | tee -a "$LOGS/pipeline.log"

#!/usr/bin/env bash
# Exploratory analyses of the mix run (正式实验规则.md §12.7), queued after run_mix.sh finishes:
# the input ablation of the history on the selected C checkpoint, then the report is rebuilt
# (it then also carries the copy-rate and ablation tables). run_mix.sh itself is not edited
# while it runs (bash reads a running script lazily).
#
#   tmux new-window -t mix -n after "experiments/history_benefits/run_mix_after.sh; exec bash"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="${CONFIG:-experimental/history_benefits/mix.yaml}"
OUT="$ROOT/$(grep '^output:' "$ROOT/$CONFIG" | awk '{print $2}')"
LOGS="$OUT/logs"; PY="$ROOT/.venv/bin/python"
cd "$ROOT/experiments/history_benefits"
echo "waiting for run_mix.sh (ALL DONE in $LOGS/pipeline.log)..."
until grep -q "ALL DONE" "$LOGS/pipeline.log" 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') start ablation_selected" | tee -a "$LOGS/pipeline.log"
"$PY" -W ignore ablate.py --config "$CONFIG" --selected --threads 4 >>"$LOGS/ablation_selected.log" 2>&1
echo "$(date '+%F %T') done  ablation_selected" | tee -a "$LOGS/pipeline.log"
echo "$(date '+%F %T') start report_after" | tee -a "$LOGS/pipeline.log"
"$PY" -W ignore mix_run.py --config "$CONFIG" report >>"$LOGS/report_after.log" 2>&1
echo "$(date '+%F %T') done  report_after" | tee -a "$LOGS/pipeline.log"
echo "$(date '+%F %T') AFTER DONE" | tee -a "$LOGS/pipeline.log"

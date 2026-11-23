#!/usr/bin/env bash
# Round 4 (r4_read_history_predictions) run (设计与预注册.md §14). Start it inside tmux:
#
#   experiments/history_benefits/r4_read_history_predictions/r4_run.sh start   # opens tmux session "r4"
#   tmux attach -t r4                                   # watch; Ctrl-b d to detach
#
# Every stage resumes, so re-running after an interruption continues where it stopped.
# Order: link -> I and IC train in parallel -> ablation of r3_C / I / IC at the compared step
#        -> copy probe of the three -> report.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIG="${CONFIG:-experiments/history_benefits/r4_read_history_predictions/config.yaml}"
OUT="$ROOT/$(grep '^output:' "$ROOT/$CONFIG" | awk '{print $2}')"
LOGS="$OUT/logs"
PY="$ROOT/.venv/bin/python"
SESSION="${SESSION:-r4}"

if [[ "${1:-}" == "start" ]]; then
  mkdir -p "$LOGS"
  tmux new-session -d -s "$SESSION" -n pipeline -c "$ROOT" \
    "CONFIG=$CONFIG caffeinate -ims $0 run 2>&1 | tee -a $LOGS/pipeline.log; echo 'pipeline finished'; exec bash"
  tmux new-window -t "$SESSION" -n logs -c "$LOGS" "sleep 5; tail -n 3 -F $LOGS/*.log"
  tmux new-window -t "$SESSION" -n shell -c "$ROOT"
  echo "started tmux session '$SESSION' (tmux attach -t $SESSION)"
  exit 0
fi

mkdir -p "$LOGS"
cd "$ROOT/experiments/history_benefits/r4_read_history_predictions"
stage() {  # stage <log-name> <args...>
  local name="$1"; shift
  echo "$(date '+%F %T') start $name"
  "$PY" -W ignore::RuntimeWarning r4_run.py --config "$CONFIG" "$@" >>"$LOGS/$name.log" 2>&1
  echo "$(date '+%F %T') done  $name"
}

stage link link

# training: I and IC in parallel
stage train_I train --group I --threads 4 &
stage train_IC train --group IC --threads 4 &
wait

for which in r3_C I IC; do
  stage "ablate_$which" ablate --which "$which" --threads 8
done
for which in r3_C I IC; do
  stage "copy_probe_$which" copy-probe --which "$which" --threads 8
done

stage report report
echo "$(date '+%F %T') ALL DONE"

#!/usr/bin/env bash
# Near-correct-history small-scale run (正式实验规则.md §12). Start it inside tmux:
#
#   experiments/history_benefits/run_mix.sh start      # opens tmux session "mix"
#   tmux attach -t mix                                  # watch; Ctrl-b d to detach
#
# Every stage resumes, so re-running after an interruption continues where it stopped.
# Order: link -> pools (3 lanes + selection) -> A and C train in parallel -> select ->
#        full-pipeline evaluation (2 lanes) + diagnostic -> report.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="${CONFIG:-experimental/history_benefits/mix.yaml}"
OUT="$ROOT/$(grep '^output:' "$ROOT/$CONFIG" | awk '{print $2}')"
LOGS="$OUT/logs"
PY="$ROOT/.venv/bin/python"
SESSION="${SESSION:-mix}"

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
cd "$ROOT/experiments/history_benefits"
stage() {  # stage <log-name> <args...>
  local name="$1"; shift
  echo "$(date '+%F %T') start $name"
  "$PY" -W ignore::RuntimeWarning mix_run.py --config "$CONFIG" "$@" >>"$LOGS/$name.log" 2>&1
  echo "$(date '+%F %T') done  $name"
}
lane() { for job in "$@"; do eval "$job"; done; }

stage link link

# pools: new training problems in 3 lanes, the selection set in a 4th
for i in 0 1 2; do
  stage "pools_train_$i" pools-train --part "$i/3" &
done
stage pools_selection pools-selection &
wait
stage pools_check pools-check

# training: A and C in parallel
lane "stage train_A train --group A --threads 4" "stage select_A select --group A" &
lane "stage train_C train --group C --threads 4" "stage select_C select --group C" &
wait

# evaluation: 3 full-pipeline conditions + the diagnostic in 2 lanes
lane "stage self_C_two self --condition self_C_two" "stage diagnose diagnose" &
lane "stage self_C_direct self --condition self_C_direct" "stage self_A_direct self --condition self_A_direct" &
wait

stage report report
echo "$(date '+%F %T') ALL DONE"

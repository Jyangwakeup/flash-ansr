#!/usr/bin/env bash
# Round 2 (r2_random_vs_model_history) pipeline (0-训练评估流程.md). Start it inside tmux:
#
#   experiments/history_benefits/r2_random_vs_model_history/r2_run.sh start     # opens tmux session "r2"
#   tmux attach -t r2                                    # watch; Ctrl-b d to detach
#
# Every stage resumes, so re-running after an interruption continues where it stopped.
# Parallel lanes: A/B/C train at once; final evaluations and equal-time reruns run in 3 lanes.
# The equal-time measurement runs alone (its timings set the budget).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIG="${CONFIG:-experiments/history_benefits/r2_random_vs_model_history/config.yaml}"
OUT="$ROOT/$(grep '^output:' "$ROOT/$CONFIG" | awk '{print $2}')"
LOGS="$OUT/logs"
PY="$ROOT/.venv/bin/python"
SESSION="${SESSION:-r2}"

if [[ "${1:-}" == "start" ]]; then
  mkdir -p "$LOGS"
  tmux new-session -d -s "$SESSION" -n pipeline -c "$ROOT" \
    "CONFIG=$CONFIG caffeinate -ims $0 run 2>&1 | tee -a $LOGS/pipeline.log; echo 'pipeline finished'; exec bash"
  tmux new-window -t "$SESSION" -n logs -c "$LOGS" \
    "sleep 2; tail -n 3 -F $LOGS/*.log 2>/dev/null"
  tmux new-window -t "$SESSION" -n shell -c "$ROOT"
  echo "started tmux session '$SESSION' (tmux attach -t $SESSION)"
  exit 0
fi

mkdir -p "$LOGS"
cd "$ROOT/experiments/history_benefits/r2_random_vs_model_history"
stage() {  # stage <log-name> <args...>
  local name="$1"; shift
  echo "$(date '+%F %T') start $name"
  "$PY" r2_run.py --config "$CONFIG" "$@" >>"$LOGS/$name.log" 2>&1
  echo "$(date '+%F %T') done  $name"
}
lane() { for job in "$@"; do eval "$job"; done; }

echo "waiting for candidate pools..."
until [[ -f "$OUT/pools-train_complete.json" && -f "$OUT/pools-validation_complete.json" ]]; do sleep 60; done

stage step0 step0

# 3 training lanes; each continues with its own cooldown branch
for g in A B C; do
  lane "stage train_$g train --group $g --threads 3" "stage cooldown_$g cooldown --group $g --threads 3" &
done
wait

# final evaluation: 8 jobs in 3 lanes
lane "stage final_1_A_none final --condition 1_A_none" "stage final_4_C_none final --condition 4_C_none" \
     "stage original original" &
lane "stage final_2_C_model final --condition 2_C_model" "stage final_5_B_none final --condition 5_B_none" \
     "stage step0_eval step0-eval" &
lane "stage final_3_B_random final --condition 3_B_random" "stage final_6_B_model final --condition 6_B_model" &
wait

# equal time: measure ALONE, then rerun at the budget in 3 lanes
stage equal_time_measure equal-time-measure
lane "stage et_original equal-time --condition original_T8" "stage et_1 equal-time --condition 1_A_none" \
     "stage et_4 equal-time --condition 4_C_none" &
lane "stage et_2 equal-time --condition 2_C_model" "stage et_5 equal-time --condition 5_B_none" &
lane "stage et_3 equal-time --condition 3_B_random" "stage et_6 equal-time --condition 6_B_model" &
wait

stage report report

# upload offline W&B runs once logged in (wandb login)
if "$PY" "$ROOT/experiments/exp_wandb.py" status 2>/dev/null | grep -q "logged_in=True"; then
  for run in "$OUT"/wandb/offline-run-*; do
    [[ -d "$run" ]] && "$ROOT/.venv/bin/wandb" sync "$run" >>"$LOGS/wandb_sync.log" 2>&1 || true
  done
fi
echo "$(date '+%F %T') ALL DONE"

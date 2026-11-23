# Rules for working in this repo

## Where experiment files go

Every experiment `<name>` is split across three places:

| What | Where |
|---|---|
| Code and scripts | `experiments/<name>/` |
| Raw run outputs (json, logs, checkpoints) | `runs/<name>/` |
| Analysis reports and write-ups | `experiment-results/<name>/` |

- Set all three paths before the first run. Never write outputs or reports into `experiments/`.
- Example: `experiments/T8vsT8-1/run.py` → `runs/T8vsT8-1/` → `experiment-results/T8vsT8-1/迁移对照.md`

## Log every experiment to Weights & Biases

Any new or changed script under `experiments/<name>/` that trains or evaluates must log to W&B through `experiments/exp_wandb.py`. Scripts that only summarize or plot existing results are exempt.

```python
sys.path.insert(0, str(ROOT / "experiments"))
from exp_wandb import Run
run = Run(config=cfg)              # group <name> and folder runs/<name>/ are inferred
run.log({"loss": loss}, step=step) # the metrics that matter: loss, solve rates, timings
run.summary({...}); run.finish()   # final numbers at the end
```

- Log progress while the script runs, not only at the end. For evaluations, log the running solve rate after each problem.
- Keep the local json/jsonl outputs as well. W&B is the viewer; the local files are the record.
- Never hard-code `mode` or `entity`. `exp_wandb` goes online when the user is logged in and offline otherwise.
- When reporting a new experiment, tell the user what shows up on W&B.

## Commit before a formal run

A formal run is one whose results will be kept, compared or reported. Smoke and debug runs are exempt.

- Commit the code first, even a version that may turn out wrong. Every kept result must lead back to the exact code that produced it, through the commit the run recorded.
- Once a run has recorded a commit, that commit is frozen: fix things with a new commit on top. Amending, rebasing or force-pushing it leaves the recorded hash pointing at nothing.
- When a round ends, tag its last commit with `git tag -a <name>/<round>` (e.g. `hb/r2-random-vs-model-history`) before the branch is merged. A squash merge drops the round's commits from `main`; the tag keeps them.

## Rounds: numbered names, shared core, one folder per round

One experiment runs in rounds. Example layout (`history_benefits`; details in `experiments/history_benefits/README.md`):

| What | Where |
|---|---|
| Shared code (keep it stable) | `experiments/history_benefits/core/` |
| Round 2: its config, smoke config and scripts | `experiments/history_benefits/r2_random_vs_model_history/` (`config.yaml`, `smoke.yaml`, `r2_*.py`) |
| Round 2 outputs | `runs/history_benefits/r2_random_vs_model_history/` |
| Round 2 report | `experiment-results/history_benefits/2-随机历史vs模型历史报告.md` |

- Name a round `r<N>_<what it tests>`, numbered in the order the rounds ran, so the name alone says what it does. Never use names like `pilot`, `medium`, `mix`. Use the same name for the code folder, the run folder, `wandb_group`, the report (`<N>-<中文名>.md`) and the tag (`hb/r<N>-...`).
- A round imports only `core/` and its own folder, never another round. Prefix its scripts with `r<N>_` (mypy and W&B job types need unique file names).
- Prefer changing only the round's config. A new way of building training samples goes in the round's folder, plugged in through `train(..., batch_maker=..., validator=...)`. A new model input structure goes in `src/flash_ansr/model/`, behind a model-config switch whose default keeps the old behaviour.
- Code writes generated tables only into `runs/`. Hand-written reports live in `experiment-results/`, and no script overwrites them.

## Keep earlier rounds runnable

- If `core/` must change, put new behaviour behind a config option whose default keeps the old behaviour, so older configs still run unchanged.
- Before committing a change to `core/`, run every round's `smoke.yaml` and confirm each one finishes.

## How to write for this user

- Always reply in Chinese. Keep only code, file names and commands in their original form.
- Use plain words. Lead with a concrete example (a table with numbers, an analogy), then name the term if needed. Explain any technical term right after you use it.
- Write analysis reports in Chinese with tables. Give a one-sentence conclusion first. State plainly when the sample is too small to support a conclusion.

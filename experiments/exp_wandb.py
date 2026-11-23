"""Shared Weights & Biases logging for every experiment under experiments/.

Usage (from any experiment script):

    sys.path.insert(0, str(ROOT / "experiments"))
    from exp_wandb import Run
    run = Run(config=cfg)          # group and folder come from the script's location
    run.log({"solved": 0.5}, step=3); run.summary({...}); run.finish()

* Called from experiments/<name>/..., group defaults to <name> and files go to runs/<name>/
  (the repo's layout rule), so a new experiment needs no extra settings.

* Project: WANDB_PROJECT env var, else "flash-ansr". Each experiment is a W&B *group*.
* Mode: WANDB_MODE env var if set; else "online" when logged in (``wandb login`` or
  WANDB_API_KEY), else "offline". Offline runs land in ``<out>/wandb`` and can be uploaded
  later with ``wandb sync <out>/wandb/offline-run-*`` (or ``python experiments/exp_wandb.py sync``).
* Set WANDB_MODE=disabled to switch logging off entirely.
* Logging never stops an experiment: any wandb failure is printed and ignored.

Global default (opt-in, once per virtualenv)
--------------------------------------------
``python experiments/exp_wandb.py install`` writes a startup hook (a .pth file) into the venv.
From then on, running ANY script under experiments/<name>/ opens a W&B run by itself, with no
code in the script: group <name>, files in runs/<name>/wandb, the command line as config, console
output and CPU/memory usage. A later ``wandb.init`` / ``Run(...)`` in the script (or the Trainer)
reuses that run instead of making a second, empty one. Metric curves still need ``run.log``.
Skip it for one command with WANDB_MODE=disabled; remove it with ``... uninstall``.
"""

from __future__ import annotations

import netrc
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PROJECT = "flash-ansr"
ROOT = Path(__file__).resolve().parents[1]


def logged_in() -> bool:
    """True when a W&B API key is available (env var or ~/.netrc), without prompting."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    hosts = {"api.wandb.ai"}
    try:  # the server wandb is configured for (newer versions default to api.forge.coreweave.com)
        from urllib.parse import urlparse

        import wandb

        hosts.add(urlparse(wandb.Settings().base_url).hostname or "")
    except Exception:  # noqa: BLE001
        pass
    if os.environ.get("WANDB_BASE_URL"):
        from urllib.parse import urlparse

        hosts.add(urlparse(os.environ["WANDB_BASE_URL"]).hostname or "")
    try:
        auth = netrc.netrc()
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    return any(h and auth.authenticators(h) is not None for h in hosts)


def mode() -> str:
    return os.environ.get("WANDB_MODE") or ("online" if logged_in() else "offline")


def _experiment_of_caller() -> str | None:
    """Name of the experiments/<name>/ folder holding the script that created the run."""
    import inspect

    exp_dir = ROOT / "experiments"
    for frame in inspect.stack()[2:]:
        try:
            rel = Path(frame.filename).resolve().relative_to(exp_dir)
        except ValueError:
            continue
        if len(rel.parts) >= 2:
            return rel.parts[0]
    return None


class Run:
    """A thin, failure-tolerant wrapper around one wandb run."""

    def __init__(self, group: str | None = None, name: str | None = None, job_type: str | None = None,
                 config: dict | None = None, out: Any = None, project: str | None = None,
                 tags: Iterable[str] | None = None) -> None:
        self.run = None
        if group is None or out is None:
            guessed = _experiment_of_caller()
            group = group or guessed or "misc"
            if out is None and guessed:
                out = ROOT / "runs" / guessed
        try:
            import wandb

            if out is not None:
                Path(out).mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(
                project=os.environ.get("WANDB_PROJECT") or project or DEFAULT_PROJECT,
                group=group,
                name=name,
                job_type=job_type,
                tags=list(tags) if tags else None,
                config=config or {},
                dir=str(out) if out is not None else None,
                mode=mode(),
                reinit="create_new",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"wandb disabled: {exc}", flush=True)

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *_: Any) -> None:
        self.finish()

    def log(self, values: dict, step: int | None = None) -> None:
        if self.run is None:
            return
        try:
            self.run.log(values, step=step)
        except Exception as exc:  # noqa: BLE001
            print(f"wandb log failed: {exc}", flush=True)

    def table(self, key: str, columns: Iterable[str], rows: Iterable[Iterable[Any]]) -> None:
        if self.run is None:
            return
        try:
            import wandb

            self.run.log({key: wandb.Table(columns=list(columns), data=[list(r) for r in rows])})
        except Exception as exc:  # noqa: BLE001
            print(f"wandb table failed: {exc}", flush=True)

    def summary(self, values: dict) -> None:
        if self.run is None:
            return
        try:
            self.run.summary.update(values)
        except Exception as exc:  # noqa: BLE001
            print(f"wandb summary failed: {exc}", flush=True)

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:  # noqa: BLE001
                pass
            self.run = None


def sync(root: Path = ROOT / "runs") -> None:
    """Upload every offline run found under runs/ (needs ``wandb login`` first)."""
    if not logged_in():
        sys.exit("not logged in: run `wandb login` first")
    wandb_bin = Path(sys.executable).with_name("wandb")
    for run in sorted(root.rglob("offline-run-*")):
        if not run.is_dir() or (run / ".synced").exists():
            continue
        newest = max((f.stat().st_mtime for f in run.rglob("*") if f.is_file()), default=0)
        if time.time() - newest < 300:   # written in the last 5 minutes: still running, sync later
            print(f"skip (still running) {run}", flush=True)
            continue
        print(f"sync {run}", flush=True)
        if subprocess.run([str(wandb_bin), "sync", str(run)]).returncode == 0:
            (run / ".synced").touch()


# ----------------------------------------------------------------------------- global default
_auto: Any = None          # the run opened at startup, until the script's own wandb.init claims it
PTH_NAME = "zz_exp_wandb_auto.pth"


def _script_experiment() -> tuple[str, str] | None:
    """(experiment name, script path) when this process runs a .py file under experiments/<name>/."""
    script = sys.argv[0] if sys.argv else ""
    if not script.endswith(".py") or not os.path.isfile(script):   # skips -c / -m / worker processes
        return None
    try:
        rel = Path(script).resolve().relative_to(ROOT / "experiments")
    except (ValueError, OSError):
        return None
    if len(rel.parts) < 2:          # experiments/exp_wandb.py itself
        return None
    return rel.parts[0], str(rel)


def autostart() -> None:
    """Run by the startup hook: open a W&B run for an experiment script."""
    global _auto
    found = _script_experiment()
    if _auto is not None or found is None or mode() == "disabled":
        return
    group, script = found
    try:
        import wandb

        out = ROOT / "runs" / group
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(script).stem
        _auto = wandb.init(
            project=os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT,
            group=group, job_type=stem,
            name=f"{stem} {' '.join(sys.argv[1:])}".strip()[:120],
            config={"script": script, "argv": sys.argv[1:]},
            dir=str(out), mode=mode(), reinit="create_new",
        )
        _patch_init(wandb)
    except Exception as exc:  # noqa: BLE001
        print(f"wandb auto-start skipped: {exc}", flush=True)
        _auto = None


def _patch_init(wandb: Any) -> None:
    """The script's first wandb.init (direct, via Run, or via the Trainer) reuses the auto run."""
    original = wandb.init

    def init(*args: Any, **kwargs: Any) -> Any:
        global _auto
        run, _auto = _auto, None
        if run is None:
            return original(*args, **kwargs)
        if kwargs.get("mode") == "disabled":       # the script explicitly wants no logging
            run.finish()
            return original(*args, **kwargs)
        try:
            cfg = kwargs.get("config")
            if cfg:
                run.config.update(cfg if isinstance(cfg, dict) else dict(cfg), allow_val_change=True)
            if kwargs.get("name"):
                run.name = str(kwargs["name"])
            if kwargs.get("tags"):
                run.tags = tuple(run.tags or ()) + tuple(kwargs["tags"])
        except Exception as exc:  # noqa: BLE001
            print(f"wandb config update failed: {exc}", flush=True)
        return run

    wandb.init = init


def _pth_path() -> Path:
    import sysconfig

    return Path(sysconfig.get_paths()["purelib"]) / PTH_NAME


def install() -> None:
    """Write the startup hook into this virtualenv (safe to repeat)."""
    me = str(Path(__file__).resolve())
    prefix = str(ROOT / "experiments") + os.sep
    boot = (
        "if sys.argv and sys.argv[0].endswith('.py') and 'exp_wandb' not in sys.modules"
        f" and os.path.abspath(sys.argv[0]).startswith({prefix!r}):\n"
        "    import importlib.util as u\n"
        f"    s = u.spec_from_file_location('exp_wandb', {me!r})\n"
        "    m = u.module_from_spec(s); sys.modules['exp_wandb'] = m; s.loader.exec_module(m)\n"
        "    m.autostart()\n"
    )
    # .pth files run lines that start with "import"; this one only acts for experiments/ scripts
    _pth_path().write_text(f"import sys, os; exec({boot!r})\n")
    print(f"installed {_pth_path()}\nremove with: python experiments/exp_wandb.py uninstall")


def uninstall() -> None:
    _pth_path().unlink(missing_ok=True)
    print(f"removed {_pth_path()}")


if __name__ == "__main__":
    if sys.argv[1:] == ["sync"]:
        sync()
    elif sys.argv[1:] == ["install"]:
        install()
    elif sys.argv[1:] == ["uninstall"]:
        uninstall()
    elif sys.argv[1:] == ["status"]:
        print(f"logged_in={logged_in()} mode={mode()} global_default={_pth_path().exists()}")
    else:
        sys.exit("usage: python experiments/exp_wandb.py install|uninstall|sync|status")

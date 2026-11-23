"""Round 1 (r1_pipeline_check): the gated CPU engineering pilot; never starts the formal experiment."""

from pathlib import Path
from typing import Any
import argparse
import json
import platform
import sys
from importlib.metadata import version as package_version
import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))  # shared code: experiments/history_benefits/core
from common import (  # noqa: E402
    ROOT,
    dump,
    sha,
    load_base,
    migrate,
    source,
    worker_config,
    read_problem,
    read_pool,
    assemble,
    collate,
    loss,
)
from flash_ansr.model.flash_ansr_model import FlashANSRModel  # noqa: E402
from pool import build  # noqa: E402
from checks import check  # noqa: E402
from train import train  # noqa: E402
from evaluate import evaluate  # noqa: E402


def main() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["small", "pilot", "costs", "all"])
    parser.add_argument("--config", default="experiments/history_benefits/r1_pipeline_check/config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    torch.set_num_threads(cfg["threads"])
    out = ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    weights = next((ROOT / cfg["model"]).glob("*.safetensors"), None)
    if weights is None:
        weights = next((ROOT / cfg["model"]).glob("*.pt"))
    manifest = dict(config=cfg, weights_sha256=sha(weights))
    path = out / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Manifest mismatch: use a new output directory")
    dump(path, manifest)
    files = [
        ROOT / cfg["recipe"],
        ROOT / cfg["train_recipe"],
        *(ROOT / cfg["recipe"]).parent.glob("*.yaml"),
        *(ROOT / cfg["model"]).glob("*.yaml"),
        *(ROOT / "experiments/history_benefits").glob("*.py"),
        ROOT / "src/flash_ansr/model/flash_ansr_model.py",
        ROOT / "src/flash_ansr/data/streaming.py",
    ]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    packages = {
        name: package_version(name)
        for name in ("torch", "numpy", "scipy", "simplipy", "symbolic-data")
    }
    version = out / "implementation.json"
    if version.exists() and json.loads(version.read_text())["hashes"] != hashes:
        raise ValueError(
            "Implementation changed: review artifacts and use a new output directory"
        )
    if version.exists() and json.loads(version.read_text()).get("packages", packages) != packages:
        raise ValueError("Runtime dependency versions changed: use a new output directory")
    dump(
        version,
        dict(
            hashes=hashes,
            python=sys.version,
            torch=torch.__version__,
            platform=platform.platform(),
            packages=packages,
        ),
    )
    if not (out / "step0.pt").exists():
        base = load_base(cfg)
        try:
            m, mc = migrate(cfg, base.flash_ansr_model)
        finally:
            base.close()
        torch.save(m.state_dict(), out / "step0.pt")
        dump(out / "model_config.json", mc)
    start_manifest = {
        name: sha(out / name) for name in ("step0.pt", "model_config.json")
    }
    start_path = out / "start_manifest.json"
    if start_path.exists() and json.loads(start_path.read_text()) != start_manifest:
        raise ValueError("Common step-0 weights/config changed: refusing to mix artifacts")
    dump(start_path, start_manifest)
    m = FlashANSRModel.from_config(json.loads((out / "model_config.json").read_text()))
    m.load_state_dict(torch.load(out / "step0.pt", weights_only=True))
    if args.stage == "costs":
        from storage import compare

        compare(cfg, out, m)
        return
    ids = list(range(8)) + [160, 161]
    if not (out / "small/passed.json").exists():
        build(cfg, out, ids)
        src, recipe = source(cfg)
        wc = worker_config(cfg, m, recipe)
        check(cfg, out, m, wc, src.catalog, ids)
        for group in ("A", "C"):
            m.load_state_dict(torch.load(out / "step0.pt", weights_only=True))
            m.train()
            m.zero_grad()
            p = read_problem(out, 0)
            pool, yh = read_pool(out, 0, "model")
            sample = assemble(cfg, m, wc, src.catalog, p, pool, yh, 0, group)
            value, _ = loss(m, collate([sample], m.tokenizer))
            value.backward()
            assert torch.isfinite(value) and all(
                torch.isfinite(p.grad).all()
                for p in m.parameters()
                if p.grad is not None
            )
            evaluate(cfg, out, m, group, [160, 161], "small")
        if not (out / "small/B/summary.json").exists():
            train(
                cfg,
                out,
                m,
                wc,
                src.catalog,
                "B",
                list(range(8)),
                [160, 161],
                cfg["small_steps"],
                "small",
            )
        m.load_state_dict(
            torch.load(out / "small/B/checkpoint_20.pt", weights_only=False)["model"]
        )
        evaluation = evaluate(cfg, out, m, "B", [160, 161], "small")
        if not any(r["actual_history"] and r["candidates"] for r in evaluation):
            raise AssertionError(
                "Small generation gate: no fitted target with valid K=1; do not expand"
            )
        dump(out / "small/passed.json", dict(passed=True, valid_history_required=True))
    if args.stage == "small":
        return
    small_eval = json.loads((out / "small/B/evaluation.json").read_text())
    if not any(r["actual_history"] and r["candidates"] for r in small_eval):
        raise AssertionError(
            "Small generation gate has no fitted K=1 target; refusing pilot"
        )
    build(cfg, out, list(range(cfg["problems"])))
    src, recipe = source(cfg)
    wc = worker_config(cfg, m, recipe)
    # A complete target that cannot fit stops expansion before the first pilot update.
    for pid in range(cfg["problems"]):
        p = read_problem(out, pid)
        pool, yh = read_pool(out, pid, "model")
        for seed in range(8):
            assemble(cfg, m, wc, src.catalog, p, pool, yh, seed, "C")
    for group in ("A", "B", "C"):
        if not (out / f"pilot/{group}/summary.json").exists():
            train(
                cfg,
                out,
                m,
                wc,
                src.catalog,
                group,
                list(range(160)),
                list(range(160, 200)),
                cfg["pilot_steps"],
                "pilot",
            )
        m.load_state_dict(
            torch.load(out / f"pilot/{group}/checkpoint_100.pt", weights_only=False)[
                "model"
            ]
        )
        if not (out / f"pilot/{group}/evaluation_complete.json").exists():
            evaluate(cfg, out, m, group, list(range(160, 200)), "pilot")
            dump(out / f"pilot/{group}/evaluation_complete.json", dict(complete=True))
    if args.stage == "all":
        from storage import compare

        compare(cfg, out, m)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Preserve a readable failure report instead of silently advancing past a gate.
        config_path = (
            sys.argv[sys.argv.index("--config") + 1]
            if "--config" in sys.argv
            else "experiments/history_benefits/r1_pipeline_check/config.yaml"
        )
        config = yaml.safe_load((ROOT / config_path).read_text())
        from r1_report import report

        report(config, ROOT / config["output"], error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        config_path = (
            sys.argv[sys.argv.index("--config") + 1]
            if "--config" in sys.argv
            else "experiments/history_benefits/r1_pipeline_check/config.yaml"
        )
        config = yaml.safe_load((ROOT / config_path).read_text())
        from r1_report import report

        report(config, ROOT / config["output"])

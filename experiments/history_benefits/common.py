"""Shared, deterministic primitives for the history engineering pilot."""

from __future__ import annotations
from typing import Any
import copy
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from flash_ansr import FlashANSR, SoftmaxSamplingConfig
from flash_ansr.data.streaming import WorkerConfig, build_training_sample
from flash_ansr.data.serialization import serialize_constant_tokens
from flash_ansr.model.flash_ansr_model import FlashANSRModel
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.skeleton import mask_literals_positional
from flash_ansr.train.train import _z_loss
from symbolic_data import ProblemSource
from symbolic_data.token_ops import tagged_canonical

ROOT = Path(__file__).resolve().parents[2]


def dump(path: Any, value: Any) -> Any:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, np.ndarray):
            return clean(v.tolist())
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2))
    tmp.replace(path)


def seed_all(seed: Any) -> Any:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha(path: Any) -> Any:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


#: Config keys of the per-split problem seeds. ProblemSource ignores ``sampling.seed`` and draws
#: from OS entropy unless it is handed an ``rng``; these keys are what makes problems reproducible.
DATA_SEED_KEYS = {"train": "data_seed_train", "validation": "data_seed_validation"}


def split_seeds(cfg: Any) -> bool:
    """True when the config pins one problem stream per split (the formal-rules layout)."""
    present = [key in cfg for key in DATA_SEED_KEYS.values()]
    if any(present) and not all(present):
        raise ValueError(f"set both or neither of {sorted(DATA_SEED_KEYS.values())}")
    return all(present)


#: Validation problem ids start here in split-seed configs, so a validation problem keeps its id
#: (and its pool file) when the number of training problems grows.
VALIDATION_ID_OFFSET = 100_000


def problem_ids(cfg: Any, split: Any, count: Any) -> list[int]:
    """Ids of the first ``count`` problems of ``split``."""
    if split == "train":
        return list(range(int(count)))
    offset = VALIDATION_ID_OFFSET if split_seeds(cfg) else int(cfg["train_problems"])
    return list(range(offset, offset + int(count)))


def fit_seed(cfg: Any, pid: Any) -> int:
    """Constant-fitting seed of one problem (pool build and evaluation refits): ``fit_seed + pid``.
    Legacy configs keep ``seed + pid``."""
    return int(cfg.get("fit_seed", cfg["seed"])) + int(pid)


def train_seed(cfg: Any) -> int:
    """Training seed: problem picks, history on/off, history choice, condition dropout."""
    return int(cfg.get("train_seed", cfg["seed"]))


def eval_seed(cfg: Any, pid: Any) -> int:
    """Evaluation seed of one problem: generation sampling, history choice, validation samples."""
    return int(cfg.get("eval_seed", cfg["seed"])) + int(pid)


def source(cfg: Any, split: Any = None) -> Any:
    """Problem source and recipe.

    With per-split data seeds the source of ``split`` is seeded by its own generator, so the
    i-th problem of a split depends only on that split's seed. Legacy configs (one ``seed``)
    keep the old unseeded single stream, which is NOT reproducible: problems must be read from
    their saved files.
    """
    recipe = yaml.safe_load((ROOT / cfg["recipe"]).read_text())
    sc = copy.deepcopy(recipe["source"])
    sc["catalog"] = str((ROOT / cfg["recipe"]).parent / sc["catalog"])
    sc["sampling"]["seed"] = cfg["seed"]
    rng = None
    if split_seeds(cfg) and split is not None:
        rng = np.random.default_rng(int(cfg[DATA_SEED_KEYS[split]]))
        if split == "validation" and cfg.get("validation_sampling"):
            # Evaluation problems follow srbf's protocol (noiseless, a held-out split): same
            # catalog as training, only the sampling block is replaced.
            sc["sampling"] = {**dict(cfg["validation_sampling"]), "seed": cfg["seed"]}
    return ProblemSource(sc, rng=rng), recipe


def load_base(cfg: Any) -> Any:
    return FlashANSR.load(
        str(ROOT / cfg["model"]),
        generation_config=SoftmaxSamplingConfig(
            draws=cfg["draws"], emission="fittable"
        ),
        refine={"n_restarts": cfg["restarts"]},
        ranking="mdl",
        compute={"device": "cpu", "workers": cfg["workers"]},
    )


def migrate(cfg: Any, base: Any) -> Any:
    seed_all(cfg["seed"])
    mc = load_config(str(ROOT / cfg["model"] / "model.yaml"))
    tc = load_config(str(ROOT / cfg["model"] / "tokenizer.yaml"))
    # Append to the final vocabulary without shifting ANY of the pretrained IDs.
    tc["variables"] = [*tc["variables"], "<history>", "</history>"]
    mc.update(
        tokenizer=tc,
        history_slots=1,
        encoder_mask_query_norms=True,
        sanitize_input_num=True,
    )
    m = FlashANSRModel.from_config(mc)
    old, state = base.state_dict(), m.state_dict()
    for key, dst in state.items():
        src = old[key]
        if dst.shape == src.shape:
            dst.copy_(src)
        elif key == "encoder.embedding.weight":
            dst.zero_()
            dst[:, : src.shape[1]].copy_(src)
        elif dst.shape[0] == src.shape[0] + 2 and dst.shape[1:] == src.shape[1:]:
            dst[: src.shape[0]].copy_(src)
            if key.startswith("next_token_head."):
                # A fresh Linear row has a completely different logit scale from this
                # pretrained head. Start new output classes at its learned mean;
                # input embeddings still use the fixed random seed.
                dst[src.shape[0]:].copy_(
                    src.mean(dim=0, keepdim=True).expand_as(dst[src.shape[0]:])
                )
        else:
            raise ValueError(f"Unexpected migration: {key}: {src.shape} -> {dst.shape}")
    m.load_state_dict(state)
    assert m.tokenizer.vocab[: len(base.tokenizer)] == base.tokenizer.vocab
    return m, mc


def worker_config(cfg: Any, model: Any, recipe: Any) -> Any:
    return WorkerConfig(
        source_config={},
        tokenizer=model.tokenizer,
        padding="zero",
        n_per_equation=1,
        batch_size=1,
        tokenizer_oov="raise",
        worker_preprocess=False,
        max_seq_len=cfg["max_length"],
        preprocessor_prompt_config=None,
        unconditional_prob=recipe["condition_dropout"],
        target_dialect=recipe["target_dialect"],
        **{
            k: recipe.get(k)
            for k in (
                "complexity_block",
                "mask_block",
                "predict_y_block",
                "residual_block",
            )
        },
    )


def history_ids(model: Any, candidate: Any) -> Any:
    expr = tagged_canonical(model.simplipy_engine, candidate["expression_prefix"])
    skel, vals = mask_literals_positional(
        model.simplipy_engine, expr, keep_specials=True
    )
    tokens, _ = serialize_constant_tokens(skel, vals)
    return model.tokenizer.encode(["<history>", *tokens, "</history>"])


def read_problem(directory: Any, pid: Any) -> Any:
    directory = Path(directory)
    row = json.loads((directory / "problems" / f"{pid}.json").read_text())
    with np.load(directory / "problems" / f"{pid}.npz") as a:
        p = SimpleNamespace(
            skeleton=row["skeleton"],
            constants=row["constants"],
            expression=row["expression"],
            variables=row["variables"],
            noise=row["noise"],
            is_placeholder=False,
            x_support=a["x"].copy(),
            y_support=a["clean"].copy(),
            y_support_noisy=a["y"].copy(),
            outlier_mask_support=a["outlier"].copy(),
            x_validation=a["x_validation"].copy() if "x_validation" in a else None,
            y_validation=a["y_validation"].copy() if "y_validation" in a else None,
        )
    return p


def read_pool(directory: Any, pid: Any, kind: Any) -> Any:
    path = Path(directory) / "pools" / kind / f"{pid}.json"
    row = json.loads(path.read_text())
    with np.load(path.with_suffix(".npz")) as a:
        yh = a["predictions"].copy()
    return row["candidates"], yh


def assemble(
    cfg: Any,
    model: Any,
    wc: Any,
    catalog: Any,
    problem: Any,
    pool: Any,
    predictions: Any,
    seed: Any,
    group: Any,
    validation: Any = False,
) -> Any:
    rng = np.random.default_rng(seed)
    want = group != "A" and bool(rng.random() < 0.5)
    chosen = (
        int(rng.integers(min(cfg["history_top"], len(pool)))) if want and pool else None
    )
    ids = history_ids(model, pool[chosen]) if chosen is not None else []
    sample = build_training_sample(
        problem,
        replace(wc, unconditional_prob=0 if validation else wc.unconditional_prob),
        catalog,
        rng,
        noise_spec=True,
        strict_length=True,
        history_ids=ids,
    )
    if sample is None or "rejected" in sample:
        raise ValueError(f"Unassemblable target: {sample}")
    valid = sample["metadata"]["history_valid"]
    sample["history"] = (
        predictions[chosen][sample["row_indices"]]
        if valid
        else np.zeros(len(sample["row_indices"]))
    )
    sample["planned_history"] = want
    sample["chosen"] = chosen if valid else None
    if want and not pool:
        sample["metadata"]["history_reason"] = "empty_pool"
    return sample


def collate(samples: Any, tokenizer: Any) -> Any:
    b = len(samples)
    n = max(len(s["x_support"]) for s in samples)
    sequence_length = max(len(s["input_ids"]) for s in samples)
    data = torch.zeros(b, n, 19, dtype=torch.float64)
    mask = torch.zeros(b, n, dtype=torch.bool)
    ids = torch.full((b, sequence_length), tokenizer["<pad>"], dtype=torch.long)
    nums = torch.full((b, sequence_length, 1), float("nan"), dtype=torch.float64)
    ignore = torch.ones(b, sequence_length, dtype=torch.bool)
    outliers = torch.zeros(b, n)
    for i, s in enumerate(samples):
        m = len(s["x_support"])
        t = len(s["input_ids"])
        md = s["metadata"]
        data[i, :m, :17] = torch.as_tensor(s["x_support"])
        data[i, :m, 17] = torch.as_tensor(s["y_encoder"]).flatten()
        data[i, :m, 18] = torch.as_tensor(s["history"])
        mask[i, :m] = True
        ids[i, :t] = torch.tensor(s["input_ids"])
        nums[i, :t, 0] = torch.tensor(md["input_num"], dtype=torch.float64)
        ignore[i, :t] = torch.tensor(md.get("task_mask", [False] * t))
        if s["outlier_mask"] is not None:
            outliers[i, :m] = torch.as_tensor(s["outlier_mask"]).flatten()
    labels = ids[:, 1:].clone()
    labels[ignore[:, 1:] | (labels == tokenizer["<float>"])] = tokenizer["<pad>"]
    return dict(
        input_tokens=ids,
        data=data,
        input_num=nums,
        data_attn_mask=mask,
        history_mask=torch.tensor([s["metadata"]["history_valid"] for s in samples]),
        condition_mask=torch.tensor(
            [s["metadata"].get("condition_mask", True) for s in samples]
        ),
        labels=labels,
        outliers=outliers,
    )


def loss(model: Any, batch: Any) -> Any:
    labels = batch["labels"]
    valid = labels != model.tokenizer["<pad>"]
    logits = model(
        **{k: v for k, v in batch.items() if k not in ("labels", "outliers")}
    )[:, :-1]
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.flatten(),
        ignore_index=model.tokenizer["<pad>"],
        reduction="sum",
    ) / valid.sum().clamp_min(1)
    ol = model.outlier_head(model.point_representations).squeeze(-1)
    pm = batch["data_attn_mask"]
    bce = (
        F.binary_cross_entropy_with_logits(ol, batch["outliers"], reduction="none") * pm
    ).sum() / pm.sum()
    z = _z_loss(logits.reshape(-1, logits.shape[-1]), valid.flatten())
    total = ce + 0.05 * bce + 1e-4 * z
    return total, {
        "loss": total.item(),
        "ce": ce.item(),
        "outlier": bce.item(),
        "z": z.item(),
    }

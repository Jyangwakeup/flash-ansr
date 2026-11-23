"""History input ablation and copy probe of round 4 (设计与预注册.md §14). Forward passes only.

Ablation: every history sample of the fixed evaluation set (r4_samples.eval_set) is scored
under five conditions that differ only in the history:

    none         no history (same problem, same rows, same target)
    full         formula + its predictions
    tokens_only  formula, predictions column zeroed
    mismatched   formula, predictions of the next sample of the same kind (another problem)
    perfect      formula, predictions column = the y the model sees (looks like a perfect fit)

Copied in spirit from round 3's ablate.py; round 4 keeps every sample's loss so the report can
pair by problem and give bootstrap intervals.

Copy probe: with the exact formula as history, teacher-forced top-1 accuracy on the expression
tokens (structure tokens and constant bytes separately) and how many expressions are entirely
right -- versus the same sample without history.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from common import collate
from r4_samples import target_ces

CONDITIONS = ("none", "full", "tokens_only", "mismatched", "perfect")


def fit_column(column: Any, n: int) -> np.ndarray:
    """``column`` resized to ``n`` rows (tiled / cut), for the mismatched condition."""
    return np.resize(np.asarray(column, dtype=float), n)


def conditions(row: dict, other: dict) -> dict:
    full = row["full"]
    n = len(full["row_indices"])
    out = {"none": row["base"], "full": full}
    for name in ("tokens_only", "mismatched", "perfect"):
        s = dict(full)
        if name == "tokens_only":
            s["history"] = np.zeros(n)
        elif name == "mismatched":
            s["history"] = fit_column(other["full"]["history"], n)
        else:
            s["history"] = np.asarray(full["y_encoder"], dtype=float).reshape(-1)
        out[name] = s
    return out


def ablate(model: Any, rows: list, micro_batch: int) -> list:
    """Per history sample: pid, kind, bin, level, fvu and the target CE under every condition."""
    hist = [r for r in rows if r["kind"] != "none"]
    variants = []
    for i, r in enumerate(hist):
        same = [j for j in range(len(hist)) if hist[j]["kind"] == r["kind"] and hist[j]["pid"] != r["pid"]]
        later = [j for j in same if j > i]
        other = hist[(later or same or [i])[0]]  # the next sample of the same kind from another problem
        variants.append(conditions(r, other))
    ces = {c: target_ces(model, [v[c] for v in variants], micro_batch) for c in CONDITIONS}
    return [dict(pid=r["pid"], kind=r["kind"], bin=r["bin"], target_bin=r.get("target_bin"), level=r["level"],
                 fvu=r["fvu"], ce={c: ces[c][i] for c in CONDITIONS}) for i, r in enumerate(hist)]


def copy_probe(model: Any, pairs: list) -> dict:
    """``pairs``: (sample with the exact formula as history, the same sample without). Teacher
    forcing over the tokens after the last ``<expression>`` up to its ``</expression>``."""
    tok = model.tokenizer
    vocab = list(tok.vocab)
    e0, e1, pad = tok["<expression>"], tok["</expression>"], tok["<pad>"]

    def is_byte(t: int) -> bool:
        word = vocab[t]
        return word in ("<ieee754>", "</ieee754>") or (word.startswith("<b") and word != "<bos>")

    stats: dict[str, dict[str, Any]] = {
        k: dict(n=0, struct=[], byte=[], ce=[], whole=0, whole_struct=0) for k in ("exact", "none")}
    for full, base in pairs:
        for name, s in (("exact", full), ("none", base)):
            b = collate([s], tok)
            with torch.no_grad():
                logits = model(**{k: v for k, v in b.items() if k not in ("labels", "outliers")})[:, :-1][0]
            lab, ids = b["labels"][0], b["input_tokens"][0][1:]
            starts, ends = (ids == e0).nonzero(), (ids == e1).nonzero()
            if not len(starts) or not len(ends):
                continue
            span = torch.arange(int(starts[-1]) + 1, int(ends[-1]) + 1)
            span = span[lab[span] != pad]
            hit = logits[span].argmax(-1) == lab[span]
            byte = torch.tensor([is_byte(int(t)) for t in lab[span]], dtype=torch.bool)
            st = stats[name]
            st["n"] += 1
            st["struct"] += hit[~byte].tolist()
            st["byte"] += hit[byte].tolist()
            st["ce"] += F.cross_entropy(logits[span], lab[span], reduction="none").tolist()
            st["whole"] += int(hit.all())
            st["whole_struct"] += int(hit[~byte].all())
    return {k: dict(samples=st["n"], structure_tokens=len(st["struct"]),
                    structure_acc=float(np.mean(st["struct"])) if st["struct"] else float("nan"),
                    byte_tokens=len(st["byte"]), byte_acc=float(np.mean(st["byte"])) if st["byte"] else float("nan"),
                    expression_ce=float(np.mean(st["ce"])) if st["ce"] else float("nan"),
                    whole_structure_right=st["whole_struct"], whole_expression_right=st["whole"])
            for k, st in stats.items()}

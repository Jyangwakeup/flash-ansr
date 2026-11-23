"""Acceptance gates for history conditioning; fail before scaling on violations."""

from typing import Any
import numpy as np
import torch
from flash_ansr.inference import evaluate_prefix
from common import assemble, collate, loss, read_pool, read_problem, dump


def check(cfg: Any, out: Any, model: Any, wc: Any, catalog: Any, ids: Any) -> Any:
    samples = []
    for pid in ids:
        p = read_problem(out, pid)
        pool, yh = read_pool(out, pid, "random")
        for kind in ("random", "model"):
            rows, predictions = read_pool(out, pid, kind)
            for row, pred in zip(rows, predictions):
                recomputed = evaluate_prefix(
                    model.simplipy_engine, row["expression_prefix"], p.x_support
                ).reshape(-1)
                np.testing.assert_allclose(recomputed, pred, rtol=1e-10, atol=1e-10)
        for seed in range(30):
            s = assemble(cfg, model, wc, catalog, p, pool, yh, seed, "B")
            if s["metadata"]["history_valid"]:
                samples.append(s)
                break
    if not samples:
        raise AssertionError("No valid K=1 samples: cannot pass history gate")
    b = collate(samples[:2], model.tokenizer)
    model.eval()
    args = {k: v for k, v in b.items() if k not in ("labels", "outliers")}
    with torch.no_grad():
        logits = model(**args).clone()
        changed = {**args, "input_tokens": args["input_tokens"].clone()}
        t = changed["input_tokens"].shape[1] // 2
        changed["input_tokens"][:, t] = model.tokenizer["x1"]
        later = model(**changed)
        torch.testing.assert_close(logits[:, :t], later[:, :t], rtol=1e-5, atol=1e-6)
        absent = {**args, "history_mask": torch.zeros_like(args["history_mask"])}
        first = model(**absent).clone()
        altered = absent["data"].clone()
        altered[:, :, -1] = 123456.0
        second = model(**{**absent, "data": altered})
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    for s in samples:
        ids_ = s["input_ids"]
        md = s["metadata"]
        start = ids_.index(model.tokenizer["<history>"])
        end = ids_.index(model.tokenizer["</history>"])
        assert all(md["task_mask"][start: end + 1])
        expr = ids_.index(model.tokenizer["<expression>"])
        assert not md["task_mask"][expr + 1]
        extracted = model.tokenizer.extract_expression_from_beam(ids_)[0]
        assert model.tokenizer["<history>"] not in extracted
    model.zero_grad()
    total, _ = loss(model, b)
    total.backward()
    grad = model.encoder.embedding.weight.grad[:, -64:]
    assert torch.isfinite(total) and torch.isfinite(grad).all() and grad.abs().max() > 0
    # New columns start at zero: test dependency after a reversible small weight change.
    saved = model.encoder.embedding.weight.detach().clone()
    with torch.no_grad():
        model.encoder.embedding.weight[:, -64:] += 0.001
        a = model(**args).clone()
        other = args["data"].clone()
        other[:, :, -1] += 3.25
        z = model(**{**args, "data": other})
        delta = (a - z).abs().max().item()
        model.encoder.embedding.weight.copy_(saved)
    assert delta > 0
    model.zero_grad()
    report = dict(
        valid_history_samples=len(samples),
        causal_mask=True,
        missing_history_invariant=True,
        history_loss_mask=True,
        target_extraction=True,
        history_gradient=True,
        numeric_history_delta=delta,
        stored_predictions_match=True,
    )
    dump(out / "acceptance.json", report)
    return report

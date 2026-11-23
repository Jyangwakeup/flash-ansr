"""Visualize the completed baseline without modifying raw results."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after matplotlib.use)
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/baseline_scaling/seed0"
rows = [json.loads(s) for s in (OUT / "baseline.jsonl").read_text().splitlines() if s.strip()]
grid = sorted({r["draws"] for r in rows})
assert {(r["problem"], r["draws"], r["seed"]) for r in rows} == {(p, n, 0) for p in range(120) for n in grid}
assert len(rows) == 120 * len(grid)
summary = json.loads((OUT / "summary.json").read_text())
plt.rcParams.update(
    {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"}
)
fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
x = np.arange(len(grid))
for key, label, color, marker in [
    ("solved_oracle", "Oracle: any candidate", "#2563eb", "o"),
    ("solved_top1", "Top-1: returned answer", "#ea580c", "x"),
]:
    vals = [100 * summary["per_N"][str(n)][key]["rate"] for n in grid]
    axs[0, 0].plot(grid, vals, marker=marker, label=label, color=color, linewidth=2)
for n in grid:
    e = summary["per_N"][str(n)]["solved_oracle"]
    y = 100 * e["rate"]
    lo, hi = np.array(e["wilson95"]) * 100
    axs[0, 0].errorbar(n, y, yerr=[[y - lo], [hi - y]], color="#2563eb", capsize=4, alpha=0.45)
    axs[0, 0].annotate(f"{y:.1f}%", (n, y), xytext=(0, 12 if n != 700 else 28), textcoords="offset points", ha="center")
axs[0, 0].set(
    xscale="log",
    ylim=(0, 70),
    xlabel="Draw budget N (log scale)",
    ylabel="Solved problems (%)",
    title="A. Solve rate (120 problems; 95% Wilson bars)",
)
axs[0, 0].set_xticks(grid, labels=grid)
axs[0, 0].legend(loc="lower right")
gained = [summary["adjacent_mcnemar"][f"{a}->{b}"]["gained"] for a, b in zip(grid, grid[1:])]
lost = [summary["adjacent_mcnemar"][f"{a}->{b}"]["lost"] for a, b in zip(grid, grid[1:])]
a = axs[0, 1]
a.bar(np.arange(len(grid) - 1) - 0.17, gained, 0.34, label="Newly solved", color="#16a34a")
a.bar(np.arange(len(grid) - 1) + 0.17, lost, 0.34, label="Previously solved, now missed", color="#dc2626")
a.set_xticks(range(len(grid) - 1), labels=[f"{a} → {b}" for a, b in zip(grid, grid[1:])], rotation=20)
a.set(title="B. Paired changes in oracle hits", ylabel="Number of problems", ylim=(0, 15))
a.legend(fontsize=9)
for i, (g, l) in enumerate(zip(gained, lost)):
    a.text(i - 0.17, g + 0.25, str(g), ha="center")
    a.text(i + 0.17, l + 0.25, str(l), ha="center")
med = [np.median([r["fit_time"] for r in rows if r["draws"] == n]) for n in grid]
a = axs[1, 0]
a.plot(grid, med, "o-", color="#7c3aed")
a.set(
    xscale="log",
    yscale="log",
    xlabel="Draw budget N",
    ylabel="Median fit time per problem (seconds)",
    title="C. Median runtime (generation + fitting)",
)
a.set_xticks(grid, labels=grid)
for n, t in zip(grid, med):
    a.annotate(f"{t:.1f}s", (n, t), xytext=(0, 9), textcoords="offset points", ha="center")
a.margins(y=0.2)
a = axs[1, 1]
unique = [100 * summary["per_N"][str(n)]["mean_unique"] / n for n in grid]
a.bar(x, unique, color="#0891b2")
a.set_xticks(x, labels=grid)
a.set(
    ylim=(0, 100),
    xlabel="Draw budget N",
    ylabel="Unique token sequences / requested draws (%)",
    title="D. Recorded sequence diversity (not structural diversity)",
)
for i, v in enumerate(unique):
    a.text(i, v + 1, f"{v:.1f}%", ha="center")
for a in axs.flat:
    a.grid(axis="y", alpha=0.18)
fig.suptitle("Flash-ANSR independent-sampling baseline | fastsrb | one fit seed", fontsize=17)
fig.savefig(OUT / "baseline_dashboard.png", dpi=180)
fig.savefig(OUT / "baseline_dashboard.svg")
plt.close(fig)
lookup = {(r["problem"], r["draws"]): r for r in rows}
mat = np.array(
    [[int(lookup[p, n]["solved_oracle"]) + int(lookup[p, n]["solved_top1"]) for n in grid] for p in range(120)]
)
fig, axs = plt.subplots(1, 4, figsize=(13, 9), constrained_layout=True)
for j, a in enumerate(axs):
    start = j * 30
    a.imshow(
        mat[start:start + 30], aspect="auto", cmap=ListedColormap(["#e2e8f0", "#f59e0b", "#2563eb"]), vmin=0, vmax=2
    )
    a.set_xticks(range(len(grid)), labels=grid, rotation=45)
    a.set_yticks(range(30), labels=range(start, start + 30), fontsize=8)
    a.set(xlabel="Draw budget N", ylabel="Problem ID (original order)")
fig.suptitle("Per-problem outcomes | gray: missed; amber: oracle only; blue: top-1 solved", fontsize=14)
fig.savefig(OUT / "problem_outcomes.png", dpi=180)
plt.close(fig)
print("Saved dashboard PNG/SVG and problem_outcomes.png")

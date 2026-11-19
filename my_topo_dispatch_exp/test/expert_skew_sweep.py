#!/usr/bin/env python3

import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path("/mnt/workspace/my_topo_dispatch_exp")
OUT_DIR = ROOT / "test" / "report_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BS = 512
NUM_EXPERTS = 64
K = 8

ALPHAS = [0.0, 0.5, 1.0, 1.5]

BASE_SEED = 1


def make_expert_ids(bs, num_classes, k, alpha, seed):
    torch.manual_seed(int(seed))

    ranks = torch.arange(
        1,
        num_classes + 1,
        dtype=torch.float32,
    )

    probs = 1.0 / (ranks ** alpha)
    probs = probs / probs.sum()

    return torch.multinomial(
        probs.expand(bs, -1),
        num_samples=k,
        replacement=False,
    )


def count_experts(expert_ids, num_experts):
    return torch.bincount(
        expert_ids.reshape(-1),
        minlength=num_experts,
    ).cpu().numpy().astype(np.int64)


results = {}

for alpha in ALPHAS:

    rank0_ids = make_expert_ids(
        BS,
        NUM_EXPERTS,
        K,
        alpha,
        BASE_SEED + 0 + 17,
    )

    rank1_ids = make_expert_ids(
        BS,
        NUM_EXPERTS,
        K,
        alpha,
        BASE_SEED + 1 + 17,
    )

    rank0 = count_experts(rank0_ids, NUM_EXPERTS)
    rank1 = count_experts(rank1_ids, NUM_EXPERTS)

    total = rank0 + rank1

    assert total.sum() == 2 * BS * K

    mean = float(total.mean())
    std = float(total.std())

    results[str(alpha)] = {
        "alpha": alpha,
        "rank0": rank0.tolist(),
        "rank1": rank1.tolist(),
        "total": total.tolist(),
        "total_routes": int(total.sum()),
        "mean": mean,
        "std": std,
        "min": int(total.min()),
        "max": int(total.max()),
        "max_mean": float(total.max() / mean),
        "cv": float(std / mean),
    }


json_file = OUT_DIR / "expert_skew_sweep.json"

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2,
    )


print("=" * 80)
print("Expert routing skew sweep")
print("=" * 80)
print(f"BS          = {BS}")
print(f"Experts     = {NUM_EXPERTS}")
print(f"Top-K       = {K}")
print(f"Routes      = {2 * BS * K}")
print()

print(
    f"{'alpha':>8} "
    f"{'min':>8} "
    f"{'max':>8} "
    f"{'mean':>10} "
    f"{'max/mean':>12} "
    f"{'CV':>10}"
)

print("-" * 65)

for alpha in ALPHAS:
    r = results[str(alpha)]

    print(
        f"{alpha:>8.1f} "
        f"{r['min']:>8d} "
        f"{r['max']:>8d} "
        f"{r['mean']:>10.3f} "
        f"{r['max_mean']:>12.3f} "
        f"{r['cv']:>10.3f}"
    )


# ============================================================
# 图 1：不同 α 下的 Max/Mean
# ============================================================

x = np.array(ALPHAS)

max_mean = np.array([
    results[str(a)]["max_mean"]
    for a in ALPHAS
])

cv = np.array([
    results[str(a)]["cv"]
    for a in ALPHAS
])


fig, ax = plt.subplots(
    figsize=(8, 5),
    dpi=200,
)

ax.plot(
    x,
    max_mean,
    marker="o",
    linewidth=2,
    markersize=7,
    color="#E45756",
)

for a, y in zip(x, max_mean):
    ax.text(
        a,
        y,
        f"{y:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_xlabel("Routing Skew Parameter α")
ax.set_ylabel("Max Expert Load / Mean Expert Load")
ax.set_title(
    "Expert Load Imbalance Under Different Routing Skews"
)

ax.grid(
    linestyle="--",
    alpha=0.3,
)

fig.tight_layout()

png1 = OUT_DIR / "expert_load_imbalance_vs_alpha.png"

fig.savefig(
    png1,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 图 2：CV
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5),
    dpi=200,
)

ax.plot(
    x,
    cv,
    marker="s",
    linewidth=2,
    markersize=7,
    color="#4C78A8",
)

for a, y in zip(x, cv):
    ax.text(
        a,
        y,
        f"{y:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_xlabel("Routing Skew Parameter α")
ax.set_ylabel("Coefficient of Variation (CV)")
ax.set_title(
    "Expert Load Variability Under Different Routing Skews"
)

ax.grid(
    linestyle="--",
    alpha=0.3,
)

fig.tight_layout()

png2 = OUT_DIR / "expert_load_cv_vs_alpha.png"

fig.savefig(
    png2,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("=" * 80)
print("Generated files")
print("=" * 80)
print(json_file)
print(png1)
print(png2)

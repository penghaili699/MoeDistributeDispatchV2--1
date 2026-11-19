#!/usr/bin/env python3

import os
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/workspace/my_topo_dispatch_exp")
OUT_DIR = ROOT / "test" / "report_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 这里采用你 golden.py 中已经使用的路由参数
# ============================================================

# 从你刚才实际 NPU 报错中确认：
# maxBs=512
# epWorldSize=2
# localMoeExpertNum=32
# k=8
#
# 因此：
# 总 Expert = 2 * 32 = 64
#
# 但这里的 expert_id 是在每个 rank 的输入中生成的，
# num_classes 对应 expert_ids.shape[1]。
#
# 如果你的实际 case shape 与下面不同，
# 请以实际 case 为准。

BS = 512
NUM_EXPERTS = 64
K = 8

ALPHAS = [0.0, 1.5]

# 与 golden.py 中的 seed 规则保持一致：
# seed = attrs["seed"] + ctx["rank"]
# expert_ids 使用 seed + 17
BASE_SEED = 1


def make_expert_ids(bs, num_classes, k, alpha, seed):
    """
    与 golden.py 中 _make_tensor() 的 expert_ids
    GRAPH_SKEW_ALPHA 分支保持一致。

    golden.py 的核心逻辑：

        ranks = torch.arange(1, num_classes + 1)
        probs = 1.0 / (ranks ** alpha)
        probs = probs / probs.sum()
        sampled = torch.multinomial(
            probs.expand(bs, -1),
            num_samples=k,
            replacement=False
        )
    """

    torch.manual_seed(int(seed))

    ranks = torch.arange(
        1,
        num_classes + 1,
        dtype=torch.float32,
    )

    probs = 1.0 / (ranks ** alpha)
    probs = probs / probs.sum()

    sampled = torch.multinomial(
        probs.expand(bs, -1),
        num_samples=k,
        replacement=False,
    )

    return sampled


def count_experts(expert_ids, num_experts):
    """
    每出现一次 expert_id，就表示一个 Token-Expert 路由关系。

    因为 Top-K=8，
    每个 Token 会产生 8 个 expert_id。

    所以总计数应该为：

        BS * K
    """

    flat = expert_ids.reshape(-1)

    counts = torch.bincount(
        flat,
        minlength=num_experts,
    )

    return counts.cpu().numpy().astype(np.int64)


def main():

    all_results = {}

    print("=" * 80)
    print("Expert Token Distribution Experiment")
    print("=" * 80)
    print(f"BS            = {BS}")
    print(f"NUM_EXPERTS   = {NUM_EXPERTS}")
    print(f"Top-K         = {K}")
    print(f"Total routes  = {BS * K}")
    print()

    for alpha in ALPHAS:

        print("-" * 80)
        print(f"GRAPH_SKEW_ALPHA = {alpha}")
        print("-" * 80)

        # rank 0
        expert_ids_rank0 = make_expert_ids(
            BS,
            NUM_EXPERTS,
            K,
            alpha,
            BASE_SEED + 0 + 17,
        )

        # rank 1
        expert_ids_rank1 = make_expert_ids(
            BS,
            NUM_EXPERTS,
            K,
            alpha,
            BASE_SEED + 1 + 17,
        )

        counts_rank0 = count_experts(
            expert_ids_rank0,
            NUM_EXPERTS,
        )

        counts_rank1 = count_experts(
            expert_ids_rank1,
            NUM_EXPERTS,
        )

        # 两个 Rank 的总体 Expert 接收量
        total_counts = counts_rank0 + counts_rank1

        assert int(counts_rank0.sum()) == BS * K
        assert int(counts_rank1.sum()) == BS * K
        assert int(total_counts.sum()) == 2 * BS * K

        result = {
            "alpha": alpha,
            "bs_per_rank": BS,
            "num_experts": NUM_EXPERTS,
            "top_k": K,
            "total_routes_per_rank": int(BS * K),
            "rank_0": counts_rank0.tolist(),
            "rank_1": counts_rank1.tolist(),
            "total": total_counts.tolist(),
            "total_routes": int(total_counts.sum()),
            "mean": float(total_counts.mean()),
            "min": int(total_counts.min()),
            "max": int(total_counts.max()),
            "max_mean_ratio": float(
                total_counts.max() / total_counts.mean()
            ),
            "std": float(total_counts.std()),
            "cv": float(
                total_counts.std() / total_counts.mean()
            ),
        }

        all_results[str(alpha)] = result

        print("Rank 0:")
        print(counts_rank0.tolist())

        print()
        print("Rank 1:")
        print(counts_rank1.tolist())

        print()
        print("Total:")
        print(total_counts.tolist())

        print()
        print(f"Total routes = {int(total_counts.sum())}")
        print(f"Mean         = {total_counts.mean():.3f}")
        print(f"Min          = {total_counts.min()}")
        print(f"Max          = {total_counts.max()}")
        print(
            f"Max / Mean  = "
            f"{total_counts.max() / total_counts.mean():.3f}"
        )
        print(
            f"CV           = "
            f"{total_counts.std() / total_counts.mean():.3f}"
        )

    # --------------------------------------------------------
    # 保存 JSON
    # --------------------------------------------------------

    json_file = OUT_DIR / "expert_token_distribution.json"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(
            all_results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # 画图
    # --------------------------------------------------------

    alpha0 = np.asarray(
        all_results["0.0"]["total"],
        dtype=np.int64,
    )

    alpha15 = np.asarray(
        all_results["1.5"]["total"],
        dtype=np.int64,
    )

    experts = np.arange(NUM_EXPERTS)

    width = 0.38

    fig, ax = plt.subplots(
        figsize=(15, 6),
        dpi=200,
    )

    ax.bar(
        experts - width / 2,
        alpha0,
        width,
        label=r"$\alpha=0.0$",
        color="#4C78A8",
    )

    ax.bar(
        experts + width / 2,
        alpha15,
        width,
        label=r"$\alpha=1.5$",
        color="#E45756",
    )

    ax.set_xlabel(
        "Expert ID",
        fontsize=12,
    )

    ax.set_ylabel(
        "Received Token Count",
        fontsize=12,
    )

    ax.set_title(
        "Token Distribution Across Experts Under Different Routing Skews",
        fontsize=13,
    )

    ax.set_xticks(experts)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    ax.legend()

    plt.tight_layout()

    png_file = OUT_DIR / "expert_token_distribution.png"

    plt.savefig(
        png_file,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("=" * 80)
    print("Finished")
    print("=" * 80)
    print(f"JSON : {json_file}")
    print(f"PNG  : {png_file}")


if __name__ == "__main__":
    main()

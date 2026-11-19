import csv, json

rows = [
    # Case 1: 小规模均匀负载 - 扁平模式 (comm_mode=0)
    {
        "case_id": 1, "bs": 32, "h": 7168, "k": 4, "dtype": "bfloat16",
        "moe_expert_num": 16, "comm_mode": 0, "note": "2npu_uniform_flat_small"
    },
    # Case 2: 小规模均匀负载 - 分层模式 (comm_mode=2)
    {
        "case_id": 2, "bs": 32, "h": 7168, "k": 4, "dtype": "bfloat16",
        "moe_expert_num": 16, "comm_mode": 2, "note": "2npu_uniform_hier_small"
    },
    # Case 3: 大规模高吞吐负载 - 扁平模式 (comm_mode=0)
    {
        "case_id": 3, "bs": 512, "h": 7168, "k": 8, "dtype": "bfloat16",
        "moe_expert_num": 64, "comm_mode": 0, "note": "2npu_heavy_flat_large"
    },
    # Case 4: 大规模高吞吐负载 - 分层模式 (comm_mode=2)
    {
        "case_id": 4, "bs": 512, "h": 7168, "k": 8, "dtype": "bfloat16",
        "moe_expert_num": 64, "comm_mode": 2, "note": "2npu_heavy_hier_large"
    },
]

with open("cases.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["operator", "case_id", "input_shape", "dtype", "attrs", "value_range", "note"])
    for r in rows:
        attrs = {
            "mc2_distributed": True,
            "world_size": 2,
            "tp_world_size": 1,
            "moe_expert_num": r["moe_expert_num"],
            "shared_expert_num": 0,
            "shared_expert_rank_num": 0,
            "quant_mode": 0,
            "seed": 42 + r["case_id"],
            "activeMask_Dim": 0,
            "is_combine_x": 0,
            "comm_mode": r["comm_mode"],
            "is_elastic": 0,
            "zero_expert_num": 0,
            "copy_expert_num": 0,
            "const_expert_num": 0,
        }
        writer.writerow([
            "MoeDistributeDispatchV2",
            r["case_id"],
            json.dumps([[r["bs"], r["h"]], [r["bs"], r["k"]]]),
            json.dumps([r["dtype"], "int32"]),
            json.dumps(attrs),
            json.dumps([[-5, 5], [0, r["moe_expert_num"] - 1]]),
            r["note"]
        ])
print("Generated 4 dual-NPU cases in cases.csv!")

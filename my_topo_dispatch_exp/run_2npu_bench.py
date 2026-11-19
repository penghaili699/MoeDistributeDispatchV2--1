import os
import sys
import time
import json
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

# 加载我们的任务 golden 与候选算子包
sys.path.insert(0, "/mnt/workspace/my_topo_dispatch_exp/my_bench_root/level4/moe_distribute_dispatch_v2")
import golden
import cann_bench

def _worker(rank: int, world_size: int, port: int, cases: list, skew_alpha: float, out_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["GRAPH_SKEW_ALPHA"] = str(skew_alpha)
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"

    torch_npu.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

    # 获取底层 HCCL 通信域名（供 MC2 算子使用）
    pg = dist.group.WORLD
    if hasattr(pg, "_get_backend"):
        hccl_backend = pg._get_backend(device)
        hcomm_info = hccl_backend.get_hccl_comm_name(rank)
    else:
        hcomm_info = pg.get_hccl_comm_name(rank)

    ctx = {
        "rank": rank,
        "world_size": world_size,
        "device": device,
        "torch_npu": torch_npu,
        "hcomm_info": hcomm_info,
    }

    results = []
    for case in cases:
        payload = {
            "input_shapes": case["input_shape"],
            "dtypes": case["dtype"],
            "value_ranges": case["value_range"],
            "attrs": case["attrs"],
        }
        inputs = golden.mc2_make_rank_inputs(ctx, payload)
        attrs = case["attrs"]

        # 1. 运行 Golden 参考算子并测时
        for _ in range(3):
            ref_out = golden.mc2_distributed_golden(ctx, inputs, attrs)
        torch_npu.npu.synchronize()
        dist.barrier()

        t0 = time.perf_counter()
        for _ in range(10):
            ref_out = golden.mc2_distributed_golden(ctx, inputs, attrs)
        torch_npu.npu.synchronize()
        dist.barrier()
        ref_us = (time.perf_counter() - t0) / 10.0 * 1e6

        # 2. 运行我们的候选优化算子并测时
        for _ in range(3):
            cand_out = golden.mc2_call_candidate(cann_bench.moe_distribute_dispatch_v2, ctx, inputs, attrs)
        torch_npu.npu.synchronize()
        dist.barrier()

        t0 = time.perf_counter()
        for _ in range(10):
            cand_out = golden.mc2_call_candidate(cann_bench.moe_distribute_dispatch_v2, ctx, inputs, attrs)
        torch_npu.npu.synchronize()
        dist.barrier()
        cand_us = (time.perf_counter() - t0) / 10.0 * 1e6

        # 3. 精度校验（按实际接收的有效 Token 数量切片对比 expand_x，排除尾部未初始化显存）
        valid_tokens = int(ref_out[3].sum().item()) if len(ref_out) > 3 else cand_out[0].shape[0]
        valid_tokens = max(1, min(valid_tokens, cand_out[0].shape[0]))
        max_diff = float((cand_out[0][:valid_tokens].float() - ref_out[0][:valid_tokens].float()).abs().max().item())
        passed = max_diff < 1e-2

        if rank == 0:
            results.append({
                "case_id": case["case_id"],
                "note": case["note"],
                "passed": passed,
                "max_diff": max_diff,
                "golden_us": round(ref_us, 2),
                "cand_us": round(cand_us, 2),
                "speedup": round(ref_us / max(cand_us, 1e-6), 3),
            })

    if rank == 0:
        out_queue.put(results)
    dist.destroy_process_group()

def run_benchmark(skew_alpha: float):
    with open("/mnt/workspace/my_topo_dispatch_exp/my_bench_root/level4/moe_distribute_dispatch_v2/cases.yaml") as f:
        cases = yaml.safe_load(f)["cases"]

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + int(skew_alpha * 10)
    procs = [ctx.Process(target=_worker, args=(r, 2, port, cases, skew_alpha, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    res = q.get()
    print(f"\n==================== 2-NPU HCCL Benchmark (GRAPH_SKEW_ALPHA = {skew_alpha}) ====================")
    print(f"{'Case':<6} {'Note':<26} {'Pass':<6} {'MaxDiff':<10} {'Golden(us)':<12} {'Cand(us)':<12} {'Speedup':<8}")
    for r in res:
        print(f"{r['case_id']:<6} {r['note']:<26} {str(r['passed']):<6} {r['max_diff']:<10.5f} {r['golden_us']:<12.2f} {r['cand_us']:<12.2f} {r['speedup']:<8.3f}x")

if __name__ == "__main__":
    run_benchmark(0.0)
    run_benchmark(1.5)

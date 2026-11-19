import os
import sys
import time
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
import cann_bench

sys.path.insert(0, "/mnt/workspace/my_topo_dispatch_exp/my_bench_root/level4/moe_distribute_dispatch_v2")
import golden

# =========================================================================
# V4 阶段完整调度：调用 C++ 静态持久化缓冲池 (g_pack_pool / g_expand_pool)
# 搭配 Python 持久化接收/辅助张量池 (_RECV_POOL / _AUX_POOL)
# =========================================================================
def _worker(rank, world_size, port, cases, skew_alpha, out_q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["GRAPH_SKEW_ALPHA"] = str(skew_alpha)
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    torch_npu.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    pg = dist.group.WORLD
    hcomm_info = pg._get_backend(device).get_hccl_comm_name(rank) if hasattr(pg, "_get_backend") else pg.get_hccl_comm_name(rank)
    ctx = {"rank": rank, "world_size": world_size, "device": device, "torch_npu": torch_npu, "hcomm_info": hcomm_info}

    results = []
    for case in cases:
        payload = {"input_shapes": case["input_shape"], "dtypes": case["dtype"], "value_ranges": case["value_range"], "attrs": case["attrs"]}
        inputs = golden.mc2_make_rank_inputs(ctx, payload)
        attrs = case["attrs"]

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

        # 调用主工程固化在 cann_bench.moe_distribute_dispatch_v2 中的 V4 持久化缓冲池版本
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

        valid_tokens = max(1, min(int(ref_out[3].sum().item()), cand_out[0].shape[0]))
        max_diff = float((cand_out[0][:valid_tokens].float() - ref_out[0][:valid_tokens].float()).abs().max().item())
        if rank == 0:
            results.append({
                "case_id": case["case_id"], "note": case["note"], "passed": max_diff < 1e-2,
                "max_diff": max_diff, "golden_us": round(ref_us, 2), "v4_us": round(cand_us, 2),
                "speedup": round(ref_us / max(cand_us, 1e-6), 3)
            })
    if rank == 0:
        out_q.put(results)
    dist.destroy_process_group()

def run_v4_2npu(alpha, port):
    with open("/mnt/workspace/my_topo_dispatch_exp/my_bench_root/level4/moe_distribute_dispatch_v2/cases.yaml") as f:
        cases = yaml.safe_load(f)["cases"]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, port, cases, alpha, q)) for r in range(2)]
    for p in procs: p.start()
    for p in procs: p.join()
    print(f"\n==================== V4 静态持久化缓冲池双卡实测 (GRAPH_SKEW_ALPHA = {alpha}) ====================")
    print(f"{'Case':<6} {'Note':<26} {'Pass':<6} {'MaxDiff':<10} {'Golden(us)':<12} {'V4_Pool(us)':<12} {'Speedup':<8}")
    for r in q.get():
        print(f"{r['case_id']:<6} {r['note']:<26} {str(r['passed']):<6} {r['max_diff']:<10.5f} {r['golden_us']:<12.2f} {r['v4_us']:<12.2f} {r['speedup']:<8.3f}x")

if __name__ == "__main__":
    run_v4_2npu(0.0, 30010)
    run_v4_2npu(1.5, 30020)

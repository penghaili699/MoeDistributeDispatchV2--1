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
# V3 阶段调度逻辑：核内直方图融合 + 顺风车元数据 + 单次定长 AlltoAll (动态分配缓冲)
# =========================================================================
def v3_piggyback_dispatch(
    x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num,
    group_tp="", tp_world_size=0, tp_rank_id=0, shared_expert_num=1, shared_expert_rank_num=0,
    quant_mode=0, global_bs=0, comm_alg="", zero_expert_num=0, copy_expert_num=0,
    const_expert_num=0, x_active_mask=None, scales=None, expert_scales=None, elastic_info=None,
):
    bs, hidden_size = x.shape
    k = expert_ids.size(1)
    device = x.device
    total_send_tokens = bs * k
    chunk_rows = total_send_tokens + 1  # 末尾预留 1 行顺风车携带专家计数元数据

    # 1. 自定义 Ascend C Kernel 1：核内直方图 + 顺风车写入元数据行 + 多核双缓冲打包
    packed_send = torch.ops.cann_bench.moe_e2e_pack(x, expert_ids, ep_world_size, moe_expert_num, chunk_rows)

    # 2. 单次定长 AlltoAll 通信（V3 阶段每次动态分配接收缓冲区，对比 V4 静态持久化缓冲池）
    recv_buf = torch.empty((ep_world_size * chunk_rows, hidden_size), dtype=x.dtype, device=device)
    dist.all_to_all_single(recv_buf, packed_send)

    # 3. 自定义 Ascend C Kernel 2：读取顺风车计数 + 多核双缓冲解包写入最终专家主序
    max_capacity = max(ep_world_size * total_send_tokens, global_bs * k if global_bs > 0 else ep_world_size * total_send_tokens)
    expand_x, ep_recv_counts = torch.ops.cann_bench.moe_e2e_unpack(
        recv_buf, ep_world_size, moe_expert_num, chunk_rows, max_capacity
    )

    # V3 阶段尚未加入 V4 持久化辅助张量池，每次动态分配辅助输出
    dynamic_scales = torch.empty((max_capacity,), dtype=torch.float32, device=device)
    assist_info = torch.zeros((max_capacity * 3,), dtype=torch.int32, device=device)
    tp_recv_counts = torch.zeros((max(tp_world_size, 1),), dtype=torch.int32, device=device)
    expand_scales = torch.ones((max_capacity,), dtype=torch.float32, device=device)

    return expand_x, dynamic_scales, assist_info, ep_recv_counts, tp_recv_counts, expand_scales

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

        for _ in range(3):
            cand_out = golden.mc2_call_candidate(v3_piggyback_dispatch, ctx, inputs, attrs)
        torch_npu.npu.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(10):
            cand_out = golden.mc2_call_candidate(v3_piggyback_dispatch, ctx, inputs, attrs)
        torch_npu.npu.synchronize()
        dist.barrier()
        cand_us = (time.perf_counter() - t0) / 10.0 * 1e6

        valid_tokens = max(1, min(int(ref_out[3].sum().item()), cand_out[0].shape[0]))
        max_diff = float((cand_out[0][:valid_tokens].float() - ref_out[0][:valid_tokens].float()).abs().max().item())
        if rank == 0:
            results.append({
                "case_id": case["case_id"], "note": case["note"], "passed": max_diff < 1e-2,
                "max_diff": max_diff, "golden_us": round(ref_us, 2), "v3_us": round(cand_us, 2),
                "speedup": round(ref_us / max(cand_us, 1e-6), 3)
            })
    if rank == 0:
        out_q.put(results)
    dist.destroy_process_group()

def run_v3_2npu(alpha, port):
    with open("/mnt/workspace/my_topo_dispatch_exp/my_bench_root/level4/moe_distribute_dispatch_v2/cases.yaml") as f:
        cases = yaml.safe_load(f)["cases"]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, port, cases, alpha, q)) for r in range(2)]
    for p in procs: p.start()
    for p in procs: p.join()
    print(f"\n==================== V3 顺风车元数据协议双卡实测 (GRAPH_SKEW_ALPHA = {alpha}) ====================")
    print(f"{'Case':<6} {'Note':<26} {'Pass':<6} {'MaxDiff':<10} {'Golden(us)':<12} {'V3_Pigg(us)':<12} {'Speedup':<8}")
    for r in q.get():
        print(f"{r['case_id']:<6} {r['note']:<26} {str(r['passed']):<6} {r['max_diff']:<10.5f} {r['golden_us']:<12.2f} {r['v3_us']:<12.2f} {r['speedup']:<8.3f}x")

if __name__ == "__main__":
    run_v3_2npu(0.0, 29980)
    run_v3_2npu(1.5, 29990)

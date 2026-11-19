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
# 第一部分：单卡纯 Ascend C Kernel (moe_dispatch_pack) 极限吞吐与 HBM 利用率实测
# =========================================================================
def bench_single_card_kernel():
    torch_npu.npu.set_device(0)
    bs, k, hidden_size = 512, 8, 7168
    total_tokens = bs * k
    x = torch.randn(bs, hidden_size, dtype=torch.bfloat16, device="npu:0")
    dst_idx = torch.arange(total_tokens, dtype=torch.int32, device="npu:0").view(bs, k)

    # 5 次预热 + 30 次计时取平均
    for _ in range(5):
        _ = torch.ops.cann_bench.moe_dispatch_pack(x, dst_idx, total_tokens)
    torch_npu.npu.synchronize()

    iters = 30
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = torch.ops.cann_bench.moe_dispatch_pack(x, dst_idx, total_tokens)
    torch_npu.npu.synchronize()
    kernel_us = (time.perf_counter() - t0) / iters * 1e6

    # 计算搬运体积与带宽利用率
    one_way_bytes = total_tokens * hidden_size * 2          # 4096 * 7168 * 2B = 58.72 MB
    two_way_bytes = one_way_bytes * 2                       # 1读 + 1写 = 117.44 MB
    one_way_tbs = (one_way_bytes / (kernel_us * 1e-6)) / 1e12
    two_way_tbs = (two_way_bytes / (kernel_us * 1e-6)) / 1e12
    hbm_util = (two_way_tbs / 3.20) * 100.0                 # dav-2201 单 Die 理论峰值 HBM 带宽 3.20 TB/s

    print("\n================ 1. 单卡纯 Ascend C Kernel 极限吞吐实测 ================")
    print(f"测试负载: bs={bs}, k={k}, hidden={hidden_size} (bfloat16) | 单向有效数据量: {one_way_bytes / 1e6:.2f} MB")
    print(f"纯 Kernel 耗时: {kernel_us:.2f} us")
    print(f"单向有效吞吐:   {one_way_tbs:.2f} TB/s | 双向读写总带宽: {two_way_tbs:.2f} TB/s")
    print(f"HBM 带宽利用率: {hbm_util:.1f}% (基于单 Die 3.20 TB/s 理论峰值)")

# =========================================================================
# 第二部分：V2 阶段双卡完整调度 (双向自定义 Kernel + 双次通信) 实测
# =========================================================================
def v2_kernel_dispatch(
    x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num,
    group_tp="", tp_world_size=0, tp_rank_id=0, shared_expert_num=1, shared_expert_rank_num=0,
    quant_mode=0, global_bs=0, comm_alg="", zero_expert_num=0, copy_expert_num=0,
    const_expert_num=0, x_active_mask=None, scales=None, expert_scales=None, elastic_info=None,
):
    bs, hidden_size = x.shape
    k = expert_ids.size(1)
    device = x.device
    experts_per_rank = moe_expert_num // ep_world_size
    total_send_tokens = bs * k

    flat_experts = expert_ids.reshape(-1)
    send_expert_counts = torch.bincount(flat_experts, minlength=moe_expert_num).to(torch.int64)
    sorted_experts, perm = torch.sort(flat_experts, stable=True)
    expert_offsets = torch.cumsum(send_expert_counts, dim=0) - send_expert_counts
    local_rank_in_expert = torch.arange(total_send_tokens, device=device, dtype=torch.int64) - expert_offsets[sorted_experts]
    dst_slots_flat = torch.empty_like(flat_experts, dtype=torch.int32)
    dst_slots_flat[perm] = (expert_offsets[sorted_experts] + local_rank_in_expert).to(torch.int32)

    # 调用自定义 Ascend C 多核双缓冲 Scatter-Pack Kernel
    packed_send = torch.ops.cann_bench.moe_dispatch_pack(x, dst_slots_flat.view(bs, k), total_send_tokens)

    recv_expert_counts = torch.empty_like(send_expert_counts)
    dist.all_to_all_single(recv_expert_counts, send_expert_counts)

    recv_matrix = recv_expert_counts.view(ep_world_size, experts_per_rank)
    send_splits = send_expert_counts.view(ep_world_size, experts_per_rank).sum(dim=1).tolist()
    recv_splits = recv_matrix.sum(dim=1).tolist()
    total_recv_tokens = sum(recv_splits)

    max_capacity = max(total_recv_tokens, global_bs * k if global_bs > 0 else bs * ep_world_size * k)
    expand_x = torch.empty((max_capacity, hidden_size), dtype=x.dtype, device=device)

    if total_recv_tokens > 0 or total_send_tokens > 0:
        recv_buf = torch.empty((total_recv_tokens, hidden_size), dtype=x.dtype, device=device)
        dist.all_to_all_single(
            recv_buf, packed_send,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        block_ids = torch.arange(moe_expert_num, device=device, dtype=torch.int64)
        sort_key = (block_ids % experts_per_rank) * ep_world_size + (block_ids // experts_per_rank)
        token_keys = sort_key[torch.repeat_interleave(block_ids, recv_expert_counts)]
        _, gather_perm = torch.sort(token_keys, stable=True)

        # 调用自定义 Ascend C 多核双缓冲 Gather-Unpack Kernel
        torch.ops.cann_bench.moe_dispatch_gather(recv_buf, gather_perm.to(torch.int32), expand_x[:total_recv_tokens])

    dynamic_scales = torch.empty((max_capacity,), dtype=torch.float32, device=device)
    assist_info = torch.zeros((max_capacity * 3,), dtype=torch.int32, device=device)
    ep_recv_counts = recv_matrix.sum(dim=0).to(torch.int32)
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
            cand_out = golden.mc2_call_candidate(v2_kernel_dispatch, ctx, inputs, attrs)
        torch_npu.npu.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(10):
            cand_out = golden.mc2_call_candidate(v2_kernel_dispatch, ctx, inputs, attrs)
        torch_npu.npu.synchronize()
        dist.barrier()
        cand_us = (time.perf_counter() - t0) / 10.0 * 1e6

        valid_tokens = max(1, min(int(ref_out[3].sum().item()), cand_out[0].shape[0]))
        max_diff = float((cand_out[0][:valid_tokens].float() - ref_out[0][:valid_tokens].float()).abs().max().item())
        if rank == 0:
            results.append({
                "case_id": case["case_id"], "note": case["note"], "passed": max_diff < 1e-2,
                "max_diff": max_diff, "golden_us": round(ref_us, 2), "v2_us": round(cand_us, 2),
                "speedup": round(ref_us / max(cand_us, 1e-6), 3)
            })
    if rank == 0:
        out_q.put(results)
    dist.destroy_process_group()

def run_v2_2npu(alpha, port):
    with open("/mnt/workspace/my_topo_dispatch_exp/my_bench_root/level4/moe_distribute_dispatch_v2/cases.yaml") as f:
        cases = yaml.safe_load(f)["cases"]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, port, cases, alpha, q)) for r in range(2)]
    for p in procs: p.start()
    for p in procs: p.join()
    print(f"\n================ 2. V2 双缓冲 Kernel 双卡端到端实测 (GRAPH_SKEW_ALPHA = {alpha}) ================")
    print(f"{'Case':<6} {'Note':<26} {'Pass':<6} {'MaxDiff':<10} {'Golden(us)':<12} {'V2_Kern(us)':<12} {'Speedup':<8}")
    for r in q.get():
        print(f"{r['case_id']:<6} {r['note']:<26} {str(r['passed']):<6} {r['max_diff']:<10.5f} {r['golden_us']:<12.2f} {r['v2_us']:<12.2f} {r['speedup']:<8.3f}x")

if __name__ == "__main__":
    bench_single_card_kernel()
    run_v2_2npu(0.0, 29950)
    run_v2_2npu(1.5, 29960)

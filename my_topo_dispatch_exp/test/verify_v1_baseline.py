import os
import sys
import time
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

sys.path.insert(0, "/mnt/workspace/my_topo_dispatch_exp/my_bench_root/level4/moe_distribute_dispatch_v2")
import golden

def v1_pytorch_dispatch_baseline(
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

    # 1. 多步 PyTorch 碎片算子计算发送端重排索引
    flat_experts = expert_ids.reshape(-1)
    send_expert_counts = torch.bincount(flat_experts, minlength=moe_expert_num).to(torch.int64)
    _, perm = torch.sort(flat_experts, stable=True)
    token_indices = torch.div(perm, k, rounding_mode="floor")
    packed_send = x.index_select(0, token_indices)

    # 2. 第一次全网通信往返 (RTT 1)：交换各专家发送计数 + 强制 D2H .tolist() 同步
    recv_expert_counts = torch.empty_like(send_expert_counts)
    dist.all_to_all_single(recv_expert_counts, send_expert_counts)

    recv_matrix = recv_expert_counts.view(ep_world_size, experts_per_rank)
    send_splits = send_expert_counts.view(ep_world_size, experts_per_rank).sum(dim=1).tolist()
    recv_splits = recv_matrix.sum(dim=1).tolist()
    total_recv_tokens = sum(recv_splits)

    max_capacity = max(total_recv_tokens, global_bs * k if global_bs > 0 else bs * ep_world_size * k)
    expand_x = torch.empty((max_capacity, hidden_size), dtype=x.dtype, device=device)

    # 3. 第二次全网通信往返 (RTT 2)：变长 AlltoAllv + 接收端再次 sort/index_select 重排
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
        expand_x[:total_recv_tokens].copy_(recv_buf.index_select(0, gather_perm))

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
            cand_out = golden.mc2_call_candidate(v1_pytorch_dispatch_baseline, ctx, inputs, attrs)
        torch_npu.npu.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(10):
            cand_out = golden.mc2_call_candidate(v1_pytorch_dispatch_baseline, ctx, inputs, attrs)
        torch_npu.npu.synchronize()
        dist.barrier()
        cand_us = (time.perf_counter() - t0) / 10.0 * 1e6

        valid_tokens = max(1, min(int(ref_out[3].sum().item()), cand_out[0].shape[0]))
        max_diff = float((cand_out[0][:valid_tokens].float() - ref_out[0][:valid_tokens].float()).abs().max().item())
        if rank == 0:
            results.append({
                "case_id": case["case_id"], "note": case["note"], "passed": max_diff < 1e-2,
                "max_diff": max_diff, "golden_us": round(ref_us, 2), "v1_us": round(cand_us, 2),
                "speedup": round(ref_us / max(cand_us, 1e-6), 3)
            })
    if rank == 0:
        out_q.put(results)
    dist.destroy_process_group()

def run_bench(alpha, port):
    with open("/mnt/workspace/my_topo_dispatch_exp/my_bench_root/level4/moe_distribute_dispatch_v2/cases.yaml") as f:
        cases = yaml.safe_load(f)["cases"]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, port, cases, alpha, q)) for r in range(2)]
    for p in procs: p.start()
    for p in procs: p.join()
    print(f"\n==================== V1 初代基线双卡实测 (GRAPH_SKEW_ALPHA = {alpha}) ====================")
    print(f"{'Case':<6} {'Note':<26} {'Pass':<6} {'MaxDiff':<10} {'Golden(us)':<12} {'V1_Base(us)':<12} {'Speedup':<8}")
    for r in q.get():
        print(f"{r['case_id']:<6} {r['note']:<26} {str(r['passed']):<6} {r['max_diff']:<10.5f} {r['golden_us']:<12.2f} {r['v1_us']:<12.2f} {r['speedup']:<8.3f}x")

if __name__ == "__main__":
    run_bench(0.0, 29930)
    run_bench(1.5, 29940)

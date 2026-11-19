import os, sys, time, yaml, torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

sys.path.insert(0, "/mnt/workspace/my_topo_dispatch_exp/my_bench_root/level4/moe_distribute_dispatch_v2")
import golden
import cann_bench

_BG_RECV_POOL = {}
_BG_AUX_POOL = {}

def beat_golden_candidate(
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
    stride_rows = ((bs * min(k, experts_per_rank) + 1 + 15) // 16) * 16

    packed_send, _ = torch.ops.cann_bench.moe_bg_pack(x, expert_ids, ep_world_size, moe_expert_num, stride_rows)
    recv_key = (device.index, packed_send.shape, packed_send.dtype)
    recv_buf = _BG_RECV_POOL.get(recv_key)
    if recv_buf is None:
        recv_buf = torch.empty_like(packed_send)
        _BG_RECV_POOL[recv_key] = recv_buf
    dist.all_to_all_single(recv_buf, packed_send)

    max_capacity = max(ep_world_size * total_send_tokens, global_bs * k if global_bs > 0 else ep_world_size * total_send_tokens)
    expand_x, ep_recv_counts = torch.ops.cann_bench.moe_bg_unpack(recv_buf, ep_world_size, moe_expert_num, stride_rows, max_capacity)

    aux_key = (device.index, max_capacity, max(tp_world_size, 1))
    aux = _BG_AUX_POOL.get(aux_key)
    if aux is None:
        aux = (
            torch.empty((max_capacity,), dtype=torch.float32, device=device),
            torch.empty((max_capacity * 3,), dtype=torch.int32, device=device),
            torch.zeros((max(tp_world_size, 1),), dtype=torch.int32, device=device),
            torch.ones((max_capacity,), dtype=torch.float32, device=device),
        )
        _BG_AUX_POOL[aux_key] = aux
    return expand_x, aux, aux, ep_recv_counts, aux, aux

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
            cand_out = golden.mc2_call_candidate(beat_golden_candidate, ctx, inputs, attrs)
        torch_npu.npu.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(10):
            cand_out = golden.mc2_call_candidate(beat_golden_candidate, ctx, inputs, attrs)
        torch_npu.npu.synchronize()
        dist.barrier()
        cand_us = (time.perf_counter() - t0) / 10.0 * 1e6

        valid_tokens = max(1, min(int(ref_out[3].sum().item()), cand_out[0].shape[0]))
        max_diff = float((cand_out[0][:valid_tokens].float() - ref_out[0][:valid_tokens].float()).abs().max().item())
        if rank == 0:
            results.append({"case_id": case["case_id"], "note": case["note"], "passed": max_diff < 1e-2, "max_diff": max_diff, "golden_us": round(ref_us, 2), "cand_us": round(cand_us, 2), "speedup": round(ref_us / max(cand_us, 1e-6), 3)})
    if rank == 0:
        out_q.put(results)
    dist.destroy_process_group()

def run_bench(alpha):
    with open("/mnt/workspace/my_topo_dispatch_exp/my_bench_root/level4/moe_distribute_dispatch_v2/cases.yaml") as f:
        cases = yaml.safe_load(f)["cases"]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, 29800 + int(alpha * 10), cases, alpha, q)) for r in range(2)]
    for p in procs: p.start()
    for p in procs: p.join()
    print(f"\n==================== BEAT-GOLDEN 2-NPU Benchmark (GRAPH_SKEW_ALPHA = {alpha}) ====================")
    print(f"{'Case':<6} {'Note':<26} {'Pass':<6} {'MaxDiff':<10} {'Golden(us)':<12} {'BG_Opt(us)':<12} {'Speedup':<8}")
    for r in q.get():
        print(f"{r['case_id']:<6} {r['note']:<26} {str(r['passed']):<6} {r['max_diff']:<10.5f} {r['golden_us']:<12.2f} {r['cand_us']:<12.2f} {r['speedup']:<8.3f}x")

if __name__ == "__main__":
    run_bench(0.0)
    run_bench(1.5)

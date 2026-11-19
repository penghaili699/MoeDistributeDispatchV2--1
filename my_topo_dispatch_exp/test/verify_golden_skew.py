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

    res = {}
    for case in cases:
        payload = {"input_shapes": case["input_shape"], "dtypes": case["dtype"], "value_ranges": case["value_range"], "attrs": case["attrs"]}
        inputs = golden.mc2_make_rank_inputs(ctx, payload)
        attrs = case["attrs"]

        # 3 次预热 + 15 次实测取平均
        for _ in range(3):
            _ = golden.mc2_distributed_golden(ctx, inputs, attrs)
        torch_npu.npu.synchronize()
        dist.barrier()

        t0 = time.perf_counter()
        for _ in range(15):
            _ = golden.mc2_distributed_golden(ctx, inputs, attrs)
        torch_npu.npu.synchronize()
        dist.barrier()
        avg_us = (time.perf_counter() - t0) / 15.0 * 1e6

        if rank == 0:
            res[case["case_id"]] = (case["note"], round(avg_us, 2))
    if rank == 0:
        out_q.put(res)
    dist.destroy_process_group()

def measure_alpha(alpha, port):
    with open("/mnt/workspace/my_topo_dispatch_exp/my_bench_root/level4/moe_distribute_dispatch_v2/cases.yaml") as f:
        cases = yaml.safe_load(f)["cases"]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, port, cases, alpha, q)) for r in range(2)]
    for p in procs: p.start()
    for p in procs: p.join()
    return q.get()

if __name__ == "__main__":
    print("正在双卡实测官方算子 (npu_moe_distribute_dispatch_v2) 在不同负载倾斜度下的延迟...")
    r_uniform = measure_alpha(0.0, 29910)
    r_skewed = measure_alpha(1.5, 29920)

    print("\n================ 官方算子负载倾斜退化对比表 ================")
    print(f"{'Case':<6} {'测试场景':<26} {'均匀 Alpha=0.0 (us)':<22} {'倾斜 Alpha=1.5 (us)':<22} {'延迟退化幅度':<12}")
    for cid in sorted(r_uniform.keys()):
        note, u_us = r_uniform[cid]
        _, s_us = r_skewed[cid]
        deg = (s_us - u_us) / u_us * 100.0
        print(f"{cid:<6} {note:<26} {u_us:<22.2f} {s_us:<22.2f} {deg:+.1f}%")

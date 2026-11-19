import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
import cann_bench

def _profile_worker(rank, world_size, port, out_q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    torch_npu.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

    bs, k, hidden_size, moe_expert_num = 512, 8, 7168, 64
    experts_per_rank = moe_expert_num // world_size
    total_send_tokens = bs * k
    x = torch.randn(bs, hidden_size, dtype=torch.bfloat16, device=device)
    expert_ids = torch.randint(0, moe_expert_num, (bs, k), dtype=torch.int32, device=device)

    # 预热 3 次
    for _ in range(3):
        flat_exp = expert_ids.reshape(-1)
        cnt = torch.bincount(flat_exp, minlength=moe_expert_num).to(torch.int64)
        s_exp, perm = torch.sort(flat_exp, stable=True)
        off = torch.cumsum(cnt, dim=0) - cnt
        slots = torch.empty_like(flat_exp, dtype=torch.int32)
        slots[perm] = (off[s_exp] + (torch.arange(total_send_tokens, device=device) - off[s_exp])).to(torch.int32)
        packed = torch.ops.cann_bench.moe_dispatch_pack(x, slots.view(bs, k), total_send_tokens)
        rcnt = torch.empty_like(cnt)
        dist.all_to_all_single(rcnt, cnt)
        rmat = rcnt.view(world_size, experts_per_rank)
        ss = cnt.view(world_size, experts_per_rank).sum(dim=1).tolist()
        rs = rmat.sum(dim=1).tolist()
        rbuf = torch.empty((sum(rs), hidden_size), dtype=x.dtype, device=device)
        dist.all_to_all_single(rbuf, packed, output_split_sizes=rs, input_split_sizes=ss)
        bids = torch.arange(moe_expert_num, device=device, dtype=torch.int64)
        skey = (bids % experts_per_rank) * world_size + (bids // experts_per_rank)
        _, gperm = torch.sort(skey[torch.repeat_interleave(bids, rcnt)], stable=True)
        out = torch.empty_like(rbuf)
        torch.ops.cann_bench.moe_dispatch_gather(rbuf, gperm.to(torch.int32), out)
    torch_npu.npu.synchronize()
    dist.barrier()

    # 正式分阶段计时 (15 次迭代取均值)
    iters = 15
    t_send_idx, t_pack_kern, t_meta_rtt, t_data_comm, t_recv_idx, t_gather_kern = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for _ in range(iters):
        # 阶段 1: 发送端 PyTorch 索引计算 (bincount + sort + cumsum)
        torch_npu.npu.synchronize()
        t0 = time.perf_counter()
        flat_exp = expert_ids.reshape(-1)
        cnt = torch.bincount(flat_exp, minlength=moe_expert_num).to(torch.int64)
        s_exp, perm = torch.sort(flat_exp, stable=True)
        off = torch.cumsum(cnt, dim=0) - cnt
        slots = torch.empty_like(flat_exp, dtype=torch.int32)
        slots[perm] = (off[s_exp] + (torch.arange(total_send_tokens, device=device) - off[s_exp])).to(torch.int32)
        torch_npu.npu.synchronize()
        t_send_idx += (time.perf_counter() - t0) * 1e6

        # 阶段 2: 自定义 Ascend C Pack Kernel
        t0 = time.perf_counter()
        packed = torch.ops.cann_bench.moe_dispatch_pack(x, slots.view(bs, k), total_send_tokens)
        torch_npu.npu.synchronize()
        t_pack_kern += (time.perf_counter() - t0) * 1e6

        # 阶段 3: 第一次元数据通信 RTT + D2H .tolist() 强制同步
        t0 = time.perf_counter()
        rcnt = torch.empty_like(cnt)
        dist.all_to_all_single(rcnt, cnt)
        rmat = rcnt.view(world_size, experts_per_rank)
        ss = cnt.view(world_size, experts_per_rank).sum(dim=1).tolist()
        rs = rmat.sum(dim=1).tolist()
        tot_r = sum(rs)
        torch_npu.npu.synchronize()
        t_meta_rtt += (time.perf_counter() - t0) * 1e6

        # 阶段 4: 第二次主数据变长通信
        rbuf = torch.empty((tot_r, hidden_size), dtype=x.dtype, device=device)
        t0 = time.perf_counter()
        dist.all_to_all_single(rbuf, packed, output_split_sizes=rs, input_split_sizes=ss)
        torch_npu.npu.synchronize()
        t_data_comm += (time.perf_counter() - t0) * 1e6

        # 阶段 5: 接收端 PyTorch 索引计算 (repeat_interleave + sort)
        t0 = time.perf_counter()
        bids = torch.arange(moe_expert_num, device=device, dtype=torch.int64)
        skey = (bids % experts_per_rank) * world_size + (bids // experts_per_rank)
        _, gperm = torch.sort(skey[torch.repeat_interleave(bids, rcnt)], stable=True)
        torch_npu.npu.synchronize()
        t_recv_idx += (time.perf_counter() - t0) * 1e6

        # 阶段 6: 自定义 Ascend C Gather Kernel
        out = torch.empty_like(rbuf)
        t0 = time.perf_counter()
        torch.ops.cann_bench.moe_dispatch_gather(rbuf, gperm.to(torch.int32), out)
        torch_npu.npu.synchronize()
        t_gather_kern += (time.perf_counter() - t0) * 1e6

    if rank == 0:
        out_q.put({
            "send_idx": t_send_idx / iters,
            "pack_kern": t_pack_kern / iters,
            "meta_rtt": t_meta_rtt / iters,
            "data_comm": t_data_comm / iters,
            "recv_idx": t_recv_idx / iters,
            "gather_kern": t_gather_kern / iters,
        })
    dist.destroy_process_group()

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_profile_worker, args=(r, 2, 29970, q)) for r in range(2)]
    for p in procs: p.start()
    for p in procs: p.join()
    d = q.get()

    total = sum(d.values())
    overhead = d["send_idx"] + d["meta_rtt"] + d["recv_idx"]
    core_work = d["pack_kern"] + d["data_comm"] + d["gather_kern"]

    print("\n================ V2 阶段耗时剖面分解 (bs=512, k=8, H=7168) ================")
    print(f"1. [控制开销] 发送端 PyTorch 排序索引 (sort/cumsum):   {d['send_idx']:8.2f} us ({d['send_idx']/total*100:5.1f}%)")
    print(f"2. [核心搬运] 自定义 Ascend C Pack Kernel:           {d['pack_kern']:8.2f} us ({d['pack_kern']/total*100:5.1f}%)")
    print(f"3. [同步瓶颈] 首次元数据通信 RTT + D2H .tolist():    {d['meta_rtt']:8.2f} us ({d['meta_rtt']/total*100:5.1f}%)")
    print(f"4. [核心通信] 主数据变长 AlltoAllv 传输:             {d['data_comm']:8.2f} us ({d['data_comm']/total*100:5.1f}%)")
    print(f"5. [控制开销] 接收端 PyTorch 排序索引 (repeat/sort): {d['recv_idx']:8.2f} us ({d['recv_idx']/total*100:5.1f}%)")
    print(f"6. [核心搬运] 自定义 Ascend C Gather Kernel:         {d['gather_kern']:8.2f} us ({d['gather_kern']/total*100:5.1f}%)")
    print("---------------------------------------------------------------------------")
    print(f"总耗时 (分阶段同步测量): {total:.2f} us")
    print(f"==> 控制面与二次通信 RTT 开销 (1+3+5): {overhead:.2f} us (占比 {overhead/total*100:.1f}%)")
    print(f"==> 有效数据搬运与主通信耗时  (2+4+6): {core_work:.2f} us (占比 {core_work/total*100:.1f}%)")

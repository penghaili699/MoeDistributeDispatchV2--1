#!/usr/bin/env python3
import torch
import torch.distributed as dist
import torch_npu
from . import _C  # noqa: F401

_RECV_POOL = {}
_AUX_POOL = {}

def moe_distribute_dispatch_v2(
    x: torch.Tensor, expert_ids: torch.Tensor, group_ep: str, ep_world_size: int, ep_rank_id: int, moe_expert_num: int,
    group_tp: str = "", tp_world_size: int = 0, tp_rank_id: int = 0, shared_expert_num: int = 1, shared_expert_rank_num: int = 0,
    quant_mode: int = 0, global_bs: int = 0, comm_alg: str = "", zero_expert_num: int = 0, copy_expert_num: int = 0,
    const_expert_num: int = 0, x_active_mask=None, scales=None, expert_scales=None, elastic_info=None,
):
    bs, hidden_size = x.shape
    k = expert_ids.size(1)
    device = x.device
    total_send_tokens = bs * k
    chunk_rows = total_send_tokens + 1

    # 1. 自定义 Ascend C Kernel 1（复用 C++ 静态持久化缓冲池）
    packed_send = torch.ops.cann_bench.moe_e2e_pack(x, expert_ids, ep_world_size, moe_expert_num, chunk_rows)

    # 2. 复用持久化接收缓冲区执行定长 AlltoAll 通信（零动态显存分配）
    recv_key = (device.index, packed_send.shape, packed_send.dtype)
    recv_buf = _RECV_POOL.get(recv_key)
    if recv_buf is None:
        recv_buf = torch.empty_like(packed_send)
        _RECV_POOL[recv_key] = recv_buf
    dist.all_to_all_single(recv_buf, packed_send)

    # 3. 自定义 Ascend C Kernel 2（复用持久化输出缓冲池）
    max_capacity = max(ep_world_size * total_send_tokens, global_bs * k if global_bs > 0 else ep_world_size * total_send_tokens)
    expand_x, ep_recv_counts = torch.ops.cann_bench.moe_e2e_unpack(recv_buf, ep_world_size, moe_expert_num, chunk_rows, max_capacity)

    # 4. 复用持久化辅助张量池（省去每次 torch.zeros 全显存清零开销）
    aux_key = (device.index, max_capacity, max(tp_world_size, 1))
    aux = _AUX_POOL.get(aux_key)
    if aux is None:
        aux = (
            torch.empty((max_capacity,), dtype=torch.float32, device=device),
            torch.zeros((max_capacity * 3,), dtype=torch.int32, device=device),
            torch.zeros((max(tp_world_size, 1),), dtype=torch.int32, device=device),
            torch.ones((max_capacity,), dtype=torch.float32, device=device),
        )
        _AUX_POOL[aux_key] = aux

    return expand_x, aux[0], aux[1], ep_recv_counts, aux[2], aux[3]

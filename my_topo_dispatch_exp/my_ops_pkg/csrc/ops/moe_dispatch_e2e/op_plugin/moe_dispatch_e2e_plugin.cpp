#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <algorithm>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "../op_kernel/moe_dispatch_e2e_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("moe_e2e_pack(Tensor x, Tensor expert_ids, int ep_world_size, int moe_expert_num, int chunk_rows) -> Tensor");
    m.def("moe_e2e_unpack(Tensor recv_buf, int ep_world_size, int moe_expert_num, int chunk_rows, int max_capacity) -> (Tensor, Tensor)");
}

torch::Tensor moe_e2e_pack_meta(const torch::Tensor &x, const torch::Tensor &expert_ids, int64_t ep_world_size, int64_t moe_expert_num, int64_t chunk_rows)
{
    return torch::empty({ep_world_size * chunk_rows, x.size(1)}, x.options());
}

std::tuple<torch::Tensor, torch::Tensor> moe_e2e_unpack_meta(const torch::Tensor &recv_buf, int64_t ep_world_size, int64_t moe_expert_num, int64_t chunk_rows, int64_t max_capacity)
{
    return std::make_tuple(torch::empty({max_capacity, recv_buf.size(1)}, recv_buf.options()), torch::empty({moe_expert_num / ep_world_size}, recv_buf.options().dtype(torch::kInt32)));
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("moe_e2e_pack", moe_e2e_pack_meta);
    m.impl("moe_e2e_unpack", moe_e2e_unpack_meta);
}

// 静态持久化缓冲池（消除每次调用的大显存分配与清零延迟）
static torch::Tensor g_pack_pool;
static torch::Tensor g_expand_pool;
static torch::Tensor g_cnt_pool;

torch::Tensor moe_e2e_pack_npu(const torch::Tensor &x, const torch::Tensor &expert_ids, int64_t ep_world_size, int64_t moe_expert_num, int64_t chunk_rows)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto x_c = x.contiguous(); auto idx_c = expert_ids.contiguous();
    int64_t req_rows = ep_world_size * chunk_rows, h = x_c.size(1);
    if (!g_pack_pool.defined() || g_pack_pool.device() != x.device() || g_pack_pool.size(0) < req_rows || g_pack_pool.size(1) != h || g_pack_pool.scalar_type() != x.scalar_type()) {
        g_pack_pool = torch::empty({req_rows, h}, x.options());
    }
    auto out = g_pack_pool.narrow(0, 0, req_rows);
    int64_t bs = idx_c.size(0), k = idx_c.size(1);
    int64_t numBlocks = std::min(get_e2e_aiv_num(), bs * k);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xp = (GM_ADDR)x_c.data_ptr(), ip = (GM_ADDR)idx_c.data_ptr(), op = (GM_ADDR)out.data_ptr();
    at_npu::native::OpCommand::RunOpApi("MoeE2ePack", [=]() -> int { launch_moe_e2e_pack_bf16(xp, ip, op, bs, k, h, ep_world_size, moe_expert_num, chunk_rows, numBlocks, stream); return 0; });
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> moe_e2e_unpack_npu(const torch::Tensor &recv_buf, int64_t ep_world_size, int64_t moe_expert_num, int64_t chunk_rows, int64_t max_capacity)
{
    const c10::OptionalDeviceGuard guard(recv_buf.device());
    auto r_c = recv_buf.contiguous();
    int64_t h = r_c.size(1), local_exp = moe_expert_num / ep_world_size;
    if (!g_expand_pool.defined() || g_expand_pool.device() != recv_buf.device() || g_expand_pool.size(0) < max_capacity || g_expand_pool.size(1) != h || g_expand_pool.scalar_type() != recv_buf.scalar_type()) {
        g_expand_pool = torch::empty({max_capacity, h}, recv_buf.options());
    }
    if (!g_cnt_pool.defined() || g_cnt_pool.device() != recv_buf.device() || g_cnt_pool.size(0) < local_exp) {
        g_cnt_pool = torch::empty({local_exp}, recv_buf.options().dtype(torch::kInt32));
    }
    auto expand_x = g_expand_pool.narrow(0, 0, max_capacity);
    auto ep_cnt = g_cnt_pool.narrow(0, 0, local_exp);
    int64_t numBlocks = get_e2e_aiv_num();
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto ip = (GM_ADDR)r_c.data_ptr(), op = (GM_ADDR)expand_x.data_ptr(), cp = (GM_ADDR)ep_cnt.data_ptr();
    at_npu::native::OpCommand::RunOpApi("MoeE2eUnpack", [=]() -> int { launch_moe_e2e_unpack_bf16(ip, op, cp, h, ep_world_size, moe_expert_num, chunk_rows, numBlocks, stream); return 0; });
    return std::make_tuple(expand_x, ep_cnt);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("moe_e2e_pack", moe_e2e_pack_npu);
    m.impl("moe_e2e_unpack", moe_e2e_unpack_npu);
}
}

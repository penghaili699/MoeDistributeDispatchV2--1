#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <algorithm>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "../op_kernel/moe_hw_aligned_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("moe_hw_pack(Tensor x, Tensor expert_ids, int ep_world_size, int moe_expert_num, int chunk_rows) -> Tensor");
    m.def("moe_hw_unpack(Tensor recv_buf, int ep_world_size, int moe_expert_num, int chunk_rows, int max_capacity) -> (Tensor, Tensor)");
}

torch::Tensor moe_hw_pack_meta(const torch::Tensor &x, const torch::Tensor &expert_ids, int64_t ep_world_size, int64_t moe_expert_num, int64_t chunk_rows)
{
    return torch::empty({ep_world_size * chunk_rows, x.size(1)}, x.options());
}

std::tuple<torch::Tensor, torch::Tensor> moe_hw_unpack_meta(const torch::Tensor &recv_buf, int64_t ep_world_size, int64_t moe_expert_num, int64_t chunk_rows, int64_t max_capacity)
{
    return std::make_tuple(torch::empty({max_capacity, recv_buf.size(1)}, recv_buf.options()), torch::empty({moe_expert_num / ep_world_size}, recv_buf.options().dtype(torch::kInt32)));
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("moe_hw_pack", moe_hw_pack_meta);
    m.impl("moe_hw_unpack", moe_hw_unpack_meta);
}

static torch::Tensor hw_pack_pool;
static torch::Tensor hw_expand_pool;
static torch::Tensor hw_cnt_pool;

torch::Tensor moe_hw_pack_npu(const torch::Tensor &x, const torch::Tensor &expert_ids, int64_t ep_world_size, int64_t moe_expert_num, int64_t chunk_rows)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto x_c = x.contiguous(); auto idx_c = expert_ids.contiguous();
    int64_t req_rows = ep_world_size * chunk_rows, h = x_c.size(1);
    if (!hw_pack_pool.defined() || hw_pack_pool.device() != x.device() || hw_pack_pool.size(0) < req_rows || hw_pack_pool.size(1) != h) {
        hw_pack_pool = torch::empty({req_rows, h}, x.options());
    }
    auto out = hw_pack_pool.narrow(0, 0, req_rows);
    int64_t bs = idx_c.size(0), k = idx_c.size(1);
    int64_t numBlocks = std::min(get_hw_aiv_num(), bs * k);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xp = (GM_ADDR)x_c.data_ptr(), ip = (GM_ADDR)idx_c.data_ptr(), op = (GM_ADDR)out.data_ptr();
    at_npu::native::OpCommand::RunOpApi("MoeHwPack", [=]() -> int { launch_moe_hw_pack_bf16(xp, ip, op, bs, k, h, ep_world_size, moe_expert_num, chunk_rows, numBlocks, stream); return 0; });
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> moe_hw_unpack_npu(const torch::Tensor &recv_buf, int64_t ep_world_size, int64_t moe_expert_num, int64_t chunk_rows, int64_t max_capacity)
{
    const c10::OptionalDeviceGuard guard(recv_buf.device());
    auto r_c = recv_buf.contiguous();
    int64_t h = r_c.size(1), local_exp = moe_expert_num / ep_world_size;
    if (!hw_expand_pool.defined() || hw_expand_pool.device() != recv_buf.device() || hw_expand_pool.size(0) < max_capacity || hw_expand_pool.size(1) != h) {
        hw_expand_pool = torch::empty({max_capacity, h}, recv_buf.options());
    }
    if (!hw_cnt_pool.defined() || hw_cnt_pool.device() != recv_buf.device() || hw_cnt_pool.size(0) < local_exp) {
        hw_cnt_pool = torch::empty({local_exp}, recv_buf.options().dtype(torch::kInt32));
    }
    auto expand_x = hw_expand_pool.narrow(0, 0, max_capacity);
    auto ep_cnt = hw_cnt_pool.narrow(0, 0, local_exp);
    int64_t numBlocks = get_hw_aiv_num();
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto ip = (GM_ADDR)r_c.data_ptr(), op = (GM_ADDR)expand_x.data_ptr(), cp = (GM_ADDR)ep_cnt.data_ptr();
    at_npu::native::OpCommand::RunOpApi("MoeHwUnpack", [=]() -> int { launch_moe_hw_unpack_bf16(ip, op, cp, h, ep_world_size, moe_expert_num, chunk_rows, numBlocks, stream); return 0; });
    return std::make_tuple(expand_x, ep_cnt);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("moe_hw_pack", moe_hw_pack_npu);
    m.impl("moe_hw_unpack", moe_hw_unpack_npu);
}
}

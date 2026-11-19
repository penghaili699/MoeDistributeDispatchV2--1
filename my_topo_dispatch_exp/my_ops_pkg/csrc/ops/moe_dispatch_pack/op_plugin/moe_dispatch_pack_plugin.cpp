#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <algorithm>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "../op_kernel/moe_dispatch_pack_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("moe_dispatch_pack(Tensor x, Tensor dst_indices, int total_slots) -> Tensor");
    m.def("moe_dispatch_gather(Tensor recv_buf, Tensor src_indices, Tensor expand_out) -> Tensor");
}

torch::Tensor moe_dispatch_pack_meta(const torch::Tensor &x, const torch::Tensor &dst_indices, int64_t total_slots)
{
    return torch::empty({total_slots, x.size(1)}, x.options());
}

torch::Tensor moe_dispatch_gather_meta(const torch::Tensor &recv_buf, const torch::Tensor &src_indices, const torch::Tensor &expand_out)
{
    return expand_out;
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("moe_dispatch_pack", moe_dispatch_pack_meta);
    m.impl("moe_dispatch_gather", moe_dispatch_gather_meta);
}

torch::Tensor moe_dispatch_pack_npu(const torch::Tensor &x, const torch::Tensor &dst_indices, int64_t total_slots)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto x_contig = x.contiguous();
    auto idx_contig = dst_indices.contiguous();
    auto out = moe_dispatch_pack_meta(x_contig, idx_contig, total_slots);

    int64_t bs = idx_contig.size(0);
    int64_t k = idx_contig.size(1);
    int64_t hidden_size = x_contig.size(1);
    int64_t total_pairs = bs * k;
    if (total_pairs == 0 || total_slots == 0) return out;

    int64_t numBlocks = std::min(get_moe_pack_aiv_num(), total_pairs);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto x_ptr = (GM_ADDR)x_contig.data_ptr();
    auto idx_ptr = (GM_ADDR)idx_contig.data_ptr();
    auto out_ptr = (GM_ADDR)out.data_ptr();

    at_npu::native::OpCommand::RunOpApi("MoeDispatchPack", [=]() -> int {
        launch_moe_dispatch_pack_bf16(x_ptr, idx_ptr, out_ptr, bs, k, hidden_size, numBlocks, stream);
        return 0;
    });
    return out;
}

torch::Tensor moe_dispatch_gather_npu(const torch::Tensor &recv_buf, const torch::Tensor &src_indices, const torch::Tensor &expand_out)
{
    const c10::OptionalDeviceGuard guard(recv_buf.device());
    auto recv_contig = recv_buf.contiguous();
    auto idx_contig = src_indices.contiguous();
    int64_t total_tokens = idx_contig.numel();
    int64_t hidden_size = recv_contig.size(1);
    if (total_tokens == 0) return expand_out;

    int64_t numBlocks = std::min(get_moe_pack_aiv_num(), total_tokens);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto in_ptr = (GM_ADDR)recv_contig.data_ptr();
    auto idx_ptr = (GM_ADDR)idx_contig.data_ptr();
    auto out_ptr = (GM_ADDR)expand_out.data_ptr();

    at_npu::native::OpCommand::RunOpApi("MoeDispatchGather", [=]() -> int {
        launch_moe_dispatch_gather_bf16(in_ptr, idx_ptr, out_ptr, total_tokens, hidden_size, numBlocks, stream);
        return 0;
    });
    return expand_out;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("moe_dispatch_pack", moe_dispatch_pack_npu);
    m.impl("moe_dispatch_gather", moe_dispatch_gather_npu);
}

} // namespace cann_bench

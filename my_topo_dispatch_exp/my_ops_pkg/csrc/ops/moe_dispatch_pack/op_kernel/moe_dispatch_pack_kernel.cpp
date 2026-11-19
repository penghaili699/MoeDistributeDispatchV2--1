#include <algorithm>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "moe_dispatch_pack_launch.h"

constexpr static int32_t BUFFER_NUM = 2;
constexpr static int64_t MAX_TILE_BYTES = 32768;

__global__ __aicore__ void moe_dispatch_pack_bf16_kernel(
    GM_ADDR x, GM_ADDR dst_indices, GM_ADDR packed_out,
    int64_t bs, int64_t k, int64_t hidden_size)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;

    int64_t total_pairs = bs * k;
    int64_t block_idx = AscendC::GetBlockIdx();
    int64_t block_num = AscendC::GetBlockNum();
    int64_t pairs_per_core = (total_pairs + block_num - 1) / block_num;
    int64_t start_pair = block_idx * pairs_per_core;
    int64_t end_pair = ((start_pair + pairs_per_core) < total_pairs ? (start_pair + pairs_per_core) : total_pairs);
    if (start_pair >= end_pair) return;

    int64_t tile_elems = (hidden_size < (MAX_TILE_BYTES / (int64_t)sizeof(bfloat16_t)) ? hidden_size : (MAX_TILE_BYTES / (int64_t)sizeof(bfloat16_t)));
    uint32_t alloc_bytes = static_cast<uint32_t>(tile_elems * sizeof(bfloat16_t));
    pipe.InitBuffer(inQueue, BUFFER_NUM, alloc_bytes);
    pipe.InitBuffer(outQueue, BUFFER_NUM, alloc_bytes);

    AscendC::GlobalTensor<bfloat16_t> xGm, outGm;
    AscendC::GlobalTensor<int32_t> dstIdxGm;
    xGm.SetGlobalBuffer((__gm__ bfloat16_t*)x);
    dstIdxGm.SetGlobalBuffer((__gm__ int32_t*)dst_indices);
    outGm.SetGlobalBuffer((__gm__ bfloat16_t*)packed_out);
    AscendC::DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};

    for (int64_t p = start_pair; p < end_pair; ++p) {
        int32_t dst_slot = dstIdxGm.GetValue(p);
        if (dst_slot < 0) continue;
        int64_t src_base = (p / k) * hidden_size;
        int64_t dst_base = static_cast<int64_t>(dst_slot) * hidden_size;

        for (int64_t offset = 0; offset < hidden_size; offset += tile_elems) {
            int64_t cur_elems = ((hidden_size - offset) < tile_elems ? (hidden_size - offset) : tile_elems);
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(cur_elems * sizeof(bfloat16_t)), 0, 0, 0};

            auto localIn = inQueue.AllocTensor<bfloat16_t>();
            AscendC::DataCopyPad(localIn, xGm[src_base + offset], copyParams, padParams);
            inQueue.EnQue(localIn);

            localIn = inQueue.DeQue<bfloat16_t>();
            auto localOut = outQueue.AllocTensor<bfloat16_t>();
            AscendC::DataCopy(localOut, localIn, cur_elems);
            outQueue.EnQue(localOut);
            inQueue.FreeTensor(localIn);

            localOut = outQueue.DeQue<bfloat16_t>();
            AscendC::DataCopyPad(outGm[dst_base + offset], localOut, copyParams);
            outQueue.FreeTensor(localOut);
        }
    }
}

__global__ __aicore__ void moe_dispatch_gather_bf16_kernel(
    GM_ADDR recv_buf, GM_ADDR src_indices, GM_ADDR expand_out,
    int64_t total_tokens, int64_t hidden_size)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;

    int64_t block_idx = AscendC::GetBlockIdx();
    int64_t block_num = AscendC::GetBlockNum();
    int64_t tokens_per_core = (total_tokens + block_num - 1) / block_num;
    int64_t start_tok = block_idx * tokens_per_core;
    int64_t end_tok = ((start_tok + tokens_per_core) < total_tokens ? (start_tok + tokens_per_core) : total_tokens);
    if (start_tok >= end_tok) return;

    int64_t tile_elems = (hidden_size < (MAX_TILE_BYTES / (int64_t)sizeof(bfloat16_t)) ? hidden_size : (MAX_TILE_BYTES / (int64_t)sizeof(bfloat16_t)));
    uint32_t alloc_bytes = static_cast<uint32_t>(tile_elems * sizeof(bfloat16_t));
    pipe.InitBuffer(inQueue, BUFFER_NUM, alloc_bytes);
    pipe.InitBuffer(outQueue, BUFFER_NUM, alloc_bytes);

    AscendC::GlobalTensor<bfloat16_t> inGm, outGm;
    AscendC::GlobalTensor<int32_t> srcIdxGm;
    inGm.SetGlobalBuffer((__gm__ bfloat16_t*)recv_buf);
    srcIdxGm.SetGlobalBuffer((__gm__ int32_t*)src_indices);
    outGm.SetGlobalBuffer((__gm__ bfloat16_t*)expand_out);
    AscendC::DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};

    for (int64_t t = start_tok; t < end_tok; ++t) {
        int32_t src_slot = srcIdxGm.GetValue(t);
        if (src_slot < 0) continue;
        int64_t src_base = static_cast<int64_t>(src_slot) * hidden_size;
        int64_t dst_base = t * hidden_size;

        for (int64_t offset = 0; offset < hidden_size; offset += tile_elems) {
            int64_t cur_elems = ((hidden_size - offset) < tile_elems ? (hidden_size - offset) : tile_elems);
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(cur_elems * sizeof(bfloat16_t)), 0, 0, 0};

            auto localIn = inQueue.AllocTensor<bfloat16_t>();
            AscendC::DataCopyPad(localIn, inGm[src_base + offset], copyParams, padParams);
            inQueue.EnQue(localIn);

            localIn = inQueue.DeQue<bfloat16_t>();
            auto localOut = outQueue.AllocTensor<bfloat16_t>();
            AscendC::DataCopy(localOut, localIn, cur_elems);
            outQueue.EnQue(localOut);
            inQueue.FreeTensor(localIn);

            localOut = outQueue.DeQue<bfloat16_t>();
            AscendC::DataCopyPad(outGm[dst_base + offset], localOut, copyParams);
            outQueue.FreeTensor(localOut);
        }
    }
}

int64_t get_moe_pack_aiv_num()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = platform->GetCoreNumAiv();
    return coreNum > 0 ? coreNum : 20;
}

extern "C" void launch_moe_dispatch_pack_bf16(
    GM_ADDR x, GM_ADDR dst_indices, GM_ADDR packed_out,
    int64_t bs, int64_t k, int64_t hidden_size, int64_t numBlocks, void* stream)
{
    moe_dispatch_pack_bf16_kernel<<<numBlocks, nullptr, stream>>>(x, dst_indices, packed_out, bs, k, hidden_size);
}

extern "C" void launch_moe_dispatch_gather_bf16(
    GM_ADDR recv_buf, GM_ADDR src_indices, GM_ADDR expand_out,
    int64_t total_tokens, int64_t hidden_size, int64_t numBlocks, void* stream)
{
    moe_dispatch_gather_bf16_kernel<<<numBlocks, nullptr, stream>>>(recv_buf, src_indices, expand_out, total_tokens, hidden_size);
}

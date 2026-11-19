#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "moe_beat_golden_launch.h"

constexpr static int32_t BUFFER_NUM = 2;
constexpr static int64_t MAX_TILE_BYTES = 65536; // 单队列可安全独享 64KB 双缓冲 (128KB/192KB)
constexpr static int64_t MAX_PAIRS_UB = 8192;
constexpr static int32_t MAX_EXPERTS = 128;
constexpr static int32_t MAX_RANKS = 16;

__global__ __aicore__ void moe_bg_pack_bf16_kernel(
    GM_ADDR x, GM_ADDR expert_ids, GM_ADDR packed_out, GM_ADDR max_rows_out,
    int64_t bs, int64_t k, int64_t hidden_size,
    int64_t ep_world_size, int64_t moe_expert_num, int64_t stride_rows)
{
    AscendC::TPipe pipe;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> dmaQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> idxBuf;

    int64_t total_pairs = bs * k;
    int64_t block_idx = AscendC::GetBlockIdx();
    int64_t block_num = AscendC::GetBlockNum();
    int64_t pairs_per_core = (total_pairs + block_num - 1) / block_num;
    int64_t start_pair = block_idx * pairs_per_core;
    int64_t end_pair = (start_pair + pairs_per_core) < total_pairs ? (start_pair + pairs_per_core) : total_pairs;
    if (start_pair >= end_pair) return;

    int64_t max_elems = MAX_TILE_BYTES / (int64_t)sizeof(bfloat16_t);
    int64_t tile_elems = hidden_size < max_elems ? hidden_size : max_elems;
    uint32_t alloc_bytes = static_cast<uint32_t>(tile_elems * sizeof(bfloat16_t));
    pipe.InitBuffer(dmaQueue, BUFFER_NUM, alloc_bytes);
    pipe.InitBuffer(idxBuf, MAX_PAIRS_UB * sizeof(int32_t));

    AscendC::GlobalTensor<bfloat16_t> xGm, outGm;
    AscendC::GlobalTensor<int32_t> expIdGm;
    xGm.SetGlobalBuffer((__gm__ bfloat16_t*)x);
    expIdGm.SetGlobalBuffer((__gm__ int32_t*)expert_ids);
    outGm.SetGlobalBuffer((__gm__ bfloat16_t*)packed_out);

    auto localExp = idxBuf.Get<int32_t>();
    int64_t aligned_pairs = ((total_pairs + 7) / 8) * 8;
    if (aligned_pairs > MAX_PAIRS_UB) aligned_pairs = MAX_PAIRS_UB;
    AscendC::DataCopyExtParams idxCopy{1, static_cast<uint32_t>(total_pairs * sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> idxPad{true, 0, static_cast<uint8_t>(aligned_pairs - total_pairs), 0};
    AscendC::DataCopyPad(localExp, expIdGm, idxCopy, idxPad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    int32_t experts_per_rank = static_cast<int32_t>(moe_expert_num / ep_world_size);
    int32_t exp_counts[MAX_EXPERTS];
    int32_t prefix_before[MAX_EXPERTS];
    for (int32_t e = 0; e < moe_expert_num; ++e) { exp_counts[e] = 0; prefix_before[e] = 0; }

    for (int64_t p = 0; p < total_pairs; ++p) {
        int32_t eid = localExp.GetValue(p);
        if (p < start_pair) prefix_before[eid]++;
        exp_counts[eid]++;
    }

    int32_t exp_dst_base[MAX_EXPERTS];
    int32_t max_rank_rows = 0;
    for (int32_t r = 0; r < ep_world_size; ++r) {
        int32_t acc = 0;
        for (int32_t le = 0; le < experts_per_rank; ++le) {
            int32_t gid = r * experts_per_rank + le;
            exp_dst_base[gid] = static_cast<int32_t>(r * stride_rows) + acc;
            acc += exp_counts[gid];
        }
        if (acc > max_rank_rows) max_rank_rows = acc;
        if (block_idx == 0) {
            __gm__ int32_t* meta_ptr = reinterpret_cast<__gm__ int32_t*>(
                reinterpret_cast<__gm__ bfloat16_t*>(packed_out) + (r * stride_rows + (stride_rows - 1)) * hidden_size);
            for (int32_t le = 0; le < experts_per_rank; ++le) meta_ptr[le] = exp_counts[r * experts_per_rank + le];
        }
    }
    if (block_idx == 0) {
        reinterpret_cast<__gm__ int32_t*>(max_rows_out)[0] = max_rank_rows;
    }

    // 使用 TQueBind 直通队列 + 原生 32B 对齐 DataCopy 突发传输（绕过 Pad 状态机，零 UB 冗余拷贝！）
    for (int64_t p = start_pair; p < end_pair; ++p) {
        int32_t eid = localExp.GetValue(p);
        int32_t dst_slot = exp_dst_base[eid] + (prefix_before[eid]++);
        int64_t src_base = (p / k) * hidden_size;
        int64_t dst_base = static_cast<int64_t>(dst_slot) * hidden_size;

        for (int64_t offset = 0; offset < hidden_size; offset += tile_elems) {
            int64_t cur_elems = (hidden_size - offset) < tile_elems ? (hidden_size - offset) : tile_elems;
            auto buf = dmaQueue.AllocTensor<bfloat16_t>();
            AscendC::DataCopy(buf, xGm[src_base + offset], cur_elems);
            dmaQueue.EnQue(buf);
            buf = dmaQueue.DeQue<bfloat16_t>();
            AscendC::DataCopy(outGm[dst_base + offset], buf, cur_elems);
            dmaQueue.FreeTensor(buf);
        }
    }
}

__global__ __aicore__ void moe_bg_unpack_bf16_kernel(
    GM_ADDR recv_buf, GM_ADDR expand_out, GM_ADDR ep_recv_counts,
    int64_t hidden_size, int64_t ep_world_size, int64_t moe_expert_num, int64_t stride_rows)
{
    AscendC::TPipe pipe;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> dmaQueue;

    int64_t block_idx = AscendC::GetBlockIdx();
    int64_t block_num = AscendC::GetBlockNum();
    int64_t max_elems = MAX_TILE_BYTES / (int64_t)sizeof(bfloat16_t);
    int64_t tile_elems = hidden_size < max_elems ? hidden_size : max_elems;
    uint32_t alloc_bytes = static_cast<uint32_t>(tile_elems * sizeof(bfloat16_t));
    pipe.InitBuffer(dmaQueue, BUFFER_NUM, alloc_bytes);

    AscendC::GlobalTensor<bfloat16_t> inGm, outGm;
    inGm.SetGlobalBuffer((__gm__ bfloat16_t*)recv_buf);
    outGm.SetGlobalBuffer((__gm__ bfloat16_t*)expand_out);

    int32_t experts_per_rank = static_cast<int32_t>(moe_expert_num / ep_world_size);
    int32_t recv_cnt[MAX_RANKS][MAX_EXPERTS];
    int32_t src_off[MAX_RANKS][MAX_EXPERTS];

    for (int32_t r = 0; r < ep_world_size; ++r) {
        __gm__ int32_t* meta_ptr = reinterpret_cast<__gm__ int32_t*>(
            reinterpret_cast<__gm__ bfloat16_t*>(recv_buf) + (r * stride_rows + (stride_rows - 1)) * hidden_size);
        int32_t acc = static_cast<int32_t>(r * stride_rows);
        for (int32_t le = 0; le < experts_per_rank; ++le) {
            int32_t c = meta_ptr[le];
            recv_cnt[r][le] = c;
            src_off[r][le] = acc;
            acc += c;
        }
    }

    int32_t dst_off[MAX_EXPERTS][MAX_RANKS];
    int32_t total_recv_tokens = 0;
    for (int32_t le = 0; le < experts_per_rank; ++le) {
        int32_t exp_sum = 0;
        for (int32_t r = 0; r < ep_world_size; ++r) {
            dst_off[le][r] = total_recv_tokens;
            total_recv_tokens += recv_cnt[r][le];
            exp_sum += recv_cnt[r][le];
        }
        if (block_idx == 0) reinterpret_cast<__gm__ int32_t*>(ep_recv_counts)[le] = exp_sum;
    }
    if (total_recv_tokens == 0) return;

    int64_t tokens_per_core = (total_recv_tokens + block_num - 1) / block_num;
    int64_t start_tok = block_idx * tokens_per_core;
    int64_t end_tok = (start_tok + tokens_per_core) < total_recv_tokens ? (start_tok + tokens_per_core) : total_recv_tokens;
    if (start_tok >= end_tok) return;

    for (int32_t le = 0; le < experts_per_rank; ++le) {
        for (int32_t r = 0; r < ep_world_size; ++r) {
            int32_t cnt = recv_cnt[r][le];
            if (cnt <= 0) continue;
            int64_t blk_dst_start = dst_off[le][r];
            int64_t ov_start = blk_dst_start > start_tok ? blk_dst_start : start_tok;
            int64_t ov_end = (blk_dst_start + cnt) < end_tok ? (blk_dst_start + cnt) : end_tok;
            for (int64_t t = ov_start; t < ov_end; ++t) {
                int64_t src_base = (static_cast<int64_t>(src_off[r][le]) + (t - blk_dst_start)) * hidden_size;
                int64_t dst_base = t * hidden_size;
                for (int64_t offset = 0; offset < hidden_size; offset += tile_elems) {
                    int64_t cur_elems = (hidden_size - offset) < tile_elems ? (hidden_size - offset) : tile_elems;
                    auto buf = dmaQueue.AllocTensor<bfloat16_t>();
                    AscendC::DataCopy(buf, inGm[src_base + offset], cur_elems);
                    dmaQueue.EnQue(buf);
                    buf = dmaQueue.DeQue<bfloat16_t>();
                    AscendC::DataCopy(outGm[dst_base + offset], buf, cur_elems);
                    dmaQueue.FreeTensor(buf);
                }
            }
        }
    }
}

int64_t get_bg_aiv_num()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = platform->GetCoreNumAiv();
    return coreNum > 0 ? coreNum : 20;
}

extern "C" void launch_moe_bg_pack_bf16(GM_ADDR x, GM_ADDR expert_ids, GM_ADDR packed_out, GM_ADDR max_rows_out, int64_t bs, int64_t k, int64_t hidden_size, int64_t ep_world_size, int64_t moe_expert_num, int64_t stride_rows, int64_t numBlocks, void* stream)
{
    moe_bg_pack_bf16_kernel<<<numBlocks, nullptr, stream>>>(x, expert_ids, packed_out, max_rows_out, bs, k, hidden_size, ep_world_size, moe_expert_num, stride_rows);
}

extern "C" void launch_moe_bg_unpack_bf16(GM_ADDR recv_buf, GM_ADDR expand_out, GM_ADDR ep_recv_counts, int64_t hidden_size, int64_t ep_world_size, int64_t moe_expert_num, int64_t stride_rows, int64_t numBlocks, void* stream)
{
    moe_bg_unpack_bf16_kernel<<<numBlocks, nullptr, stream>>>(recv_buf, expand_out, ep_recv_counts, hidden_size, ep_world_size, moe_expert_num, stride_rows);
}

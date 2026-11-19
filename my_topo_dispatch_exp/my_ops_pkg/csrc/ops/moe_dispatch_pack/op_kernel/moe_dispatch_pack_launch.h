#ifndef MOE_DISPATCH_PACK_LAUNCH_H
#define MOE_DISPATCH_PACK_LAUNCH_H
#include <cstdint>
#ifndef GM_ADDR
#define GM_ADDR void*
#endif
int64_t get_moe_pack_aiv_num();
extern "C" {
void launch_moe_dispatch_pack_bf16(GM_ADDR x, GM_ADDR dst_indices, GM_ADDR packed_out, int64_t bs, int64_t k, int64_t hidden_size, int64_t numBlocks, void* stream);
void launch_moe_dispatch_gather_bf16(GM_ADDR recv_buf, GM_ADDR src_indices, GM_ADDR expand_out, int64_t total_tokens, int64_t hidden_size, int64_t numBlocks, void* stream);
}
#endif

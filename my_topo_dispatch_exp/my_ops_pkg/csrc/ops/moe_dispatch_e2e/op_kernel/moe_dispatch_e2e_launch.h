#ifndef MOE_DISPATCH_E2E_LAUNCH_H
#define MOE_DISPATCH_E2E_LAUNCH_H
#include <cstdint>
#ifndef GM_ADDR
#define GM_ADDR void*
#endif
int64_t get_e2e_aiv_num();
extern "C" {
void launch_moe_e2e_pack_bf16(GM_ADDR x, GM_ADDR expert_ids, GM_ADDR packed_out, int64_t bs, int64_t k, int64_t hidden_size, int64_t ep_world_size, int64_t moe_expert_num, int64_t chunk_rows, int64_t numBlocks, void* stream);
void launch_moe_e2e_unpack_bf16(GM_ADDR recv_buf, GM_ADDR expand_out, GM_ADDR ep_recv_counts, int64_t hidden_size, int64_t ep_world_size, int64_t moe_expert_num, int64_t chunk_rows, int64_t numBlocks, void* stream);
}
#endif

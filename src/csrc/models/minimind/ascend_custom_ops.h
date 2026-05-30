#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace minimind::model {

struct DeviceTensor;

struct CustomAttentionCache;

bool custom_ops_available();

std::vector<float> custom_add(const std::vector<float>& lhs,
                              const std::vector<float>& rhs);

std::vector<float> custom_rms_norm(const std::vector<float>& input,
                                   const std::vector<float>& weight,
                                   float eps);

std::vector<float> custom_swiglu(const std::vector<float>& gate,
                                 const std::vector<float>& up);

std::vector<float> custom_rope(const std::vector<float>& input,
                               int64_t heads,
                               int64_t head_dim,
                               int64_t position,
                               float theta);

std::vector<float> custom_attention(const std::vector<float>& query,
                                    const std::vector<float>& keys,
                                    const std::vector<float>& values,
                                    int64_t tokens,
                                    int64_t q_heads,
                                    int64_t kv_heads,
                                    int64_t head_dim);

std::vector<float> custom_prefill_attention(const std::vector<float>& query,
                                            const std::vector<float>& keys,
                                            const std::vector<float>& values,
                                            std::shared_ptr<CustomAttentionCache>& cache,
                                            int64_t seq_len,
                                            int64_t q_heads,
                                            int64_t kv_heads,
                                            int64_t head_dim);

std::vector<float> custom_attention_cached(const std::vector<float>& query,
                                           const std::vector<float>& keys,
                                           const std::vector<float>& values,
                                           std::shared_ptr<CustomAttentionCache>& cache,
                                           int64_t tokens,
                                           int64_t q_heads,
                                           int64_t kv_heads,
                                           int64_t head_dim);

#if defined(MINIMIND_USE_CUSTOM_ASCEND_OPS)
std::vector<float> custom_rms_norm_rows(const std::vector<float>& input,
                                         const std::vector<float>& weight,
                                         int64_t rows,
                                         int64_t cols,
                                         float eps);

std::vector<float> custom_swiglu_rows(const std::vector<float>& gate_up,
                                      int64_t rows,
                                      int64_t cols);
#endif

#if defined(MINIMIND_USE_ASCEND)
void device_add(const DeviceTensor& lhs, const DeviceTensor& rhs, DeviceTensor& out);

void device_rms_norm_rows(const DeviceTensor& input,
                          const DeviceTensor& weight,
                          float eps,
                          int64_t rows,
                          int64_t cols,
                          DeviceTensor& out);

void device_swiglu_rows(const DeviceTensor& gate_up, int64_t rows, int64_t cols, DeviceTensor& out);

void device_qk_norm_rope(DeviceTensor& query,
                         DeviceTensor& key,
                         const DeviceTensor& q_weight,
                         const DeviceTensor& k_weight,
                         int64_t q_heads,
                         int64_t kv_heads,
                         int64_t head_dim,
                         int64_t position,
                         float theta);

void device_qkv_split_rows(const DeviceTensor& qkv,
                           int64_t rows,
                           int64_t q_size,
                           int64_t kv_size,
                           DeviceTensor& query,
                           DeviceTensor& key,
                           DeviceTensor& value);

void device_copy_last_row(const DeviceTensor& input, int64_t rows, int64_t cols, DeviceTensor& out);

void device_rope_rows(DeviceTensor& tensor,
                      int64_t rows,
                      int64_t heads,
                      int64_t head_dim,
                      int64_t start_position,
                      int64_t max_positions,
                      float theta);

void device_embedding_gather(const DeviceTensor& weight,
                             const DeviceTensor& token_ids,
                             int64_t tokens,
                             int64_t vocab_size,
                             int64_t hidden_size,
                             DeviceTensor& out);

void device_embedding_gather(const DeviceTensor& weight,
                             const std::vector<int32_t>& token_ids,
                             int64_t vocab_size,
                             int64_t hidden_size,
                             DeviceTensor& out);

void device_prefill_attention(const DeviceTensor& query,
                              const DeviceTensor& keys,
                              const DeviceTensor& values,
                              std::shared_ptr<CustomAttentionCache>& cache,
                              int64_t seq_len,
                              int64_t q_heads,
                              int64_t kv_heads,
                              int64_t head_dim,
                              DeviceTensor& out);

void device_decode_qkv_postprocess(const DeviceTensor& qkv,
                                   const DeviceTensor& q_weight,
                                   const DeviceTensor& k_weight,
                                   std::shared_ptr<CustomAttentionCache>& cache,
                                   int64_t tokens,
                                   int64_t q_heads,
                                   int64_t kv_heads,
                                   int64_t head_dim,
                                   int64_t position,
                                   int64_t max_positions,
                                   float theta,
                                   DeviceTensor& query);

void device_attention_from_cache(const DeviceTensor& query,
                                 const std::shared_ptr<CustomAttentionCache>& cache,
                                 int64_t tokens,
                                 int64_t q_heads,
                                 int64_t kv_heads,
                                 int64_t head_dim,
                                 DeviceTensor& out);

void device_decode_attention(const DeviceTensor& query,
                             const DeviceTensor& key,
                             const DeviceTensor& value,
                             std::shared_ptr<CustomAttentionCache>& cache,
                             int64_t tokens,
                             int64_t q_heads,
                             int64_t kv_heads,
                             int64_t head_dim,
                             DeviceTensor& out);
#endif

}  // namespace minimind::model

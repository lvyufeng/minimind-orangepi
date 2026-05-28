#pragma once

#include <cstdint>
#include <vector>

namespace minimind::model {

bool custom_ops_available();

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

}  // namespace minimind::model

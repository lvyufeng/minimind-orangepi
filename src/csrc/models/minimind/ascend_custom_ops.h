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

}  // namespace minimind::model

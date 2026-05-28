#pragma once

#include <cstdint>
#include <vector>

namespace minimind::model {

std::vector<float> cube_matvec(const std::vector<float>& matrix,
                               int64_t rows,
                               int64_t cols,
                               const std::vector<float>& input);

bool cube_matvec_available();

}  // namespace minimind::model

#pragma once

#include "config.h"

#include <cstdint>
#include <vector>

namespace minimind::model_v {

struct ImageTokenSpan {
  int64_t start = 0;
  int64_t length = 0;
};

std::vector<ImageTokenSpan> find_image_token_spans(const std::vector<int32_t>& input_ids,
                                                   int32_t image_token_id);

std::vector<float> replace_image_token_embeddings(
    const MiniMindVConfig& config,
    const std::vector<int32_t>& input_ids,
    const std::vector<float>& token_embeddings,
    const std::vector<float>& image_embeddings);

}  // namespace minimind::model_v

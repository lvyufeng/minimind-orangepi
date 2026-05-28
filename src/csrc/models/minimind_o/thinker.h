#pragma once

#include "config.h"

#include <cstdint>
#include <vector>

namespace minimind::model_o {

struct ThinkerInputs {
  std::vector<int32_t> input_ids;
  std::vector<float> audio_embeddings;
  std::vector<float> image_embeddings;
};

class ThinkerBoundary {
 public:
  explicit ThinkerBoundary(MiniMindOConfig config);
  const MiniMindOConfig& config() const noexcept { return config_; }
  int64_t bridge_layer() const noexcept { return config_.bridge_layer; }
  bool has_audio(const ThinkerInputs& inputs) const noexcept;
  bool has_image(const ThinkerInputs& inputs) const noexcept;

 private:
  MiniMindOConfig config_;
};

}  // namespace minimind::model_o

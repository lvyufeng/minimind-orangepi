#include "vision_encoder.h"

namespace minimind::model_v {

std::vector<float> make_synthetic_image_features(const MiniMindVConfig& config) {
  std::vector<float> features(static_cast<std::size_t>(config.image_token_len * config.image_hidden_size));
  for (int64_t token = 0; token < config.image_token_len; ++token) {
    for (int64_t dim = 0; dim < config.image_hidden_size; ++dim) {
      features[static_cast<std::size_t>(token * config.image_hidden_size + dim)] =
          static_cast<float>((token + dim) % 17) / 16.0F;
    }
  }
  return features;
}

}  // namespace minimind::model_v

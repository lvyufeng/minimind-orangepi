#include "rope.h"

#include <cmath>
#include <stdexcept>

namespace minimind::model {

namespace {

float yarn_ramp(float position, float low, float high) {
  if (low == high) {
    return position >= high ? 1.0F : 0.0F;
  }
  float value = (position - low) / (high - low);
  if (value < 0.0F) {
    return 0.0F;
  }
  if (value > 1.0F) {
    return 1.0F;
  }
  return value;
}

float inverse_frequency(const MiniMindConfig& config, int64_t dim_index) {
  const float base = std::pow(config.rope_theta,
                              static_cast<float>(dim_index) / static_cast<float>(config.head_dim));
  float inv_freq = 1.0F / base;

  if (!config.rope_scaling.enabled) {
    return inv_freq;
  }

  const float scaled = inv_freq / config.rope_scaling.factor;
  const float low = static_cast<float>(config.head_dim) / config.rope_scaling.beta_fast;
  const float high = static_cast<float>(config.head_dim) / config.rope_scaling.beta_slow;
  const float ramp = yarn_ramp(static_cast<float>(dim_index), low, high);
  return scaled * (1.0F - ramp) + inv_freq * ramp;
}

}  // namespace

RopeTables make_rope_tables(const MiniMindConfig& config, int64_t positions) {
  const auto errors = validate_config(config);
  if (!errors.empty()) {
    throw std::invalid_argument("invalid MiniMind config for RoPE: " + errors.front());
  }
  if (positions <= 0) {
    throw std::invalid_argument("RoPE positions must be positive");
  }
  if (config.head_dim % 2 != 0) {
    throw std::invalid_argument("RoPE head_dim must be even");
  }

  RopeTables tables{
      HostTensor(TensorShape({positions, config.head_dim}), DType::kFloat32),
      HostTensor(TensorShape({positions, config.head_dim}), DType::kFloat32),
  };

  auto* cos_data = tables.cos.data_as<float>();
  auto* sin_data = tables.sin.data_as<float>();
  for (int64_t pos = 0; pos < positions; ++pos) {
    for (int64_t dim = 0; dim < config.head_dim / 2; ++dim) {
      const float inv_freq = inverse_frequency(config, dim * 2);
      const float angle = static_cast<float>(pos) * inv_freq;
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      cos_data[pos * config.head_dim + dim] = c;
      sin_data[pos * config.head_dim + dim] = s;
      cos_data[pos * config.head_dim + dim + config.head_dim / 2] = c;
      sin_data[pos * config.head_dim + dim + config.head_dim / 2] = s;
    }
  }

  return tables;
}

}  // namespace minimind::model

#include "../../src/csrc/models/minimind/ascend_custom_ops.h"

#include <cmath>
#include <iostream>
#include <vector>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "check failed: " #condition << '\n';                       \
      return 1;                                                                \
    }                                                                          \
  } while (false)

int main() {
  if (!minimind::model::custom_ops_available()) {
    return 0;
  }

  std::vector<float> input(128);
  std::vector<float> weight(128);
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125F;
    weight[i] = 0.5F + static_cast<float>(i % 7) * 0.03125F;
  }

  const float eps = 1e-5F;
  const auto normalized = minimind::model::custom_rms_norm(input, weight, eps);
  CHECK(normalized.size() == input.size());

  float sum = 0.0F;
  for (float value : input) {
    sum += value * value;
  }
  const float scale = 1.0F / std::sqrt(sum / static_cast<float>(input.size()) + eps);
  for (std::size_t i = 0; i < input.size(); ++i) {
    const float expected = input[i] * scale * weight[i];
    CHECK(std::fabs(normalized[i] - expected) < 5e-2F);
  }

  std::vector<float> gate(128);
  std::vector<float> up(128);
  for (std::size_t i = 0; i < gate.size(); ++i) {
    gate[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.125F;
    up[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.0625F;
  }

  const auto activated = minimind::model::custom_swiglu(gate, up);
  CHECK(activated.size() == gate.size());
  for (std::size_t i = 0; i < gate.size(); ++i) {
    const float silu = gate[i] / (1.0F + std::exp(-gate[i]));
    CHECK(std::fabs(activated[i] - silu * up[i]) < 5e-2F);
  }

  std::vector<float> rope_input(192);
  for (std::size_t i = 0; i < rope_input.size(); ++i) {
    rope_input[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.03125F;
  }
  const int64_t heads = 2;
  const int64_t head_dim = 96;
  const int64_t position = 7;
  const float theta = 1000000.0F;
  const auto rope_output = minimind::model::custom_rope(rope_input, heads, head_dim, position, theta);
  CHECK(rope_output.size() == rope_input.size());
  for (int64_t head = 0; head < heads; ++head) {
    const std::size_t base = static_cast<std::size_t>(head * head_dim);
    for (int64_t dim = 0; dim < head_dim / 2; ++dim) {
      const float inv_freq = 1.0F / std::pow(theta, static_cast<float>(dim * 2) / static_cast<float>(head_dim));
      const float angle = static_cast<float>(position) * inv_freq;
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      const float first = rope_input[base + static_cast<std::size_t>(dim)];
      const float second = rope_input[base + static_cast<std::size_t>(dim + head_dim / 2)];
      CHECK(std::fabs(rope_output[base + static_cast<std::size_t>(dim)] - (first * c - second * s)) < 5e-2F);
      CHECK(std::fabs(rope_output[base + static_cast<std::size_t>(dim + head_dim / 2)] - (second * c + first * s)) < 5e-2F);
    }
  }

  const int64_t attn_tokens = 5;
  const int64_t attn_q_heads = 4;
  const int64_t attn_kv_heads = 2;
  const int64_t attn_head_dim = 16;
  std::vector<float> query(static_cast<std::size_t>(attn_q_heads * attn_head_dim));
  std::vector<float> keys(static_cast<std::size_t>(attn_tokens * attn_kv_heads * attn_head_dim));
  std::vector<float> values(keys.size());
  for (std::size_t i = 0; i < query.size(); ++i) {
    query[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.03125F;
  }
  for (std::size_t i = 0; i < keys.size(); ++i) {
    keys[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.025F;
    values[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.0375F;
  }
  const auto attended = minimind::model::custom_attention(query, keys, values, attn_tokens, attn_q_heads,
                                                          attn_kv_heads, attn_head_dim);
  CHECK(attended.size() == query.size());
  const int64_t kv_repeat = attn_q_heads / attn_kv_heads;
  for (int64_t q_head = 0; q_head < attn_q_heads; ++q_head) {
    const int64_t kv_head = q_head / kv_repeat;
    std::vector<float> scores(static_cast<std::size_t>(attn_tokens));
    float max_score = -INFINITY;
    for (int64_t token = 0; token < attn_tokens; ++token) {
      float dot = 0.0F;
      for (int64_t dim = 0; dim < attn_head_dim; ++dim) {
        dot += query[static_cast<std::size_t>(q_head * attn_head_dim + dim)] *
               keys[static_cast<std::size_t>((token * attn_kv_heads + kv_head) * attn_head_dim + dim)];
      }
      const float score = dot / std::sqrt(static_cast<float>(attn_head_dim));
      scores[static_cast<std::size_t>(token)] = score;
      max_score = std::max(max_score, score);
    }
    float denom = 0.0F;
    for (float& score : scores) {
      score = std::exp(score - max_score);
      denom += score;
    }
    for (int64_t dim = 0; dim < attn_head_dim; ++dim) {
      float expected = 0.0F;
      for (int64_t token = 0; token < attn_tokens; ++token) {
        const float weight_value = scores[static_cast<std::size_t>(token)] / denom;
        expected += weight_value * values[static_cast<std::size_t>((token * attn_kv_heads + kv_head) * attn_head_dim + dim)];
      }
      CHECK(std::fabs(attended[static_cast<std::size_t>(q_head * attn_head_dim + dim)] - expected) < 5e-2F);
    }
  }

  return 0;
}

#include "decoder_layer.h"
#include "ascend_matmul.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace minimind::model {

namespace {

void require_size(const std::vector<float>& values, int64_t expected, const char* name) {
  if (static_cast<int64_t>(values.size()) != expected) {
    throw std::invalid_argument(std::string(name) + " has invalid size");
  }
}

std::vector<float> rms_norm(const std::vector<float>& input,
                            const std::vector<float>& weight,
                            float eps) {
  require_size(weight, static_cast<int64_t>(input.size()), "rms_norm weight");
  float sum = 0.0F;
  for (float value : input) {
    sum += value * value;
  }
  const float scale = 1.0F / std::sqrt(sum / static_cast<float>(input.size()) + eps);
  std::vector<float> output(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    output[i] = input[i] * scale * weight[i];
  }
  return output;
}

std::vector<float> cpu_matvec(const std::vector<float>& matrix,
                              int64_t rows,
                              int64_t cols,
                              const std::vector<float>& input) {
  std::vector<float> output(static_cast<std::size_t>(rows), 0.0F);
  for (int64_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    for (int64_t col = 0; col < cols; ++col) {
      sum += matrix[static_cast<std::size_t>(row * cols + col)] * input[static_cast<std::size_t>(col)];
    }
    output[static_cast<std::size_t>(row)] = sum;
  }
  return output;
}

std::vector<float> matvec(const std::vector<float>& matrix,
                          int64_t rows,
                          int64_t cols,
                          const std::vector<float>& input) {
  require_size(matrix, rows * cols, "matrix");
  require_size(input, cols, "matvec input");
  if (cube_matvec_available() && rows >= 128 && cols >= 128) {
    return cube_matvec(matrix, rows, cols, input);
  }
  return cpu_matvec(matrix, rows, cols, input);
}

void apply_rope(std::vector<float>& values, int64_t heads, int64_t head_dim, int64_t position, float theta) {
  for (int64_t head = 0; head < heads; ++head) {
    float* base = values.data() + head * head_dim;
    for (int64_t dim = 0; dim < head_dim / 2; ++dim) {
      const float inv_freq = 1.0F / std::pow(theta, static_cast<float>(dim * 2) / static_cast<float>(head_dim));
      const float angle = static_cast<float>(position) * inv_freq;
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      const float first = base[dim];
      const float second = base[dim + head_dim / 2];
      base[dim] = first * c - second * s;
      base[dim + head_dim / 2] = second * c + first * s;
    }
  }
}

float silu(float value) {
  return value / (1.0F + std::exp(-value));
}

std::vector<float> attention_step(const MiniMindConfig& config,
                                  const std::vector<float>& query,
                                  const LayerKvCache& cache) {
  const int64_t q_heads = config.num_attention_heads;
  const int64_t kv_heads = config.num_key_value_heads;
  const int64_t head_dim = config.head_dim;
  const int64_t kv_repeat = q_heads / kv_heads;
  std::vector<float> output(static_cast<std::size_t>(q_heads * head_dim), 0.0F);

  for (int64_t q_head = 0; q_head < q_heads; ++q_head) {
    const int64_t kv_head = q_head / kv_repeat;
    std::vector<float> scores(static_cast<std::size_t>(cache.tokens));
    float max_score = -INFINITY;
    for (int64_t token = 0; token < cache.tokens; ++token) {
      float dot = 0.0F;
      for (int64_t dim = 0; dim < head_dim; ++dim) {
        const float q = query[static_cast<std::size_t>(q_head * head_dim + dim)];
        const float k = cache.keys[static_cast<std::size_t>((token * kv_heads + kv_head) * head_dim + dim)];
        dot += q * k;
      }
      const float score = dot / std::sqrt(static_cast<float>(head_dim));
      scores[static_cast<std::size_t>(token)] = score;
      max_score = std::max(max_score, score);
    }

    float denom = 0.0F;
    for (float& score : scores) {
      score = std::exp(score - max_score);
      denom += score;
    }

    for (int64_t token = 0; token < cache.tokens; ++token) {
      const float weight = scores[static_cast<std::size_t>(token)] / denom;
      for (int64_t dim = 0; dim < head_dim; ++dim) {
        const float v = cache.values[static_cast<std::size_t>((token * kv_heads + kv_head) * head_dim + dim)];
        output[static_cast<std::size_t>(q_head * head_dim + dim)] += weight * v;
      }
    }
  }

  return output;
}

}  // namespace

std::vector<float> run_dense_decoder_layer(
    const MiniMindConfig& config,
    const DenseLayerWeights& weights,
    LayerKvCache& cache,
    const std::vector<float>& hidden,
    int64_t position) {
  require_size(hidden, config.hidden_size, "hidden");
  require_size(weights.input_norm, config.hidden_size, "input_norm");
  require_size(weights.post_attention_norm, config.hidden_size, "post_attention_norm");
  require_size(weights.q_norm, config.head_dim, "q_norm");
  require_size(weights.k_norm, config.head_dim, "k_norm");

  const int64_t q_size = config.num_attention_heads * config.head_dim;
  const int64_t kv_size = config.num_key_value_heads * config.head_dim;

  auto normed = rms_norm(hidden, weights.input_norm, config.rms_norm_eps);
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  if (!weights.qkv_proj.empty()) {
    const auto qkv = matvec(weights.qkv_proj, q_size + 2 * kv_size, config.hidden_size, normed);
    query.assign(qkv.begin(), qkv.begin() + q_size);
    key.assign(qkv.begin() + q_size, qkv.begin() + q_size + kv_size);
    value.assign(qkv.begin() + q_size + kv_size, qkv.end());
  } else {
    query = matvec(weights.q_proj, q_size, config.hidden_size, normed);
    key = matvec(weights.k_proj, kv_size, config.hidden_size, normed);
    value = matvec(weights.v_proj, kv_size, config.hidden_size, normed);
  }

  for (int64_t head = 0; head < config.num_attention_heads; ++head) {
    std::vector<float> slice(query.begin() + head * config.head_dim,
                             query.begin() + (head + 1) * config.head_dim);
    slice = rms_norm(slice, weights.q_norm, config.rms_norm_eps);
    std::copy(slice.begin(), slice.end(), query.begin() + head * config.head_dim);
  }
  for (int64_t head = 0; head < config.num_key_value_heads; ++head) {
    std::vector<float> slice(key.begin() + head * config.head_dim,
                             key.begin() + (head + 1) * config.head_dim);
    slice = rms_norm(slice, weights.k_norm, config.rms_norm_eps);
    std::copy(slice.begin(), slice.end(), key.begin() + head * config.head_dim);
  }

  apply_rope(query, config.num_attention_heads, config.head_dim, position, config.rope_theta);
  apply_rope(key, config.num_key_value_heads, config.head_dim, position, config.rope_theta);

  cache.keys.insert(cache.keys.end(), key.begin(), key.end());
  cache.values.insert(cache.values.end(), value.begin(), value.end());
  cache.tokens += 1;

  auto context = attention_step(config, query, cache);
  auto attention_out = matvec(weights.o_proj, config.hidden_size, q_size, context);

  std::vector<float> residual(hidden.size());
  for (std::size_t i = 0; i < hidden.size(); ++i) {
    residual[i] = hidden[i] + attention_out[i];
  }

  auto post_normed = rms_norm(residual, weights.post_attention_norm, config.rms_norm_eps);
  std::vector<float> mlp;
  if (config.use_moe) {
    require_size(weights.moe_gate, config.num_experts * config.hidden_size, "moe_gate");
    if (static_cast<int64_t>(weights.experts.size()) != config.num_experts) {
      throw std::invalid_argument("expert count does not match config");
    }
    const auto router = matvec(weights.moe_gate, config.num_experts, config.hidden_size, post_normed);
    const int64_t expert_index = static_cast<int64_t>(
        std::distance(router.begin(), std::max_element(router.begin(), router.end())));
    const auto& expert = weights.experts[static_cast<std::size_t>(expert_index)];
    std::vector<float> gate;
    std::vector<float> up;
    if (!expert.gate_up_proj.empty()) {
      const auto gate_up = matvec(expert.gate_up_proj, 2 * config.moe_intermediate_size, config.hidden_size, post_normed);
      gate.assign(gate_up.begin(), gate_up.begin() + config.moe_intermediate_size);
      up.assign(gate_up.begin() + config.moe_intermediate_size, gate_up.end());
    } else {
      gate = matvec(expert.gate_proj, config.moe_intermediate_size, config.hidden_size, post_normed);
      up = matvec(expert.up_proj, config.moe_intermediate_size, config.hidden_size, post_normed);
    }
    std::vector<float> activated(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
      activated[i] = silu(gate[i]) * up[i];
    }
    mlp = matvec(expert.down_proj, config.hidden_size, config.moe_intermediate_size, activated);
  } else {
    std::vector<float> gate;
    std::vector<float> up;
    if (!weights.gate_up_proj.empty()) {
      const auto gate_up = matvec(weights.gate_up_proj, 2 * config.intermediate_size, config.hidden_size, post_normed);
      gate.assign(gate_up.begin(), gate_up.begin() + config.intermediate_size);
      up.assign(gate_up.begin() + config.intermediate_size, gate_up.end());
    } else {
      gate = matvec(weights.gate_proj, config.intermediate_size, config.hidden_size, post_normed);
      up = matvec(weights.up_proj, config.intermediate_size, config.hidden_size, post_normed);
    }
    std::vector<float> activated(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
      activated[i] = silu(gate[i]) * up[i];
    }
    mlp = matvec(weights.down_proj, config.hidden_size, config.intermediate_size, activated);
  }

  for (std::size_t i = 0; i < residual.size(); ++i) {
    residual[i] += mlp[i];
  }
  return residual;
}

}  // namespace minimind::model

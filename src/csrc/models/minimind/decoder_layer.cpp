#include "decoder_layer.h"
#include "ascend_custom_ops.h"
#include "ascend_matmul.h"
#if defined(MINIMIND_USE_ASCEND)
#include "ascend_weights.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace minimind::model {

namespace {

void require_size(const std::vector<float>& values, int64_t expected, const char* name) {
  if (static_cast<int64_t>(values.size()) != expected) {
    throw std::invalid_argument(std::string(name) + " has invalid size");
  }
}

bool use_host_vector_custom_ops() {
  return false;
}

void rms_norm_inplace(float* values, const std::vector<float>& weight, int64_t size, float eps) {
  require_size(weight, size, "rms_norm weight");
  float sum = 0.0F;
  for (int64_t i = 0; i < size; ++i) {
    sum += values[static_cast<std::size_t>(i)] * values[static_cast<std::size_t>(i)];
  }
  const float scale = 1.0F / std::sqrt(sum / static_cast<float>(size) + eps);
  for (int64_t i = 0; i < size; ++i) {
    values[static_cast<std::size_t>(i)] *= scale * weight[static_cast<std::size_t>(i)];
  }
}

std::vector<float> cpu_rms_norm(const std::vector<float>& input,
                                const std::vector<float>& weight,
                                float eps) {
  std::vector<float> output = input;
  rms_norm_inplace(output.data(), weight, static_cast<int64_t>(output.size()), eps);
  return output;
}

std::vector<float> rms_norm(const std::vector<float>& input,
                            const std::vector<float>& weight,
                            float eps) {
  require_size(weight, static_cast<int64_t>(input.size()), "rms_norm weight");
  if (use_host_vector_custom_ops() && custom_ops_available() && input.size() >= 128) {
    try {
      return custom_rms_norm(input, weight, eps);
    } catch (const std::exception&) {
    }
  }
  return cpu_rms_norm(input, weight, eps);
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

struct RopeTables {
  std::vector<float> cos;
  std::vector<float> sin;
};

const RopeTables& rope_tables(int64_t max_position, int64_t head_dim, float theta) {
  static int64_t cached_max_position = 0;
  static int64_t cached_head_dim = 0;
  static float cached_theta = 0.0F;
  static RopeTables cached;
  if (cached_max_position == max_position && cached_head_dim == head_dim && cached_theta == theta) {
    return cached;
  }

  cached_max_position = max_position;
  cached_head_dim = head_dim;
  cached_theta = theta;
  const int64_t half_dim = head_dim / 2;
  cached.cos.resize(static_cast<std::size_t>(max_position * half_dim));
  cached.sin.resize(static_cast<std::size_t>(max_position * half_dim));
  for (int64_t position = 0; position < max_position; ++position) {
    for (int64_t dim = 0; dim < half_dim; ++dim) {
      const float inv_freq = 1.0F / std::pow(theta, static_cast<float>(dim * 2) / static_cast<float>(head_dim));
      const float angle = static_cast<float>(position) * inv_freq;
      const std::size_t index = static_cast<std::size_t>(position * half_dim + dim);
      cached.cos[index] = std::cos(angle);
      cached.sin[index] = std::sin(angle);
    }
  }
  return cached;
}

void apply_rope(std::vector<float>& values,
                int64_t heads,
                int64_t head_dim,
                int64_t position,
                float theta,
                int64_t max_position) {
  require_size(values, heads * head_dim, "rope input");
  if (use_host_vector_custom_ops() && custom_ops_available() && values.size() >= 128) {
    try {
      values = custom_rope(values, heads, head_dim, position, theta);
      return;
    } catch (const std::exception&) {
    }
  }

  const int64_t half_dim = head_dim / 2;
  const auto& tables = rope_tables(max_position, head_dim, theta);
  const std::size_t table_base = static_cast<std::size_t>(position * half_dim);
  for (int64_t head = 0; head < heads; ++head) {
    float* base = values.data() + head * head_dim;
    for (int64_t dim = 0; dim < half_dim; ++dim) {
      const float c = tables.cos[table_base + static_cast<std::size_t>(dim)];
      const float s = tables.sin[table_base + static_cast<std::size_t>(dim)];
      const float first = base[dim];
      const float second = base[dim + half_dim];
      base[dim] = first * c - second * s;
      base[dim + half_dim] = second * c + first * s;
    }
  }
}

float silu(float value) {
  return value / (1.0F + std::exp(-value));
}

std::vector<float> swiglu(const std::vector<float>& gate, const std::vector<float>& up) {
  require_size(up, static_cast<int64_t>(gate.size()), "swiglu up");
  if (use_host_vector_custom_ops() && custom_ops_available() && gate.size() >= 128) {
    try {
      return custom_swiglu(gate, up);
    } catch (const std::exception&) {
    }
  }

  std::vector<float> activated(gate.size());
  for (std::size_t i = 0; i < gate.size(); ++i) {
    activated[i] = silu(gate[i]) * up[i];
  }
  return activated;
}

std::vector<float> swiglu_from_gate_up(const std::vector<float>& gate_up) {
  const std::size_t size = gate_up.size() / 2;
  std::vector<float> activated(size);
  for (std::size_t i = 0; i < size; ++i) {
    activated[i] = silu(gate_up[i]) * gate_up[i + size];
  }
  return activated;
}

std::vector<float> attention_step(const MiniMindConfig& config,
                                  const std::vector<float>& query,
                                  const std::vector<float>& key,
                                  const std::vector<float>& value,
                                  LayerKvCache& cache) {
  const int64_t q_heads = config.num_attention_heads;
  const int64_t kv_heads = config.num_key_value_heads;
  const int64_t head_dim = config.head_dim;
  const int64_t kv_repeat = q_heads / kv_heads;
  if (custom_ops_available() && cache.tokens > 0 && cache.tokens <= 2048 && head_dim <= 128) {
    try {
      return custom_attention_cached(query, key, value, cache.custom_attention_cache, cache.tokens, q_heads, kv_heads,
                                     head_dim);
    } catch (const std::exception&) {
    }
  }
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

std::vector<float> prefill_matmul(const std::vector<float>& matrix,
                                  int64_t rows,
                                  int64_t cols,
                                  const std::vector<float>& input,
                                  int64_t batch) {
  require_size(matrix, rows * cols, "prefill matrix");
  require_size(input, batch * cols, "prefill matmul input");
  if (!cube_matmul_available()) {
    throw std::runtime_error("prefill Cube matmul requires Ascend");
  }
  return cube_matmul(matrix, rows, cols, input, batch);
}

void rms_norm_rows_inplace(std::vector<float>& values,
                           const std::vector<float>& weight,
                           int64_t rows,
                           int64_t cols,
                           float eps) {
  require_size(values, rows * cols, "rms_norm rows input");
  require_size(weight, cols, "rms_norm rows weight");
  for (int64_t row = 0; row < rows; ++row) {
    rms_norm_inplace(values.data() + row * cols, weight, cols, eps);
  }
}

void apply_rope_row(float* values,
                    int64_t heads,
                    int64_t head_dim,
                    int64_t position,
                    float theta,
                    int64_t max_position) {
  const int64_t half_dim = head_dim / 2;
  const auto& tables = rope_tables(max_position, head_dim, theta);
  const std::size_t table_base = static_cast<std::size_t>(position * half_dim);
  for (int64_t head = 0; head < heads; ++head) {
    float* base = values + head * head_dim;
    for (int64_t dim = 0; dim < half_dim; ++dim) {
      const float c = tables.cos[table_base + static_cast<std::size_t>(dim)];
      const float s = tables.sin[table_base + static_cast<std::size_t>(dim)];
      const float first = base[dim];
      const float second = base[dim + half_dim];
      base[dim] = first * c - second * s;
      base[dim + half_dim] = second * c + first * s;
    }
  }
}

std::vector<float> swiglu_rows_from_gate_up(const std::vector<float>& gate_up, int64_t rows, int64_t cols) {
  require_size(gate_up, rows * cols * 2, "prefill gate_up");
  std::vector<float> activated(static_cast<std::size_t>(rows * cols));
  for (int64_t row = 0; row < rows; ++row) {
    const float* gate = gate_up.data() + row * cols * 2;
    const float* up = gate + cols;
    float* out = activated.data() + row * cols;
    for (int64_t col = 0; col < cols; ++col) {
      out[col] = silu(gate[col]) * up[col];
    }
  }
  return activated;
}

std::vector<float> prefill_attention(const MiniMindConfig& config,
                                     LayerKvCache& cache,
                                     const std::vector<float>& query,
                                     const std::vector<float>& keys,
                                     const std::vector<float>& values,
                                     int64_t seq_len) {
  if (!custom_ops_available()) {
    throw std::runtime_error("prefill attention requires Ascend custom ops");
  }
  return custom_prefill_attention(query, keys, values, cache.custom_attention_cache, seq_len,
                                  config.num_attention_heads, config.num_key_value_heads, config.head_dim);
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
    rms_norm_inplace(query.data() + head * config.head_dim, weights.q_norm, config.head_dim, config.rms_norm_eps);
  }
  for (int64_t head = 0; head < config.num_key_value_heads; ++head) {
    rms_norm_inplace(key.data() + head * config.head_dim, weights.k_norm, config.head_dim, config.rms_norm_eps);
  }

  apply_rope(query, config.num_attention_heads, config.head_dim, position, config.rope_theta,
             config.max_position_embeddings);
  apply_rope(key, config.num_key_value_heads, config.head_dim, position, config.rope_theta,
             config.max_position_embeddings);

  const bool use_resident_attention = custom_ops_available() && config.hidden_size >= 128 && config.head_dim <= 128;
  if (!use_resident_attention) {
    cache.keys.insert(cache.keys.end(), key.begin(), key.end());
    cache.values.insert(cache.values.end(), value.begin(), value.end());
  }
  cache.tokens += 1;

  auto context = attention_step(config, query, key, value, cache);
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
    std::vector<float> activated;
    if (!expert.gate_up_proj.empty()) {
      const auto gate_up = matvec(expert.gate_up_proj, 2 * config.moe_intermediate_size, config.hidden_size, post_normed);
      activated = swiglu_from_gate_up(gate_up);
    } else {
      const auto gate = matvec(expert.gate_proj, config.moe_intermediate_size, config.hidden_size, post_normed);
      const auto up = matvec(expert.up_proj, config.moe_intermediate_size, config.hidden_size, post_normed);
      activated = swiglu(gate, up);
    }
    mlp = matvec(expert.down_proj, config.hidden_size, config.moe_intermediate_size, activated);
  } else {
    std::vector<float> activated;
    if (!weights.gate_up_proj.empty()) {
      const auto gate_up = matvec(weights.gate_up_proj, 2 * config.intermediate_size, config.hidden_size, post_normed);
      activated = swiglu_from_gate_up(gate_up);
    } else {
      const auto gate = matvec(weights.gate_proj, config.intermediate_size, config.hidden_size, post_normed);
      const auto up = matvec(weights.up_proj, config.intermediate_size, config.hidden_size, post_normed);
      activated = swiglu(gate, up);
    }
    mlp = matvec(weights.down_proj, config.hidden_size, config.intermediate_size, activated);
  }

  for (std::size_t i = 0; i < residual.size(); ++i) {
    residual[i] += mlp[i];
  }
  return residual;
}

std::vector<float> run_dense_prefill_layer(
    const MiniMindConfig& config,
    const DenseLayerWeights& weights,
    LayerKvCache& cache,
    const std::vector<float>& hidden_seq,
    int64_t start_position,
    int64_t seq_len) {
  if (cache.tokens != 0) {
    throw std::invalid_argument("prefill layer requires an empty KV cache");
  }
  require_size(hidden_seq, seq_len * config.hidden_size, "prefill hidden_seq");
  require_size(weights.input_norm, config.hidden_size, "input_norm");
  require_size(weights.post_attention_norm, config.hidden_size, "post_attention_norm");
  require_size(weights.q_norm, config.head_dim, "q_norm");
  require_size(weights.k_norm, config.head_dim, "k_norm");

  const int64_t q_size = config.num_attention_heads * config.head_dim;
  const int64_t kv_size = config.num_key_value_heads * config.head_dim;

  auto normed = hidden_seq;
  rms_norm_rows_inplace(normed, weights.input_norm, seq_len, config.hidden_size, config.rms_norm_eps);

  std::vector<float> query(static_cast<std::size_t>(seq_len * q_size));
  std::vector<float> keys(static_cast<std::size_t>(seq_len * kv_size));
  std::vector<float> values(static_cast<std::size_t>(seq_len * kv_size));
  if (!weights.qkv_proj.empty()) {
    const auto qkv = prefill_matmul(weights.qkv_proj, q_size + 2 * kv_size, config.hidden_size, normed, seq_len);
    for (int64_t token = 0; token < seq_len; ++token) {
      const float* src = qkv.data() + token * (q_size + 2 * kv_size);
      std::copy(src, src + q_size, query.data() + token * q_size);
      std::copy(src + q_size, src + q_size + kv_size, keys.data() + token * kv_size);
      std::copy(src + q_size + kv_size, src + q_size + 2 * kv_size, values.data() + token * kv_size);
    }
  } else {
    query = prefill_matmul(weights.q_proj, q_size, config.hidden_size, normed, seq_len);
    keys = prefill_matmul(weights.k_proj, kv_size, config.hidden_size, normed, seq_len);
    values = prefill_matmul(weights.v_proj, kv_size, config.hidden_size, normed, seq_len);
  }

  for (int64_t token = 0; token < seq_len; ++token) {
    for (int64_t head = 0; head < config.num_attention_heads; ++head) {
      rms_norm_inplace(query.data() + token * q_size + head * config.head_dim, weights.q_norm, config.head_dim,
                       config.rms_norm_eps);
    }
    for (int64_t head = 0; head < config.num_key_value_heads; ++head) {
      rms_norm_inplace(keys.data() + token * kv_size + head * config.head_dim, weights.k_norm, config.head_dim,
                       config.rms_norm_eps);
    }
    apply_rope_row(query.data() + token * q_size, config.num_attention_heads, config.head_dim,
                   start_position + token, config.rope_theta, config.max_position_embeddings);
    apply_rope_row(keys.data() + token * kv_size, config.num_key_value_heads, config.head_dim,
                   start_position + token, config.rope_theta, config.max_position_embeddings);
  }

  const auto context = prefill_attention(config, cache, query, keys, values, seq_len);
  const auto attention_out = prefill_matmul(weights.o_proj, config.hidden_size, q_size, context, seq_len);

  std::vector<float> residual(hidden_seq.size());
  for (std::size_t i = 0; i < hidden_seq.size(); ++i) {
    residual[i] = hidden_seq[i] + attention_out[i];
  }

  auto post_normed = residual;
  rms_norm_rows_inplace(post_normed, weights.post_attention_norm, seq_len, config.hidden_size, config.rms_norm_eps);

  std::vector<float> mlp;
  if (config.use_moe) {
    require_size(weights.moe_gate, config.num_experts * config.hidden_size, "moe_gate");
    if (static_cast<int64_t>(weights.experts.size()) != config.num_experts) {
      throw std::invalid_argument("expert count does not match config");
    }
    mlp.resize(hidden_seq.size());
    const auto router = prefill_matmul(weights.moe_gate, config.num_experts, config.hidden_size, post_normed, seq_len);
    for (int64_t token = 0; token < seq_len; ++token) {
      const float* router_row = router.data() + token * config.num_experts;
      const int64_t expert_index = static_cast<int64_t>(
          std::distance(router_row, std::max_element(router_row, router_row + config.num_experts)));
      const auto& expert = weights.experts[static_cast<std::size_t>(expert_index)];
      std::vector<float> token_input(post_normed.begin() + token * config.hidden_size,
                                     post_normed.begin() + (token + 1) * config.hidden_size);
      std::vector<float> activated;
      if (!expert.gate_up_proj.empty()) {
        const auto gate_up = prefill_matmul(expert.gate_up_proj, 2 * config.moe_intermediate_size, config.hidden_size,
                                            token_input, 1);
        activated = swiglu_rows_from_gate_up(gate_up, 1, config.moe_intermediate_size);
      } else {
        const auto gate = prefill_matmul(expert.gate_proj, config.moe_intermediate_size, config.hidden_size,
                                         token_input, 1);
        const auto up = prefill_matmul(expert.up_proj, config.moe_intermediate_size, config.hidden_size, token_input, 1);
        activated = swiglu(gate, up);
      }
      const auto token_mlp = prefill_matmul(expert.down_proj, config.hidden_size, config.moe_intermediate_size,
                                            activated, 1);
      std::copy(token_mlp.begin(), token_mlp.end(), mlp.begin() + token * config.hidden_size);
    }
  } else {
    std::vector<float> activated;
    if (!weights.gate_up_proj.empty()) {
      const auto gate_up = prefill_matmul(weights.gate_up_proj, 2 * config.intermediate_size, config.hidden_size,
                                          post_normed, seq_len);
      activated = swiglu_rows_from_gate_up(gate_up, seq_len, config.intermediate_size);
    } else {
      const auto gate = prefill_matmul(weights.gate_proj, config.intermediate_size, config.hidden_size,
                                       post_normed, seq_len);
      const auto up = prefill_matmul(weights.up_proj, config.intermediate_size, config.hidden_size, post_normed, seq_len);
      activated = swiglu(gate, up);
    }
    mlp = prefill_matmul(weights.down_proj, config.hidden_size, config.intermediate_size, activated, seq_len);
  }

  for (std::size_t i = 0; i < residual.size(); ++i) {
    residual[i] += mlp[i];
  }

  cache.keys.clear();
  cache.values.clear();
  cache.tokens = seq_len;
  return residual;
}

#if defined(MINIMIND_USE_ASCEND)
namespace {

void require_device_dense(const MiniMindConfig& config) {
  if (config.use_moe) {
    throw std::runtime_error("Ascend device-resident MoE path is not implemented");
  }
}

int64_t shape_elements(const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    count *= dim;
  }
  return shape.empty() ? 0 : count;
}

void prepare_device_tensor(DeviceTensor& tensor, const std::vector<int64_t>& shape) {
  tensor.shape = shape;
  tensor.dtype = ACL_FLOAT16;
  tensor.strides.resize(shape.size());
  int64_t stride = 1;
  for (std::size_t i = shape.size(); i-- > 0;) {
    tensor.strides[i] = stride;
    stride *= shape[i];
  }
  tensor.buffer.reserve(static_cast<std::size_t>(shape_elements(shape)) * sizeof(uint16_t));
}

struct DeviceLayerScratch {
  DeviceTensor normed;
  DeviceTensor qkv;
  DeviceTensor query;
  DeviceTensor key;
  DeviceTensor value;
  DeviceTensor attention_context;
  DeviceTensor attention_out;
  DeviceTensor residual;
  DeviceTensor post_normed;
  DeviceTensor gate_up;
  DeviceTensor activated;
  DeviceTensor mlp;
};

DeviceLayerScratch& device_layer_scratch() {
  static DeviceLayerScratch* scratch = new DeviceLayerScratch();
  return *scratch;
}

bool profile_prefill_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("MINIMIND_PROFILE_PREFILL");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
  }();
  return enabled;
}

class PrefillProfiler {
 public:
  PrefillProfiler(int64_t layer, int64_t seq_len)
      : enabled_(profile_prefill_enabled()), layer_(layer), seq_len_(seq_len) {
    if (enabled_) {
      check_acl(aclrtSynchronizeStream(runtime().stream()), "aclrtSynchronizeStream prefill profile start failed");
      last_ = std::chrono::steady_clock::now();
    }
  }

  void mark(const char* name) {
    if (!enabled_) {
      return;
    }
    check_acl(aclrtSynchronizeStream(runtime().stream()), "aclrtSynchronizeStream prefill profile mark failed");
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - last_).count();
    std::cerr << "prefill_profile layer=" << layer_ << " seq_len=" << seq_len_ << " stage=" << name
              << " ms=" << ms << '\n';
    last_ = now;
  }

 private:
  bool enabled_ = false;
  int64_t layer_ = 0;
  int64_t seq_len_ = 0;
  std::chrono::steady_clock::time_point last_;
};

}  // namespace

void run_dense_decoder_layer_device(const MiniMindConfig& config,
                                    const AscendLayerWeights& weights,
                                    LayerKvCache& cache,
                                    const DeviceTensor& hidden,
                                    int64_t position,
                                    DeviceTensor& out) {
  require_device_dense(config);
  const int64_t q_size = config.num_attention_heads * config.head_dim;
  const int64_t kv_size = config.num_key_value_heads * config.head_dim;
  auto& scratch = device_layer_scratch();

  prepare_device_tensor(scratch.normed, {1, config.hidden_size});
  prepare_device_tensor(scratch.qkv, {1, q_size + kv_size * 2});
  prepare_device_tensor(scratch.query, {1, q_size});
  prepare_device_tensor(scratch.key, {1, kv_size});
  prepare_device_tensor(scratch.value, {1, kv_size});
  prepare_device_tensor(scratch.attention_context, {1, q_size});
  prepare_device_tensor(scratch.attention_out, {1, config.hidden_size});
  prepare_device_tensor(scratch.residual, {1, config.hidden_size});
  prepare_device_tensor(scratch.post_normed, {1, config.hidden_size});
  prepare_device_tensor(scratch.gate_up, {1, config.intermediate_size * 2});
  prepare_device_tensor(scratch.activated, {1, config.intermediate_size});
  prepare_device_tensor(scratch.mlp, {1, config.hidden_size});

  device_rms_norm_rows(hidden, weights.input_norm, config.rms_norm_eps, 1, config.hidden_size, scratch.normed);
  device_matmul(weights.qkv_proj, scratch.normed, 1, scratch.qkv);

  cache.tokens += 1;
  device_decode_qkv_postprocess(scratch.qkv, weights.q_norm, weights.k_norm, cache.custom_attention_cache, cache.tokens,
                                config.num_attention_heads, config.num_key_value_heads, config.head_dim, position,
                                config.max_position_embeddings, config.rope_theta, scratch.query);
  device_attention_from_cache(scratch.query, cache.custom_attention_cache, cache.tokens, config.num_attention_heads,
                              config.num_key_value_heads, config.head_dim, scratch.attention_context);
  device_matmul(weights.o_proj, scratch.attention_context, 1, scratch.attention_out);
  device_add(hidden, scratch.attention_out, scratch.residual);

  device_rms_norm_rows(scratch.residual, weights.post_attention_norm, config.rms_norm_eps, 1, config.hidden_size,
                       scratch.post_normed);
  device_matmul(weights.gate_up_proj, scratch.post_normed, 1, scratch.gate_up);
  device_swiglu_rows(scratch.gate_up, 1, config.intermediate_size, scratch.activated);
  device_matmul(weights.down_proj, scratch.activated, 1, scratch.mlp);
  device_add(scratch.residual, scratch.mlp, out);
}

void run_dense_prefill_layer_device(const MiniMindConfig& config,
                                    const AscendLayerWeights& weights,
                                    LayerKvCache& cache,
                                    const DeviceTensor& hidden_seq,
                                    int64_t start_position,
                                    int64_t seq_len,
                                    DeviceTensor& out) {
  require_device_dense(config);
  if (cache.tokens != 0) {
    throw std::invalid_argument("prefill layer requires an empty KV cache");
  }
  const int64_t q_size = config.num_attention_heads * config.head_dim;
  const int64_t kv_size = config.num_key_value_heads * config.head_dim;
  auto& scratch = device_layer_scratch();
  PrefillProfiler profiler(cache.profile_layer++, seq_len);

  prepare_device_tensor(scratch.normed, {seq_len, config.hidden_size});
  prepare_device_tensor(scratch.qkv, {seq_len, q_size + kv_size * 2});
  prepare_device_tensor(scratch.query, {seq_len, q_size});
  prepare_device_tensor(scratch.key, {seq_len, kv_size});
  prepare_device_tensor(scratch.value, {seq_len, kv_size});
  prepare_device_tensor(scratch.attention_context, {seq_len, q_size});
  prepare_device_tensor(scratch.attention_out, {seq_len, config.hidden_size});
  prepare_device_tensor(scratch.residual, {seq_len, config.hidden_size});
  prepare_device_tensor(scratch.post_normed, {seq_len, config.hidden_size});
  prepare_device_tensor(scratch.gate_up, {seq_len, config.intermediate_size * 2});
  prepare_device_tensor(scratch.activated, {seq_len, config.intermediate_size});
  prepare_device_tensor(scratch.mlp, {seq_len, config.hidden_size});

  device_rms_norm_rows(hidden_seq, weights.input_norm, config.rms_norm_eps, seq_len, config.hidden_size, scratch.normed);
  profiler.mark("input_norm");
  device_matmul(weights.qkv_proj, scratch.normed, seq_len, scratch.qkv);
  profiler.mark("qkv_matmul");
  device_qkv_split_rows(scratch.qkv, seq_len, q_size, kv_size, scratch.query, scratch.key, scratch.value);
  profiler.mark("qkv_split");

  device_rms_norm_rows(scratch.query, weights.q_norm, config.rms_norm_eps,
                       seq_len * config.num_attention_heads, config.head_dim, scratch.query);
  profiler.mark("q_norm_rows");
  device_rms_norm_rows(scratch.key, weights.k_norm, config.rms_norm_eps,
                       seq_len * config.num_key_value_heads, config.head_dim, scratch.key);
  profiler.mark("k_norm_rows");
  device_rope_rows(scratch.query, seq_len, config.num_attention_heads, config.head_dim, start_position,
                   config.max_position_embeddings, config.rope_theta);
  profiler.mark("rope_query");
  device_rope_rows(scratch.key, seq_len, config.num_key_value_heads, config.head_dim, start_position,
                   config.max_position_embeddings, config.rope_theta);
  profiler.mark("rope_key");

  device_prefill_attention(scratch.query, scratch.key, scratch.value, cache.custom_attention_cache, seq_len,
                           config.num_attention_heads, config.num_key_value_heads, config.head_dim,
                           scratch.attention_context);
  profiler.mark("prefill_attention");
  device_matmul(weights.o_proj, scratch.attention_context, seq_len, scratch.attention_out);
  profiler.mark("o_proj");
  device_add(hidden_seq, scratch.attention_out, scratch.residual);
  profiler.mark("residual_add1");

  device_rms_norm_rows(scratch.residual, weights.post_attention_norm, config.rms_norm_eps, seq_len, config.hidden_size,
                       scratch.post_normed);
  profiler.mark("post_norm");
  device_matmul(weights.gate_up_proj, scratch.post_normed, seq_len, scratch.gate_up);
  profiler.mark("gate_up");
  device_swiglu_rows(scratch.gate_up, seq_len, config.intermediate_size, scratch.activated);
  profiler.mark("swiglu");
  device_matmul(weights.down_proj, scratch.activated, seq_len, scratch.mlp);
  profiler.mark("down_proj");
  device_add(scratch.residual, scratch.mlp, out);
  profiler.mark("residual_add2");

  cache.keys.clear();
  cache.values.clear();
  cache.tokens = seq_len;
}

#endif

}  // namespace minimind::model

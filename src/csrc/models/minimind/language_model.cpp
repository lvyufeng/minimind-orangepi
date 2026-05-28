#include "language_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

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

std::vector<float> matvec(const std::vector<float>& matrix,
                          int64_t rows,
                          int64_t cols,
                          const std::vector<float>& input) {
  require_size(matrix, rows * cols, "matrix");
  require_size(input, cols, "matvec input");
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

std::vector<float> embedding_row(const std::vector<float>& embeddings,
                                 int64_t vocab_size,
                                 int64_t hidden_size,
                                 int32_t token) {
  if (token < 0 || token >= vocab_size) {
    throw std::out_of_range("token id out of vocabulary range");
  }
  require_size(embeddings, vocab_size * hidden_size, "embed_tokens");
  const auto begin = embeddings.begin() + static_cast<int64_t>(token) * hidden_size;
  return std::vector<float>(begin, begin + hidden_size);
}

std::vector<float> identity_matrix(int64_t rows, int64_t cols) {
  std::vector<float> matrix(static_cast<std::size_t>(rows * cols), 0.0F);
  for (int64_t i = 0; i < std::min(rows, cols); ++i) {
    matrix[static_cast<std::size_t>(i * cols + i)] = 1.0F;
  }
  return matrix;
}

std::vector<float> filled(int64_t size, float value) {
  return std::vector<float>(static_cast<std::size_t>(size), value);
}

}  // namespace

LanguageModel::LanguageModel(MiniMindConfig config, DenseModelWeights weights)
    : config_(std::move(config)), weights_(std::move(weights)) {
  const auto errors = validate_config(config_);
  if (!errors.empty()) {
    throw std::invalid_argument("invalid MiniMind config: " + errors.front());
  }
  require_size(weights_.embed_tokens, config_.vocab_size * config_.hidden_size, "embed_tokens");
  require_size(weights_.final_norm, config_.hidden_size, "final_norm");
  require_size(weights_.lm_head, config_.vocab_size * config_.hidden_size, "lm_head");
  if (static_cast<int64_t>(weights_.layers.size()) != config_.num_hidden_layers) {
    throw std::invalid_argument("layer weight count does not match config");
  }
}

DecoderState LanguageModel::make_state() const {
  DecoderState state;
  state.layers.resize(static_cast<std::size_t>(config_.num_hidden_layers));
  return state;
}

std::vector<float> LanguageModel::forward_token(int32_t token, DecoderState& state) const {
  auto hidden = embedding_row(weights_.embed_tokens, config_.vocab_size, config_.hidden_size, token);
  for (int64_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
    hidden = run_dense_decoder_layer(config_, weights_.layers[static_cast<std::size_t>(layer)],
                                     state.layers[static_cast<std::size_t>(layer)], hidden,
                                     state.position);
  }
  state.position += 1;

  hidden = rms_norm(hidden, weights_.final_norm, config_.rms_norm_eps);
  return matvec(weights_.lm_head, config_.vocab_size, config_.hidden_size, hidden);
}

std::vector<int32_t> LanguageModel::generate(const std::vector<int32_t>& prompt_tokens,
                                             int64_t max_new_tokens) const {
  if (prompt_tokens.empty()) {
    throw std::invalid_argument("prompt_tokens cannot be empty");
  }
  if (max_new_tokens < 0) {
    throw std::invalid_argument("max_new_tokens cannot be negative");
  }

  DecoderState state = make_state();
  std::vector<float> logits;
  for (int32_t token : prompt_tokens) {
    logits = forward_token(token, state);
  }

  std::vector<int32_t> generated;
  generated.reserve(static_cast<std::size_t>(max_new_tokens));
  for (int64_t i = 0; i < max_new_tokens; ++i) {
    const int32_t next = argmax_token(logits);
    generated.push_back(next);
    if (next == config_.eos_token_id) {
      break;
    }
    logits = forward_token(next, state);
  }
  return generated;
}

int32_t argmax_token(const std::vector<float>& logits) {
  if (logits.empty()) {
    throw std::invalid_argument("logits cannot be empty");
  }
  return static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

LanguageModel make_toy_language_model() {
  MiniMindConfig config;
  config.hidden_size = 4;
  config.num_hidden_layers = 1;
  config.use_moe = false;
  config.vocab_size = 8;
  config.bos_token_id = 1;
  config.eos_token_id = 2;
  config.num_attention_heads = 2;
  config.num_key_value_heads = 1;
  config.head_dim = 2;
  config.intermediate_size = 8;
  config.max_position_embeddings = 16;
  config.rope_theta = 10000.0F;
  config.tie_word_embeddings = false;
  config.moe_intermediate_size = 8;

  DenseModelWeights weights;
  weights.embed_tokens = {
      0.0F, 0.0F, 0.0F, 0.0F,
      1.0F, 0.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F, 0.0F,
      0.0F, 0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 0.0F, 1.0F,
      1.0F, 1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.0F, 1.0F,
  };
  weights.final_norm = filled(config.hidden_size, 1.0F);
  weights.lm_head = {
      0.0F, 0.0F, 0.0F, 0.0F,
      0.2F, 0.0F, 0.0F, 0.0F,
      0.0F, 0.2F, 0.0F, 0.0F,
      0.0F, 0.0F, 0.2F, 0.0F,
      0.0F, 0.0F, 0.0F, 0.2F,
      2.0F, 0.1F, 0.0F, 0.0F,
      0.0F, 2.0F, 0.1F, 0.0F,
      0.0F, 0.0F, 2.0F, 0.1F,
  };

  DenseLayerWeights layer;
  layer.input_norm = filled(config.hidden_size, 1.0F);
  layer.post_attention_norm = filled(config.hidden_size, 1.0F);
  layer.q_norm = filled(config.head_dim, 1.0F);
  layer.k_norm = filled(config.head_dim, 1.0F);
  layer.q_proj = identity_matrix(config.num_attention_heads * config.head_dim, config.hidden_size);
  layer.k_proj = identity_matrix(config.num_key_value_heads * config.head_dim, config.hidden_size);
  layer.v_proj = identity_matrix(config.num_key_value_heads * config.head_dim, config.hidden_size);
  layer.o_proj = identity_matrix(config.hidden_size, config.num_attention_heads * config.head_dim);
  layer.gate_proj = identity_matrix(config.intermediate_size, config.hidden_size);
  layer.up_proj = identity_matrix(config.intermediate_size, config.hidden_size);
  layer.down_proj = identity_matrix(config.hidden_size, config.intermediate_size);
  weights.layers.push_back(std::move(layer));

  return LanguageModel(config, std::move(weights));
}

LanguageModel make_toy_moe_language_model() {
  MiniMindConfig config;
  config.hidden_size = 4;
  config.num_hidden_layers = 1;
  config.use_moe = true;
  config.vocab_size = 8;
  config.bos_token_id = 1;
  config.eos_token_id = 2;
  config.num_attention_heads = 2;
  config.num_key_value_heads = 1;
  config.head_dim = 2;
  config.intermediate_size = 8;
  config.max_position_embeddings = 16;
  config.rope_theta = 10000.0F;
  config.tie_word_embeddings = false;
  config.num_experts = 2;
  config.num_experts_per_tok = 1;
  config.moe_intermediate_size = 8;

  DenseModelWeights weights;
  weights.embed_tokens = {
      0.0F, 0.0F, 0.0F, 0.0F,
      1.0F, 0.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F, 0.0F,
      0.0F, 0.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 0.0F, 1.0F,
      1.0F, 1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 1.0F, 0.0F,
      0.0F, 0.0F, 1.0F, 1.0F,
  };
  weights.final_norm = filled(config.hidden_size, 1.0F);
  weights.lm_head = {
      0.0F, 0.0F, 0.0F, 0.0F,
      0.2F, 0.0F, 0.0F, 0.0F,
      0.0F, 0.2F, 0.0F, 0.0F,
      0.0F, 0.0F, 0.2F, 0.0F,
      0.0F, 0.0F, 0.0F, 0.2F,
      2.0F, 0.1F, 0.0F, 0.0F,
      0.0F, 2.0F, 0.1F, 0.0F,
      0.0F, 0.0F, 2.0F, 0.1F,
  };

  DenseLayerWeights layer;
  layer.input_norm = filled(config.hidden_size, 1.0F);
  layer.post_attention_norm = filled(config.hidden_size, 1.0F);
  layer.q_norm = filled(config.head_dim, 1.0F);
  layer.k_norm = filled(config.head_dim, 1.0F);
  layer.q_proj = identity_matrix(config.num_attention_heads * config.head_dim, config.hidden_size);
  layer.k_proj = identity_matrix(config.num_key_value_heads * config.head_dim, config.hidden_size);
  layer.v_proj = identity_matrix(config.num_key_value_heads * config.head_dim, config.hidden_size);
  layer.o_proj = identity_matrix(config.hidden_size, config.num_attention_heads * config.head_dim);
  layer.moe_gate = {
      1.0F, 0.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F, 0.0F,
  };
  for (int expert = 0; expert < config.num_experts; ++expert) {
    ExpertWeights expert_weights;
    expert_weights.gate_proj = identity_matrix(config.moe_intermediate_size, config.hidden_size);
    expert_weights.up_proj = identity_matrix(config.moe_intermediate_size, config.hidden_size);
    expert_weights.down_proj = identity_matrix(config.hidden_size, config.moe_intermediate_size);
    layer.experts.push_back(std::move(expert_weights));
  }
  weights.layers.push_back(std::move(layer));

  return LanguageModel(config, std::move(weights));
}

}  // namespace minimind::model

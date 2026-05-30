#include "language_model.h"

#if defined(MINIMIND_USE_ASCEND)
#include "ascend_weights.h"
#endif

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

void append_matrix(std::vector<float>& output, const std::vector<float>& matrix) {
  output.insert(output.end(), matrix.begin(), matrix.end());
}

void initialize_fused_layer_weights(const MiniMindConfig& config, DenseLayerWeights& layer) {
  const int64_t hidden_size = config.hidden_size;
  const int64_t q_size = config.num_attention_heads * config.head_dim;
  const int64_t kv_size = config.num_key_value_heads * config.head_dim;
  if (layer.qkv_proj.empty() && !layer.q_proj.empty() && !layer.k_proj.empty() && !layer.v_proj.empty()) {
    layer.qkv_proj.reserve(static_cast<std::size_t>((q_size + 2 * kv_size) * hidden_size));
    append_matrix(layer.qkv_proj, layer.q_proj);
    append_matrix(layer.qkv_proj, layer.k_proj);
    append_matrix(layer.qkv_proj, layer.v_proj);
  }
  if (!config.use_moe && layer.gate_up_proj.empty() && !layer.gate_proj.empty() && !layer.up_proj.empty()) {
    layer.gate_up_proj.reserve(static_cast<std::size_t>(2 * config.intermediate_size * hidden_size));
    append_matrix(layer.gate_up_proj, layer.gate_proj);
    append_matrix(layer.gate_up_proj, layer.up_proj);
  }
  if (config.use_moe) {
    for (auto& expert : layer.experts) {
      if (expert.gate_up_proj.empty() && !expert.gate_proj.empty() && !expert.up_proj.empty()) {
        expert.gate_up_proj.reserve(static_cast<std::size_t>(2 * config.moe_intermediate_size * hidden_size));
        append_matrix(expert.gate_up_proj, expert.gate_proj);
        append_matrix(expert.gate_up_proj, expert.up_proj);
      }
    }
  }
}

#if defined(MINIMIND_USE_ASCEND)
int64_t device_shape_elements(const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    count *= dim;
  }
  return shape.empty() ? 0 : count;
}

void prepare_device_tensor(DeviceTensor& tensor,
                           const std::vector<int64_t>& shape,
                           aclDataType dtype = ACL_FLOAT16) {
  tensor.shape = shape;
  tensor.dtype = dtype;
  tensor.strides.resize(shape.size());
  int64_t stride = 1;
  for (std::size_t i = shape.size(); i-- > 0;) {
    tensor.strides[i] = stride;
    stride *= shape[i];
  }
  const std::size_t element_size = (dtype == ACL_INT32) ? sizeof(int32_t) : sizeof(uint16_t);
  tensor.buffer.reserve(static_cast<std::size_t>(device_shape_elements(shape)) * element_size);
}

struct DeviceDecodeScratch {
  DeviceTensor token_ids;
  DeviceTensor hidden;
  DeviceTensor next_hidden;
  DeviceTensor normed;
  DeviceTensor logits;
  int32_t token_host = 0;
};

DeviceDecodeScratch& device_decode_scratch() {
  static DeviceDecodeScratch* scratch = new DeviceDecodeScratch();
  return *scratch;
}

void upload_decode_token_async(int32_t token, DeviceTensor& token_ids, int32_t& token_host) {
  if (token_ids.elements() != 1 || token_ids.dtype != ACL_INT32) {
    throw std::invalid_argument("decode token scratch shape mismatch");
  }
  token_host = token;
  check_acl(aclrtMemcpyAsync(token_ids.data(), token_ids.bytes(), &token_host, sizeof(token_host),
                             ACL_MEMCPY_HOST_TO_DEVICE, runtime().stream()),
            "aclrtMemcpyAsync decode token H2D failed");
}

AscendLayerWeights upload_ascend_layer_weights(const MiniMindConfig& config, const DenseLayerWeights& layer) {
  const int64_t q_size = config.num_attention_heads * config.head_dim;
  const int64_t kv_size = config.num_key_value_heads * config.head_dim;
  AscendLayerWeights uploaded;
  uploaded.input_norm = upload_tensor_fp16(layer.input_norm, {config.hidden_size});
  uploaded.post_attention_norm = upload_tensor_fp16(layer.post_attention_norm, {config.hidden_size});
  uploaded.q_norm = upload_tensor_fp16(layer.q_norm, {config.head_dim});
  uploaded.k_norm = upload_tensor_fp16(layer.k_norm, {config.head_dim});
  uploaded.qkv_proj = upload_matrix_transposed_fp16(layer.qkv_proj, q_size + 2 * kv_size, config.hidden_size);
  uploaded.o_proj = upload_matrix_transposed_fp16(layer.o_proj, config.hidden_size, q_size);
  if (config.use_moe) {
    uploaded.moe_gate = upload_matrix_transposed_fp16(layer.moe_gate, config.num_experts, config.hidden_size);
    uploaded.experts.reserve(layer.experts.size());
    for (const auto& expert : layer.experts) {
      AscendExpertWeights uploaded_expert;
      uploaded_expert.gate_up_proj = upload_matrix_transposed_fp16(expert.gate_up_proj,
                                                                   2 * config.moe_intermediate_size,
                                                                   config.hidden_size);
      uploaded_expert.down_proj = upload_matrix_transposed_fp16(expert.down_proj,
                                                                config.hidden_size,
                                                                config.moe_intermediate_size);
      uploaded.experts.push_back(std::move(uploaded_expert));
    }
  } else {
    uploaded.gate_up_proj = upload_matrix_transposed_fp16(layer.gate_up_proj,
                                                         2 * config.intermediate_size,
                                                         config.hidden_size);
    uploaded.down_proj = upload_matrix_transposed_fp16(layer.down_proj, config.hidden_size, config.intermediate_size);
  }
  return uploaded;
}
#endif

#if defined(MINIMIND_USE_ASCEND)
int32_t forward_next_token_device(const MiniMindConfig& config,
                                  const AscendModelWeights& weights,
                                  int32_t token,
                                  DecoderState& state) {
  auto& scratch = device_decode_scratch();
  prepare_device_tensor(scratch.token_ids, {1}, ACL_INT32);
  prepare_device_tensor(scratch.hidden, {1, config.hidden_size});
  prepare_device_tensor(scratch.next_hidden, {1, config.hidden_size});
  prepare_device_tensor(scratch.normed, {1, config.hidden_size});
  prepare_device_tensor(scratch.logits, {1, config.vocab_size});

  upload_decode_token_async(token, scratch.token_ids, scratch.token_host);
  device_embedding_gather(weights.embed_tokens, scratch.token_ids, 1, config.vocab_size, config.hidden_size,
                          scratch.hidden);
  for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    run_dense_decoder_layer_device(config, weights.layers[static_cast<std::size_t>(layer)],
                                   state.layers[static_cast<std::size_t>(layer)], scratch.hidden, state.position,
                                   scratch.next_hidden);
    std::swap(scratch.hidden, scratch.next_hidden);
  }
  state.position += 1;

  device_rms_norm_rows(scratch.hidden, weights.final_norm, config.rms_norm_eps, 1, config.hidden_size, scratch.normed);
  return device_matmul_argmax(weights.lm_head, scratch.normed, scratch.logits);
}

int32_t prefill_next_token_device(const MiniMindConfig& config,
                                  const AscendModelWeights& weights,
                                  const std::vector<int32_t>& prompt_tokens,
                                  DecoderState& state) {
  const int64_t seq_len = static_cast<int64_t>(prompt_tokens.size());
  DeviceTensor token_ids = upload_tensor_int32(prompt_tokens, {seq_len});
  DeviceTensor hidden = make_device_tensor({seq_len, config.hidden_size});
  DeviceTensor next_hidden = make_device_tensor({seq_len, config.hidden_size});
  device_embedding_gather(weights.embed_tokens, token_ids, seq_len, config.vocab_size, config.hidden_size, hidden);
  for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    run_dense_prefill_layer_device(config, weights.layers[static_cast<std::size_t>(layer)],
                                   state.layers[static_cast<std::size_t>(layer)], hidden, state.position, seq_len,
                                   next_hidden);
    std::swap(hidden, next_hidden);
  }
  state.position += seq_len;

  DeviceTensor last_hidden = make_device_tensor({1, config.hidden_size});
  DeviceTensor normed = make_device_tensor({1, config.hidden_size});
  device_copy_last_row(hidden, seq_len, config.hidden_size, last_hidden);
  device_rms_norm_rows(last_hidden, weights.final_norm, config.rms_norm_eps, 1, config.hidden_size, normed);
  return device_matmul_argmax(weights.lm_head, normed);
}
#endif

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
  for (auto& layer : weights_.layers) {
    initialize_fused_layer_weights(config_, layer);
  }
}

#if defined(MINIMIND_USE_ASCEND)
const AscendModelWeights& LanguageModel::ascend_weights() const {
  if (!weights_.ascend) {
    (void)runtime();
    auto uploaded = std::make_shared<AscendModelWeights>();
    uploaded->embed_tokens = upload_tensor_fp16(weights_.embed_tokens, {config_.vocab_size, config_.hidden_size});
    uploaded->final_norm = upload_tensor_fp16(weights_.final_norm, {config_.hidden_size});
    uploaded->lm_head = upload_matrix_transposed_fp16(weights_.lm_head, config_.vocab_size, config_.hidden_size);
    uploaded->layers.reserve(weights_.layers.size());
    for (const auto& layer : weights_.layers) {
      uploaded->layers.push_back(upload_ascend_layer_weights(config_, layer));
    }
    weights_.ascend = std::move(uploaded);
  }
  return *weights_.ascend;
}
#endif

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

int32_t LanguageModel::forward_next_token(int32_t token, DecoderState& state) const {
#if defined(MINIMIND_USE_ASCEND)
  if (cube_matvec_available() && custom_ops_available() && config_.hidden_size >= 128 && config_.vocab_size >= 128 &&
      config_.head_dim <= 128 && !config_.use_moe) {
    return forward_next_token_device(config_, ascend_weights(), token, state);
  }
#endif
  auto hidden = embedding_row(weights_.embed_tokens, config_.vocab_size, config_.hidden_size, token);
  for (int64_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
    hidden = run_dense_decoder_layer(config_, weights_.layers[static_cast<std::size_t>(layer)],
                                     state.layers[static_cast<std::size_t>(layer)], hidden,
                                     state.position);
  }
  state.position += 1;

  hidden = rms_norm(hidden, weights_.final_norm, config_.rms_norm_eps);
  if (cube_matvec_available() && config_.vocab_size >= 128 && config_.hidden_size >= 128) {
    try {
      return cube_matvec_argmax(weights_.lm_head, config_.vocab_size, config_.hidden_size, hidden);
    } catch (const std::exception&) {
    }
  }
  return argmax_token(matvec(weights_.lm_head, config_.vocab_size, config_.hidden_size, hidden));
}

int32_t LanguageModel::prefill_next_token(const std::vector<int32_t>& prompt_tokens, DecoderState& state) const {
  if (prompt_tokens.empty()) {
    throw std::invalid_argument("prompt_tokens cannot be empty");
  }
  if (!cube_matmul_available() || !custom_ops_available()) {
    throw std::runtime_error("prefill requires Ascend Cube and custom ops");
  }
#if defined(MINIMIND_USE_ASCEND)
  if (config_.hidden_size >= 128 && config_.vocab_size >= 128 && config_.head_dim <= 128 &&
      static_cast<int64_t>(prompt_tokens.size()) <= 2048 && !config_.use_moe) {
    return prefill_next_token_device(config_, ascend_weights(), prompt_tokens, state);
  }
#endif
  const int64_t seq_len = static_cast<int64_t>(prompt_tokens.size());
  std::vector<float> hidden_seq(static_cast<std::size_t>(seq_len * config_.hidden_size));
  for (int64_t token = 0; token < seq_len; ++token) {
    auto row = embedding_row(weights_.embed_tokens, config_.vocab_size, config_.hidden_size, prompt_tokens[static_cast<std::size_t>(token)]);
    std::copy(row.begin(), row.end(), hidden_seq.begin() + token * config_.hidden_size);
  }

  for (int64_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
    hidden_seq = run_dense_prefill_layer(config_, weights_.layers[static_cast<std::size_t>(layer)],
                                         state.layers[static_cast<std::size_t>(layer)], hidden_seq, state.position,
                                         seq_len);
  }
  state.position += seq_len;

  std::vector<float> last_hidden(hidden_seq.end() - config_.hidden_size, hidden_seq.end());
  last_hidden = rms_norm(last_hidden, weights_.final_norm, config_.rms_norm_eps);
  const auto logits = cube_matmul(weights_.lm_head, config_.vocab_size, config_.hidden_size, last_hidden, 1);
  return argmax_token(logits);
}

void LanguageModel::generate_stream(const std::vector<int32_t>& prompt_tokens,
                                    int64_t max_new_tokens,
                                    const std::function<void(int32_t)>& on_token) const {
  if (prompt_tokens.empty()) {
    throw std::invalid_argument("prompt_tokens cannot be empty");
  }
  if (max_new_tokens < 0) {
    throw std::invalid_argument("max_new_tokens cannot be negative");
  }

  DecoderState state = make_state();
  int32_t next = 0;
  if (cube_matmul_available() && custom_ops_available() && config_.head_dim <= 128 &&
      static_cast<int64_t>(prompt_tokens.size()) <= 2048) {
    next = prefill_next_token(prompt_tokens, state);
  } else {
    for (int32_t token : prompt_tokens) {
      next = forward_next_token(token, state);
    }
  }

  for (int64_t i = 0; i < max_new_tokens; ++i) {
    on_token(next);
    if (next == config_.eos_token_id || i + 1 == max_new_tokens) {
      break;
    }
    next = forward_next_token(next, state);
  }
}

std::vector<int32_t> LanguageModel::generate(const std::vector<int32_t>& prompt_tokens,
                                             int64_t max_new_tokens) const {
  std::vector<int32_t> generated;
  generated.reserve(static_cast<std::size_t>(std::max<int64_t>(max_new_tokens, 0)));
  generate_stream(prompt_tokens, max_new_tokens, [&generated](int32_t token) {
    generated.push_back(token);
  });
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

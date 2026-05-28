#pragma once

#include "minimind_orangepi/weights.h"

#include <cstdint>
#include <string>
#include <vector>

namespace minimind::model {

struct RopeScalingConfig {
  bool enabled = false;
  float factor = 16.0F;
  float beta_fast = 32.0F;
  float beta_slow = 1.0F;
  int64_t original_max_position_embeddings = 2048;
  float attention_factor = 1.0F;
  std::string type = "yarn";
};

struct MiniMindConfig {
  int64_t hidden_size = 768;
  int64_t num_hidden_layers = 8;
  bool use_moe = false;
  float dropout = 0.0F;
  int64_t vocab_size = 6400;
  int32_t bos_token_id = 1;
  int32_t eos_token_id = 2;
  bool flash_attn = true;
  int64_t num_attention_heads = 8;
  int64_t num_key_value_heads = 4;
  int64_t head_dim = 96;
  std::string hidden_act = "silu";
  int64_t intermediate_size = 2048;
  int64_t max_position_embeddings = 32768;
  float rms_norm_eps = 1e-6F;
  float rope_theta = 1000000.0F;
  bool tie_word_embeddings = true;
  RopeScalingConfig rope_scaling;
  int64_t num_experts = 4;
  int64_t num_experts_per_tok = 1;
  int64_t moe_intermediate_size = 2048;
  bool norm_topk_prob = true;
  float router_aux_loss_coef = 5e-4F;
};

MiniMindConfig default_minimind_config();
std::vector<std::string> validate_config(const MiniMindConfig& config);
std::vector<TensorSpec> expected_weight_specs(const MiniMindConfig& config);

}  // namespace minimind::model

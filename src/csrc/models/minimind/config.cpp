#include "config.h"

#include <cmath>

namespace minimind::model {

MiniMindConfig default_minimind_config() {
  MiniMindConfig config;
  config.head_dim = config.hidden_size / config.num_attention_heads;
  config.intermediate_size = static_cast<int64_t>(std::ceil(config.hidden_size * 3.14159265358979323846 / 64.0)) * 64;
  config.moe_intermediate_size = config.intermediate_size;
  return config;
}

std::vector<std::string> validate_config(const MiniMindConfig& config) {
  std::vector<std::string> errors;

  auto require_positive = [&](int64_t value, const char* name) {
    if (value <= 0) {
      errors.push_back(std::string(name) + " must be positive");
    }
  };

  require_positive(config.hidden_size, "hidden_size");
  require_positive(config.num_hidden_layers, "num_hidden_layers");
  require_positive(config.vocab_size, "vocab_size");
  require_positive(config.num_attention_heads, "num_attention_heads");
  require_positive(config.num_key_value_heads, "num_key_value_heads");
  require_positive(config.head_dim, "head_dim");
  require_positive(config.intermediate_size, "intermediate_size");
  require_positive(config.max_position_embeddings, "max_position_embeddings");

  if (config.hidden_size > 0 && config.num_attention_heads > 0 &&
      config.hidden_size % config.num_attention_heads != 0) {
    errors.push_back("hidden_size must be divisible by num_attention_heads");
  }
  if (config.hidden_size > 0 && config.num_attention_heads > 0 &&
      config.head_dim != config.hidden_size / config.num_attention_heads) {
    errors.push_back("head_dim must equal hidden_size / num_attention_heads");
  }
  if (config.num_attention_heads > 0 && config.num_key_value_heads > 0 &&
      config.num_attention_heads % config.num_key_value_heads != 0) {
    errors.push_back("num_attention_heads must be divisible by num_key_value_heads");
  }
  if (config.rope_theta <= 0.0F) {
    errors.push_back("rope_theta must be positive");
  }
  if (config.rms_norm_eps <= 0.0F) {
    errors.push_back("rms_norm_eps must be positive");
  }
  if (config.hidden_act != "silu") {
    errors.push_back("only silu hidden_act is currently supported");
  }
  if (config.use_moe) {
    require_positive(config.num_experts, "num_experts");
    require_positive(config.num_experts_per_tok, "num_experts_per_tok");
    require_positive(config.moe_intermediate_size, "moe_intermediate_size");
    if (config.num_experts_per_tok != 1) {
      errors.push_back("only top-1 MoE routing is currently planned");
    }
    if (config.num_experts_per_tok > config.num_experts) {
      errors.push_back("num_experts_per_tok cannot exceed num_experts");
    }
  }
  if (config.rope_scaling.enabled) {
    if (config.rope_scaling.type != "yarn") {
      errors.push_back("only yarn rope scaling is supported");
    }
    if (config.rope_scaling.factor <= 1.0F) {
      errors.push_back("rope scaling factor must be greater than 1");
    }
    if (config.rope_scaling.original_max_position_embeddings <= 0) {
      errors.push_back("rope scaling original_max_position_embeddings must be positive");
    }
  }

  return errors;
}

std::vector<TensorSpec> expected_weight_specs(const MiniMindConfig& config) {
  std::vector<TensorSpec> specs;
  specs.push_back({"model.embed_tokens.weight", {config.vocab_size, config.hidden_size}, true});
  specs.push_back({"model.norm.weight", {config.hidden_size}, true});
  specs.push_back({"lm_head.weight", {config.vocab_size, config.hidden_size}, !config.tie_word_embeddings});

  const int64_t q_out = config.num_attention_heads * config.head_dim;
  const int64_t kv_out = config.num_key_value_heads * config.head_dim;

  for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    specs.push_back({prefix + "input_layernorm.weight", {config.hidden_size}, true});
    specs.push_back({prefix + "post_attention_layernorm.weight", {config.hidden_size}, true});
    specs.push_back({prefix + "self_attn.q_proj.weight", {q_out, config.hidden_size}, true});
    specs.push_back({prefix + "self_attn.k_proj.weight", {kv_out, config.hidden_size}, true});
    specs.push_back({prefix + "self_attn.v_proj.weight", {kv_out, config.hidden_size}, true});
    specs.push_back({prefix + "self_attn.o_proj.weight", {config.hidden_size, q_out}, true});
    specs.push_back({prefix + "self_attn.q_norm.weight", {config.head_dim}, true});
    specs.push_back({prefix + "self_attn.k_norm.weight", {config.head_dim}, true});

    if (config.use_moe) {
      specs.push_back({prefix + "mlp.gate.weight", {config.num_experts, config.hidden_size}, true});
      for (int64_t expert = 0; expert < config.num_experts; ++expert) {
        const std::string expert_prefix = prefix + "mlp.experts." + std::to_string(expert) + ".";
        specs.push_back({expert_prefix + "gate_proj.weight", {config.moe_intermediate_size, config.hidden_size}, true});
        specs.push_back({expert_prefix + "up_proj.weight", {config.moe_intermediate_size, config.hidden_size}, true});
        specs.push_back({expert_prefix + "down_proj.weight", {config.hidden_size, config.moe_intermediate_size}, true});
      }
    } else {
      specs.push_back({prefix + "mlp.gate_proj.weight", {config.intermediate_size, config.hidden_size}, true});
      specs.push_back({prefix + "mlp.up_proj.weight", {config.intermediate_size, config.hidden_size}, true});
      specs.push_back({prefix + "mlp.down_proj.weight", {config.hidden_size, config.intermediate_size}, true});
    }
  }

  return specs;
}

}  // namespace minimind::model

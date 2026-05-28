#include "../../src/csrc/models/minimind/config.h"
#include "minimind_orangepi/weights.h"

#include <iostream>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "check failed: " #condition << '\n';                       \
      return 1;                                                                \
    }                                                                          \
  } while (false)

int main() {
  auto config = minimind::model::default_minimind_config();
  config.num_hidden_layers = 1;

  auto specs = minimind::model::expected_weight_specs(config);
  minimind::WeightManifest manifest;
  for (const auto& spec : specs) {
    if (spec.required) {
      manifest.add(spec.name, spec.shape);
    }
  }

  auto errors = minimind::validate_weight_manifest(manifest, specs);
  CHECK(errors.empty());
  CHECK(manifest.contains("model.layers.0.self_attn.q_proj.weight"));
  CHECK(manifest.shape("model.layers.0.self_attn.q_proj.weight") ==
        std::vector<int64_t>({config.num_attention_heads * config.head_dim, config.hidden_size}));

  minimind::WeightManifest broken;
  broken.add("model.embed_tokens.weight", {config.vocab_size, config.hidden_size + 1});
  errors = minimind::validate_weight_manifest(broken, specs);
  CHECK(!errors.empty());

  config.use_moe = true;
  config.num_hidden_layers = 1;
  specs = minimind::model::expected_weight_specs(config);
  bool has_expert = false;
  for (const auto& spec : specs) {
    if (spec.name == "model.layers.0.mlp.experts.0.gate_proj.weight") {
      has_expert = true;
      CHECK(spec.shape == std::vector<int64_t>({config.moe_intermediate_size, config.hidden_size}));
    }
  }
  CHECK(has_expert);

  return 0;
}

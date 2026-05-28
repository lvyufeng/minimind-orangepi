#include "../../src/csrc/models/minimind_v/projector.h"
#include "../../src/csrc/models/minimind_v/vision_encoder.h"

#include <iostream>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "check failed: " #condition << '\n';                       \
      return 1;                                                                \
    }                                                                          \
  } while (false)

int main() {
  minimind::model_v::MiniMindVConfig config;
  config.text.hidden_size = 8;
  config.text.num_attention_heads = 2;
  config.text.num_key_value_heads = 1;
  config.text.head_dim = 4;
  config.text.intermediate_size = 16;
  config.text.moe_intermediate_size = 16;
  config.image_hidden_size = 8;

  auto errors = minimind::model_v::validate_config(config);
  CHECK(errors.empty());

  auto features = minimind::model_v::make_synthetic_image_features(config);
  CHECK(static_cast<int64_t>(features.size()) == config.image_token_len * config.image_hidden_size);

  auto weights = minimind::model_v::make_identity_projector_weights(config);
  auto projected = minimind::model_v::run_projector(config, weights, features);
  CHECK(static_cast<int64_t>(projected.size()) == config.image_token_len * config.text.hidden_size);

  bool any_nonzero = false;
  for (float value : projected) {
    any_nonzero = any_nonzero || value != 0.0F;
  }
  CHECK(any_nonzero);

  features.pop_back();
  bool threw = false;
  try {
    (void)minimind::model_v::run_projector(config, weights, features);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);

  return 0;
}

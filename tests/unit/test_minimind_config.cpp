#include "../../src/csrc/models/minimind/config.h"

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
  CHECK(config.hidden_size == 768);
  CHECK(config.num_hidden_layers == 8);
  CHECK(config.vocab_size == 6400);
  CHECK(config.num_attention_heads == 8);
  CHECK(config.num_key_value_heads == 4);
  CHECK(config.head_dim == 96);
  CHECK(config.intermediate_size == 2432);
  CHECK(config.max_position_embeddings == 32768);

  auto errors = minimind::model::validate_config(config);
  CHECK(errors.empty());

  config.num_key_value_heads = 3;
  errors = minimind::model::validate_config(config);
  CHECK(!errors.empty());

  config = minimind::model::default_minimind_config();
  config.use_moe = true;
  config.num_experts_per_tok = 2;
  errors = minimind::model::validate_config(config);
  CHECK(!errors.empty());

  config.num_experts_per_tok = 1;
  errors = minimind::model::validate_config(config);
  CHECK(errors.empty());

  return 0;
}

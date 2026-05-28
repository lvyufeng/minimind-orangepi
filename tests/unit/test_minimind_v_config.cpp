#include "../../src/csrc/models/minimind_v/config.h"

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
  auto errors = minimind::model_v::validate_config(config);
  CHECK(errors.empty());
  CHECK(config.image_special_token == "<|image_pad|>");
  CHECK(config.image_token_id == 12);
  CHECK(config.image_token_len == 64);
  CHECK(config.image_size == 256);

  config.image_token_len = 63;
  errors = minimind::model_v::validate_config(config);
  CHECK(!errors.empty());

  config.image_token_len = 64;
  config.text.num_key_value_heads = 3;
  errors = minimind::model_v::validate_config(config);
  CHECK(!errors.empty());

  return 0;
}

#include "../../src/csrc/models/minimind/rope.h"

#include <cmath>
#include <iostream>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "check failed: " #condition << '\n';                       \
      return 1;                                                                \
    }                                                                          \
  } while (false)

namespace {

bool close(float lhs, float rhs) {
  return std::fabs(lhs - rhs) < 1e-5F;
}

}  // namespace

int main() {
  auto config = minimind::model::default_minimind_config();
  config.hidden_size = 8;
  config.num_attention_heads = 2;
  config.num_key_value_heads = 1;
  config.head_dim = 4;
  config.intermediate_size = 16;
  config.moe_intermediate_size = 16;
  config.max_position_embeddings = 16;
  config.rope_theta = 10000.0F;

  auto tables = minimind::model::make_rope_tables(config, 3);
  CHECK(tables.cos.shape().dims() == std::vector<int64_t>({3, 4}));
  CHECK(tables.sin.shape().dims() == std::vector<int64_t>({3, 4}));

  const auto* cos = tables.cos.data_as<float>();
  const auto* sin = tables.sin.data_as<float>();
  CHECK(close(cos[0], 1.0F));
  CHECK(close(sin[0], 0.0F));
  CHECK(close(cos[4], std::cos(1.0F)));
  CHECK(close(sin[4], std::sin(1.0F)));
  CHECK(close(cos[5], std::cos(0.01F)));
  CHECK(close(sin[5], std::sin(0.01F)));

  config.rope_scaling.enabled = true;
  auto scaled = minimind::model::make_rope_tables(config, 3);
  CHECK(scaled.cos.shape().dims() == std::vector<int64_t>({3, 4}));

  return 0;
}

#include "../../src/csrc/models/minimind_v/vlm_model.h"

#include <iostream>
#include <stdexcept>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "check failed: " #condition << '\n';                       \
      return 1;                                                                \
    }                                                                          \
  } while (false)

int main() {
  minimind::model_v::MiniMindVConfig config;
  config.text.hidden_size = 4;
  config.text.num_attention_heads = 2;
  config.text.num_key_value_heads = 1;
  config.text.head_dim = 2;
  config.text.intermediate_size = 8;
  config.text.moe_intermediate_size = 8;

  std::vector<int32_t> input_ids;
  input_ids.push_back(1);
  for (int i = 0; i < config.image_token_len; ++i) {
    input_ids.push_back(config.image_token_id);
  }
  input_ids.push_back(2);

  const auto spans = minimind::model_v::find_image_token_spans(input_ids, config.image_token_id);
  CHECK(spans.size() == 1);
  CHECK(spans[0].start == 1);
  CHECK(spans[0].length == 64);

  std::vector<float> token_embeddings(input_ids.size() * config.text.hidden_size, -1.0F);
  std::vector<float> image_embeddings(config.image_token_len * config.text.hidden_size, 3.0F);
  const auto replaced = minimind::model_v::replace_image_token_embeddings(
      config, input_ids, token_embeddings, image_embeddings);
  CHECK(replaced[0] == -1.0F);
  CHECK(replaced[static_cast<std::size_t>(config.text.hidden_size)] == 3.0F);
  CHECK(replaced[replaced.size() - 1] == -1.0F);

  input_ids.pop_back();
  input_ids.pop_back();
  bool threw = false;
  try {
    (void)minimind::model_v::replace_image_token_embeddings(
        config, input_ids, token_embeddings, image_embeddings);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);

  return 0;
}

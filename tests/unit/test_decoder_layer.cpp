#include "../../src/csrc/models/minimind/language_model.h"

#include <iostream>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "check failed: " #condition << '\n';                       \
      return 1;                                                                \
    }                                                                          \
  } while (false)

int main() {
  auto model = minimind::model::make_toy_language_model();
  auto state = model.make_state();
  auto logits = model.forward_token(1, state);
  CHECK(logits.size() == static_cast<std::size_t>(model.config().vocab_size));
  CHECK(state.position == 1);
  CHECK(state.layers.size() == 1);
  CHECK(state.layers[0].tokens == 1);
  CHECK(static_cast<int64_t>(state.layers[0].keys.size()) ==
        model.config().num_key_value_heads * model.config().head_dim);

  auto logits2 = model.forward_token(2, state);
  CHECK(logits2.size() == logits.size());
  CHECK(state.position == 2);
  CHECK(state.layers[0].tokens == 2);

  return 0;
}

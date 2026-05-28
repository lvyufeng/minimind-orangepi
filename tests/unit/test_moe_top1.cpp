#include "../../src/csrc/models/minimind/language_model.h"

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
  auto model = minimind::model::make_toy_moe_language_model();
  CHECK(model.config().use_moe);
  CHECK(model.config().num_experts == 2);
  CHECK(model.config().num_experts_per_tok == 1);

  auto state = model.make_state();
  const auto logits = model.forward_token(1, state);
  CHECK(logits.size() == static_cast<std::size_t>(model.config().vocab_size));
  CHECK(state.layers[0].tokens == 1);

  const auto generated = model.generate({1, 2}, 3);
  CHECK(!generated.empty());
  CHECK(generated.size() <= 3);

  return 0;
}

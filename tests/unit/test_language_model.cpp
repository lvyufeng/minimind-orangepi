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
  auto model = minimind::model::make_toy_language_model();
  auto generated = model.generate({1, 2}, 4);
  CHECK(!generated.empty());
  CHECK(generated.size() <= 4);
  for (int32_t token : generated) {
    CHECK(token >= 0);
    CHECK(token < model.config().vocab_size);
  }

  bool threw = false;
  try {
    model.generate({}, 1);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);

  return 0;
}

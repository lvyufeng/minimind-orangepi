#include "minimind_orangepi/tokenizer.h"

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
  minimind::ByteTokenizer tokenizer;
  const auto tokens = tokenizer.encode("MiniMind");
  CHECK(tokens.size() == 8);
  CHECK(tokens[0] == 'M');
  CHECK(tokenizer.decode(tokens) == "MiniMind");

  bool threw = false;
  try {
    tokenizer.decode({256});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);

  return 0;
}

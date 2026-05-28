#include "minimind_orangepi/tokenizer.h"

#include <stdexcept>

namespace minimind {

std::vector<int32_t> ByteTokenizer::encode(const std::string& text) const {
  std::vector<int32_t> tokens;
  tokens.reserve(text.size());
  for (unsigned char byte : text) {
    tokens.push_back(static_cast<int32_t>(byte));
  }
  return tokens;
}

std::string ByteTokenizer::decode(const std::vector<int32_t>& tokens) const {
  std::string text;
  text.reserve(tokens.size());
  for (int32_t token : tokens) {
    if (token < 0 || token > 255) {
      throw std::invalid_argument("ByteTokenizer can only decode tokens in [0, 255]");
    }
    text.push_back(static_cast<char>(token));
  }
  return text;
}

}  // namespace minimind

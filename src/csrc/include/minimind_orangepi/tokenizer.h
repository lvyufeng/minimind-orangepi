#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace minimind {

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;
  virtual std::vector<int32_t> encode(const std::string& text) const = 0;
  virtual std::string decode(const std::vector<int32_t>& tokens) const = 0;
};

class ByteTokenizer final : public Tokenizer {
 public:
  std::vector<int32_t> encode(const std::string& text) const override;
  std::string decode(const std::vector<int32_t>& tokens) const override;
};

}  // namespace minimind

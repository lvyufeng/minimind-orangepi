#include "../models/minimind/language_model.h"
#include "minimind_orangepi/tokenizer.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string arg_value(int argc, char** argv, const std::string& name, const std::string& fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) {
      return argv[i + 1];
    }
  }
  return fallback;
}

int64_t int_arg_value(int argc, char** argv, const std::string& name, int64_t fallback) {
  const std::string value = arg_value(argc, argv, name, "");
  if (value.empty()) {
    return fallback;
  }
  return std::strtoll(value.c_str(), nullptr, 10);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string prompt = arg_value(argc, argv, "--prompt", "MiniMind");
  const int64_t max_new_tokens = int_arg_value(argc, argv, "--max-new-tokens", 8);

  minimind::ByteTokenizer tokenizer;
  auto prompt_tokens = tokenizer.encode(prompt);
  for (auto& token : prompt_tokens) {
    token = token % 8;
  }
  if (prompt_tokens.empty()) {
    prompt_tokens.push_back(1);
  }

  auto model = minimind::model::make_toy_language_model();
  const auto generated = model.generate(prompt_tokens, max_new_tokens);

  std::cout << "prompt_tokens:";
  for (int32_t token : prompt_tokens) {
    std::cout << ' ' << token;
  }
  std::cout << "\ngenerated_tokens:";
  for (int32_t token : generated) {
    std::cout << ' ' << token;
  }
  std::cout << '\n';
  return 0;
}

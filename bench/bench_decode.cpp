#include "../src/csrc/models/minimind/language_model.h"
#include "../src/csrc/models/minimind/runtime_loader.h"
#include "minimind_orangepi/tokenizer.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
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

std::vector<int32_t> parse_token_ids(const std::string& value) {
  std::vector<int32_t> tokens;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) {
      tokens.push_back(static_cast<int32_t>(std::strtol(item.c_str(), nullptr, 10)));
    }
  }
  return tokens;
}

std::vector<int32_t> prompt_tokens(const std::string& prompt,
                                   const std::string& explicit_tokens,
                                   const minimind::model::LanguageModel& model) {
  auto tokens = explicit_tokens.empty() ? std::vector<int32_t>{} : parse_token_ids(explicit_tokens);
  if (tokens.empty()) {
    minimind::ByteTokenizer tokenizer;
    tokens = tokenizer.encode(prompt);
    for (auto& token : tokens) {
      token = token % static_cast<int32_t>(model.config().vocab_size);
    }
  }
  if (tokens.empty()) {
    tokens.push_back(model.config().bos_token_id);
  }
  return tokens;
}

std::vector<int32_t> decode_fixed_steps(const minimind::model::LanguageModel& model,
                                        const std::vector<int32_t>& input_tokens,
                                        int64_t decode_tokens) {
  auto state = model.make_state();
  std::vector<float> logits;
  for (int32_t token : input_tokens) {
    logits = model.forward_token(token, state);
  }

  std::vector<int32_t> generated;
  generated.reserve(static_cast<std::size_t>(decode_tokens));
  for (int64_t i = 0; i < decode_tokens; ++i) {
    const int32_t next = minimind::model::argmax_token(logits);
    generated.push_back(next);
    logits = model.forward_token(next, state);
  }
  return generated;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model_dir = arg_value(argc, argv, "--model", "");
  const std::string prompt = arg_value(argc, argv, "--prompt", "hello");
  const std::string explicit_tokens = arg_value(argc, argv, "--input-tokens", "");
  const int64_t decode_tokens = int_arg_value(argc, argv, "--tokens", 16);
  const int64_t warmup = int_arg_value(argc, argv, "--warmup", 1);

  auto model = (!model_dir.empty() && minimind::model::has_runtime_language_model(model_dir))
                   ? minimind::model::load_runtime_language_model(model_dir)
                   : minimind::model::make_toy_language_model();
  const auto tokens = prompt_tokens(prompt, explicit_tokens, model);

  for (int64_t i = 0; i < warmup; ++i) {
    (void)decode_fixed_steps(model, tokens, decode_tokens);
  }

  const auto start = std::chrono::steady_clock::now();
  const auto generated = decode_fixed_steps(model, tokens, decode_tokens);
  const auto end = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(end - start).count();
  const double tokens_per_second = generated.empty() || seconds == 0.0
                                       ? 0.0
                                       : static_cast<double>(generated.size()) / seconds;

  std::cout << "prompt_tokens=" << tokens.size() << '\n';
  std::cout << "generated_tokens=" << generated.size() << '\n';
  std::cout << "seconds=" << seconds << '\n';
  std::cout << "tokens_per_second=" << tokens_per_second << '\n';
  return 0;
}

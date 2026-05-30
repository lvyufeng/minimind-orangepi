#include "../src/csrc/models/minimind/language_model.h"
#include "../src/csrc/models/minimind/runtime_loader.h"
#include "minimind_orangepi/tokenizer.h"

#include <algorithm>
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
  const int32_t first = model.prefill_next_token(input_tokens, state);

  std::vector<int32_t> generated;
  generated.reserve(static_cast<std::size_t>(decode_tokens));
  int32_t next = first;
  for (int64_t i = 0; i < decode_tokens; ++i) {
    generated.push_back(next);
    if (i + 1 < decode_tokens) {
      next = model.forward_next_token(next, state);
    }
  }
  return generated;
}

struct TimingResult {
  double prefill_seconds = 0.0;
  double decode_seconds = 0.0;
  int64_t prefill_tokens = 0;
  int64_t decode_steps = 0;
};

TimingResult timed_decode_fixed_steps(const minimind::model::LanguageModel& model,
                                      const std::vector<int32_t>& input_tokens,
                                      int64_t decode_tokens,
                                      std::vector<int32_t>& generated) {
  auto state = model.make_state();

  const auto prefill_start = std::chrono::steady_clock::now();
  int32_t next = model.prefill_next_token(input_tokens, state);
  const auto prefill_end = std::chrono::steady_clock::now();

  generated.clear();
  generated.reserve(static_cast<std::size_t>(decode_tokens));
  const auto decode_start = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < decode_tokens; ++i) {
    generated.push_back(next);
    if (i + 1 < decode_tokens) {
      next = model.forward_next_token(next, state);
    }
  }
  const auto decode_end = std::chrono::steady_clock::now();

  TimingResult timing;
  timing.prefill_seconds = std::chrono::duration<double>(prefill_end - prefill_start).count();
  timing.decode_seconds = std::chrono::duration<double>(decode_end - decode_start).count();
  timing.prefill_tokens = static_cast<int64_t>(input_tokens.size());
  timing.decode_steps = std::max<int64_t>(0, decode_tokens - 1);
  return timing;
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

  std::vector<int32_t> generated;
  const auto timing = timed_decode_fixed_steps(model, tokens, decode_tokens, generated);
  const double decode_tokens_per_second = timing.decode_steps == 0 || timing.decode_seconds == 0.0
                                              ? 0.0
                                              : static_cast<double>(timing.decode_steps) / timing.decode_seconds;
  const double prefill_tokens_per_second = timing.prefill_tokens == 0 || timing.prefill_seconds == 0.0
                                               ? 0.0
                                               : static_cast<double>(timing.prefill_tokens) / timing.prefill_seconds;

  std::cout << "prompt_tokens=" << tokens.size() << '\n';
  std::cout << "generated_tokens=" << generated.size() << '\n';
  std::cout << "prefill_tokens=" << timing.prefill_tokens << '\n';
  std::cout << "prefill_seconds=" << timing.prefill_seconds << '\n';
  std::cout << "prefill_tokens_per_second=" << prefill_tokens_per_second << '\n';
  std::cout << "decode_steps=" << timing.decode_steps << '\n';
  std::cout << "decode_seconds=" << timing.decode_seconds << '\n';
  std::cout << "decode_tokens_per_second=" << decode_tokens_per_second << '\n';
  return 0;
}

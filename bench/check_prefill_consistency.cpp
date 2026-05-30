#include "../src/csrc/models/minimind/language_model.h"
#include "../src/csrc/models/minimind/runtime_loader.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: check_prefill_consistency <model-dir> [tokens]\n";
    return 1;
  }
  auto model = minimind::model::load_runtime_language_model(argv[1]);
  int64_t token_count = 8;
  if (argc >= 3) {
    token_count = std::strtoll(argv[2], nullptr, 10);
  }
  if (token_count <= 0) {
    std::cerr << "tokens must be positive\n";
    return 1;
  }
  std::vector<int32_t> prompt;
  prompt.reserve(static_cast<std::size_t>(token_count));
  for (int64_t i = 0; i < token_count; ++i) {
    prompt.push_back(static_cast<int32_t>((i % 1000) + 1));
  }

  auto decode_state = model.make_state();
  int32_t decode_next = 0;
  for (int32_t token : prompt) {
    decode_next = model.forward_next_token(token, decode_state);
  }

  auto prefill_state = model.make_state();
  const int32_t prefill_next = model.prefill_next_token(prompt, prefill_state);
  if (decode_next != prefill_next || decode_state.position != prefill_state.position) {
    std::cerr << "decode_next=" << decode_next << " prefill_next=" << prefill_next
              << " decode_pos=" << decode_state.position << " prefill_pos=" << prefill_state.position << '\n';
    return 1;
  }

  int32_t decode_follow = decode_next;
  int32_t prefill_follow = prefill_next;
  for (int step = 0; step < 4; ++step) {
    decode_follow = model.forward_next_token(decode_follow, decode_state);
    prefill_follow = model.forward_next_token(prefill_follow, prefill_state);
    if (decode_follow != prefill_follow || decode_state.position != prefill_state.position) {
      std::cerr << "step=" << step << " decode=" << decode_follow << " prefill=" << prefill_follow
                << " decode_pos=" << decode_state.position << " prefill_pos=" << prefill_state.position << '\n';
      return 1;
    }
  }

  std::cout << "prefill consistency ok\n";
  return 0;
}

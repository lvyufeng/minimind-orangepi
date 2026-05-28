#include "talker.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace minimind::model_o {

TalkerBoundary::TalkerBoundary(MiniMindOConfig config) : config_(std::move(config)) {
  const auto errors = validate_config(config_);
  if (!errors.empty()) {
    throw std::invalid_argument("invalid MiniMind-O talker config: " + errors.front());
  }
}

bool TalkerBoundary::is_stop_frame(const AudioFrame& frame) const noexcept {
  return std::any_of(frame.codes.begin(), frame.codes.end(), [&](int32_t code) {
    return code >= config_.audio_pad_token;
  });
}

bool TalkerBoundary::is_valid_frame(const AudioFrame& frame) const noexcept {
  return std::all_of(frame.codes.begin(), frame.codes.end(), [&](int32_t code) {
    return code >= 0 && code < config_.audio_pad_token;
  });
}

}  // namespace minimind::model_o

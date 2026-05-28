#pragma once

#include "config.h"

#include <array>
#include <cstdint>
#include <vector>

namespace minimind::model_o {

struct AudioFrame {
  std::array<int32_t, 8> codes{};
};

class TalkerBoundary {
 public:
  explicit TalkerBoundary(MiniMindOConfig config);
  const MiniMindOConfig& config() const noexcept { return config_; }
  bool is_stop_frame(const AudioFrame& frame) const noexcept;
  bool is_valid_frame(const AudioFrame& frame) const noexcept;

 private:
  MiniMindOConfig config_;
};

}  // namespace minimind::model_o

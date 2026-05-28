#include "thinker.h"

#include <stdexcept>
#include <utility>

namespace minimind::model_o {

ThinkerBoundary::ThinkerBoundary(MiniMindOConfig config) : config_(std::move(config)) {
  const auto errors = validate_config(config_);
  if (!errors.empty()) {
    throw std::invalid_argument("invalid MiniMind-O thinker config: " + errors.front());
  }
}

bool ThinkerBoundary::has_audio(const ThinkerInputs& inputs) const noexcept {
  return !inputs.audio_embeddings.empty();
}

bool ThinkerBoundary::has_image(const ThinkerInputs& inputs) const noexcept {
  return !inputs.image_embeddings.empty();
}

}  // namespace minimind::model_o

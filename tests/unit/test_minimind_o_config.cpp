#include "../../src/csrc/models/minimind_o/talker.h"
#include "../../src/csrc/models/minimind_o/thinker.h"

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
  minimind::model_o::MiniMindOConfig config;
  auto errors = minimind::model_o::validate_config(config);
  CHECK(errors.empty());
  CHECK(config.audio_token_id == 16);
  CHECK(config.audio_pad_token == 2049);
  CHECK(config.audio_stop_token == 2050);
  CHECK(config.image_token_id == 12);
  CHECK(config.bridge_layer == 3);

  minimind::model_o::ThinkerBoundary thinker(config);
  CHECK(thinker.bridge_layer() == 3);
  CHECK(!thinker.has_audio({}));
  minimind::model_o::ThinkerInputs inputs;
  inputs.audio_embeddings = {1.0F};
  inputs.image_embeddings = {2.0F};
  CHECK(thinker.has_audio(inputs));
  CHECK(thinker.has_image(inputs));

  minimind::model_o::TalkerBoundary talker(config);
  minimind::model_o::AudioFrame valid{{1, 2, 3, 4, 5, 6, 7, 8}};
  CHECK(talker.is_valid_frame(valid));
  CHECK(!talker.is_stop_frame(valid));
  minimind::model_o::AudioFrame stop{{1, 2, 3, 2049, 5, 6, 7, 8}};
  CHECK(!talker.is_valid_frame(stop));
  CHECK(talker.is_stop_frame(stop));

  config.bridge_layer = 99;
  errors = minimind::model_o::validate_config(config);
  CHECK(!errors.empty());

  return 0;
}

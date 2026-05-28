#pragma once

#include "../minimind/config.h"
#include "../minimind_v/config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace minimind::model_o {

struct MiniMindOConfig {
  minimind::model::MiniMindConfig thinker;
  int64_t num_talker_hidden_layers = 4;
  int64_t talker_hidden_size = 768;
  int32_t audio_token_id = 16;
  std::string audio_special_token = "<|audio_pad|>";
  int64_t audio_hidden_size = 512;
  int64_t audio_vocab_size = 2112;
  int32_t audio_pad_token = 2049;
  int32_t audio_stop_token = 2050;
  int32_t audio_spk_token = 2051;
  int64_t spk_emb_size = 192;
  std::vector<int32_t> think_end_ids = {26, 234, 234};
  int32_t image_token_id = 12;
  std::string image_special_token = "<|image_pad|>";
  int64_t image_hidden_size = 768;
  int64_t image_token_len = 64;
  int64_t bridge_layer = 3;
};

std::vector<std::string> validate_config(const MiniMindOConfig& config);

}  // namespace minimind::model_o

#pragma once

#include "../minimind/config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace minimind::model_v {

struct MiniMindVConfig {
  minimind::model::MiniMindConfig text;
  std::string image_special_token = "<|image_pad|>";
  int32_t image_token_id = 12;
  int64_t image_hidden_size = 768;
  int64_t image_token_len = 64;
  int64_t image_size = 256;
  int64_t patch_size = 32;
};

std::vector<std::string> validate_config(const MiniMindVConfig& config);

}  // namespace minimind::model_v

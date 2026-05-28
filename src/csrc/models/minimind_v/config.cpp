#include "config.h"

namespace minimind::model_v {

std::vector<std::string> validate_config(const MiniMindVConfig& config) {
  auto errors = minimind::model::validate_config(config.text);
  if (config.image_special_token.empty()) {
    errors.push_back("image_special_token must not be empty");
  }
  if (config.image_token_id < 0) {
    errors.push_back("image_token_id must be non-negative");
  }
  if (config.image_hidden_size <= 0) {
    errors.push_back("image_hidden_size must be positive");
  }
  if (config.image_token_len != 64) {
    errors.push_back("MiniMind-V currently expects exactly 64 image tokens");
  }
  if (config.image_size != 256) {
    errors.push_back("SigLIP2 preprocessing currently expects 256x256 images");
  }
  if (config.patch_size <= 0 || config.image_size % config.patch_size != 0) {
    errors.push_back("patch_size must divide image_size");
  }
  return errors;
}

}  // namespace minimind::model_v

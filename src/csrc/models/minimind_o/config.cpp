#include "config.h"

namespace minimind::model_o {

std::vector<std::string> validate_config(const MiniMindOConfig& config) {
  auto errors = minimind::model::validate_config(config.thinker);
  if (config.num_talker_hidden_layers <= 0) {
    errors.push_back("num_talker_hidden_layers must be positive");
  }
  if (config.talker_hidden_size <= 0) {
    errors.push_back("talker_hidden_size must be positive");
  }
  if (config.audio_token_id < 0) {
    errors.push_back("audio_token_id must be non-negative");
  }
  if (config.audio_special_token.empty()) {
    errors.push_back("audio_special_token must not be empty");
  }
  if (config.audio_hidden_size <= 0) {
    errors.push_back("audio_hidden_size must be positive");
  }
  if (config.audio_vocab_size <= 0) {
    errors.push_back("audio_vocab_size must be positive");
  }
  if (config.audio_pad_token < 0 || config.audio_stop_token < 0 || config.audio_spk_token < 0) {
    errors.push_back("audio special token ids must be non-negative");
  }
  if (config.spk_emb_size <= 0) {
    errors.push_back("spk_emb_size must be positive");
  }
  if (config.think_end_ids.empty()) {
    errors.push_back("think_end_ids must not be empty");
  }
  if (config.image_token_id < 0) {
    errors.push_back("image_token_id must be non-negative");
  }
  if (config.image_special_token.empty()) {
    errors.push_back("image_special_token must not be empty");
  }
  if (config.image_hidden_size <= 0) {
    errors.push_back("image_hidden_size must be positive");
  }
  if (config.image_token_len != 64) {
    errors.push_back("MiniMind-O currently expects 64 image tokens");
  }
  if (config.bridge_layer < 0 || config.bridge_layer >= config.thinker.num_hidden_layers) {
    errors.push_back("bridge_layer must reference a thinker layer");
  }
  return errors;
}

}  // namespace minimind::model_o

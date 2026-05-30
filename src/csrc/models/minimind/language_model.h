#pragma once

#include "decoder_layer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace minimind::model {

struct AscendModelWeights;

struct DenseModelWeights {
  std::vector<float> embed_tokens;
  std::vector<DenseLayerWeights> layers;
  std::vector<float> final_norm;
  std::vector<float> lm_head;
  mutable std::shared_ptr<AscendModelWeights> ascend;
};

struct DecoderState {
  std::vector<LayerKvCache> layers;
  int64_t position = 0;
};

class LanguageModel {
 public:
  LanguageModel(MiniMindConfig config, DenseModelWeights weights);

  const MiniMindConfig& config() const noexcept { return config_; }
  DecoderState make_state() const;
  std::vector<float> forward_token(int32_t token, DecoderState& state) const;
  int32_t forward_next_token(int32_t token, DecoderState& state) const;
  int32_t prefill_next_token(const std::vector<int32_t>& prompt_tokens, DecoderState& state) const;
  void generate_stream(const std::vector<int32_t>& prompt_tokens,
                       int64_t max_new_tokens,
                       const std::function<void(int32_t)>& on_token) const;
  std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens,
                                int64_t max_new_tokens) const;

 private:
#if defined(MINIMIND_USE_ASCEND)
  const AscendModelWeights& ascend_weights() const;
#endif

  MiniMindConfig config_;
  DenseModelWeights weights_;
};

LanguageModel make_toy_language_model();
LanguageModel make_toy_moe_language_model();
int32_t argmax_token(const std::vector<float>& logits);

}  // namespace minimind::model

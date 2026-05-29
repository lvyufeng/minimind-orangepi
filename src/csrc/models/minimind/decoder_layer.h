#pragma once

#include "ascend_custom_ops.h"
#include "config.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace minimind::model {

struct ExpertWeights {
  std::vector<float> gate_proj;
  std::vector<float> up_proj;
  std::vector<float> down_proj;
  std::vector<float> gate_up_proj;
};

struct DenseLayerWeights {
  std::vector<float> input_norm;
  std::vector<float> post_attention_norm;
  std::vector<float> q_norm;
  std::vector<float> k_norm;
  std::vector<float> q_proj;
  std::vector<float> k_proj;
  std::vector<float> v_proj;
  std::vector<float> qkv_proj;
  std::vector<float> o_proj;
  std::vector<float> gate_proj;
  std::vector<float> up_proj;
  std::vector<float> gate_up_proj;
  std::vector<float> down_proj;
  std::vector<float> moe_gate;
  std::vector<ExpertWeights> experts;
};

struct LayerKvCache {
  std::vector<float> keys;
  std::vector<float> values;
  std::shared_ptr<CustomAttentionCache> custom_attention_cache;
  int64_t tokens = 0;
};

std::vector<float> run_dense_decoder_layer(
    const MiniMindConfig& config,
    const DenseLayerWeights& weights,
    LayerKvCache& cache,
    const std::vector<float>& hidden,
    int64_t position);

}  // namespace minimind::model

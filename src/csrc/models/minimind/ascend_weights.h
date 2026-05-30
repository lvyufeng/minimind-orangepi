#pragma once

#include "ascend_matmul.h"

#include <vector>

namespace minimind::model {

#if defined(MINIMIND_USE_ASCEND)
struct AscendExpertWeights {
  DeviceMatrix gate_up_proj;
  DeviceMatrix down_proj;
};

struct AscendLayerWeights {
  DeviceTensor input_norm;
  DeviceTensor post_attention_norm;
  DeviceTensor q_norm;
  DeviceTensor k_norm;
  DeviceMatrix qkv_proj;
  DeviceMatrix o_proj;
  DeviceMatrix gate_up_proj;
  DeviceMatrix down_proj;
  DeviceMatrix moe_gate;
  std::vector<AscendExpertWeights> experts;
};

struct AscendModelWeights {
  DeviceTensor embed_tokens;
  std::vector<AscendLayerWeights> layers;
  DeviceTensor final_norm;
  DeviceMatrix lm_head;
};
#endif

}  // namespace minimind::model

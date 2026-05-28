#pragma once

#include "config.h"

#include <vector>

namespace minimind::model_v {

struct ProjectorWeights {
  std::vector<float> norm_weight;
  std::vector<float> norm_bias;
  std::vector<float> fc1_weight;
  std::vector<float> fc1_bias;
  std::vector<float> fc2_weight;
  std::vector<float> fc2_bias;
};

std::vector<float> run_projector(const MiniMindVConfig& config,
                                 const ProjectorWeights& weights,
                                 const std::vector<float>& image_features);

ProjectorWeights make_identity_projector_weights(const MiniMindVConfig& config);

}  // namespace minimind::model_v
